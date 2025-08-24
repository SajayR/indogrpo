# train_grpo_mistake_id.py
import os, re, math, random, argparse, json
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

"""
torchrun --standalone --nproc_per_node=1 train_grpo_mistake_id.py \
  --model_name Qwen/Qwen3-1.7B \
  --train_path dataset/train.json \
  --save_dir outputs/grpo_models \
  --epochs 1 --batch_size 4 --K 4 --ppo_epochs 4
"""

# -----------------------------
# Helpers: parsing & scoring
# -----------------------------
EXPECTED = {"Yes", "To some extent", "No"}


def _normalize_label(s: str) -> str:
    t = re.sub(r"\s+", " ", s).strip().lower()
    if t.startswith("yes"):
        return "Yes"
    if "some extent" in t:
        return "To some extent"
    if t.startswith("no"):
        return "No"
    return s  # fallback (won't match EXPECTED)


def extract_xml_answer(text: str) -> str:
    """
    Pull the label from anywhere inside <answer>...</answer>, case/whitespace tolerant.
    Allows extra text before/after (e.g., <think>...</think>).
    """
    m = re.search(
        r"<answer>\s*(Yes|To\s+some\s+extent|No)\s*</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return _normalize_label(m.group(1)) if m else ""


def compute_format_score(batch_responses):
    """
    Accepts str or list[str]. Returns 1.0/0.0 for a str, or list[float] for a batch.
    """
    pat = re.compile(
        r"<answer>\s*(Yes|To\s+some\s+extent|No)\s*</answer>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    if isinstance(batch_responses, str):
        return 1.0 if pat.search(batch_responses) else 0.0
    return [1.0 if pat.search(r) else 0.0 for r in batch_responses]


def compute_reward(batch_answers, gold_answers):
    """4.0 if the predicted label equals gold, else 0.0."""
    return [4.0 if g_a == a else 0.0 for g_a, a in zip(batch_answers, gold_answers)]


# --------------------------------------
# System prompt (your rubric, verbatim)
# --------------------------------------
SYSTEM_PROMPT = (
    """
You are an evaluator. Your task is to judge whether a "Candidate Tutor Response" correctly identifies a student's mistake in a math tutoring dialogue.

Input you will receive:
- A DIALOGUE CONTEXT: prior tutor–student turns and the math problem.
- A CANDIDATE TUTOR RESPONSE TO EVALUATE: the specific tutor message you must judge.

Your output:
- Return exactly one of these labels with exact capitalization and no extra text: Yes, To some extent, or No enclosed in <answer> tags.

Scope and focus:
- Base your judgment ONLY on the DIALOGUE CONTEXT and the CANDIDATE TUTOR RESPONSE. Use standard math reasoning; do not rely on outside knowledge.
- Evaluate mistake identification only. Do not judge whether the candidate computed a correct final answer. Minor slips or informal computation in the candidate's response are acceptable if the mistake identification is accurate.

Core rubric (tuned to examples):
- Yes: The candidate replies in a way that accurately localizes or targets the student's mistake. This can be explicit (naming the erroneous step, operation, or interpretation) or indirect but specific (steering the student to re-examine the exact faulty equation, step, quantity, base, unit, or requested object). It is sufficient if the candidate:
  - Points to the specific equation/step likely wrong or asks the student to reconsider that specific equation.
  - Identifies wrong operation (e.g., multiplied instead of divided, inverted operands).
  - Clarifies the correct base/quantity for a percentage or ratio, or the correct units/timeframe (e.g., per month vs per year) and directs the student to align with it.
  - Redirects to the precise requested object when the student provided the wrong type of response (e.g., asked for a formula but student gave a number; "let's review the volume formula" is Yes).
  - Guides the student to connect a computed intermediate value to the correct interpretation (e.g., "60 ÷ 2 = 30—connect that to birthday presents" is Yes).
  - Challenges an unrealistic result while pointing to the exact step/equation producing it (e.g., "take a closer look at 0.1x = 50; does 500 students seem realistic?" is Yes).
  - Emphasizes using the correct base for an offer/discount (e.g., focusing on which item is cheaper by original price or that discounts apply to the cheaper/original price) when that's the crux.

- To some extent: The candidate partially identifies or hints at the error but lacks clarity or precision.
- No: The candidate fails to identify the mistake, points to an unrelated issue, endorses incorrect logic, or contradicts the problem's facts.

Procedure:
1. Identify the student's actual mistake(s) from the dialogue.
2. Examine the candidate response for how specifically it targets that mistake:
   - Explicit/localized → Yes.
   - Vague but relevant → To some extent.
   - Irrelevant/incorrect → No.
3. Output only one label: Yes, To some extent, or No enclosed in <answer> tags.

Formatting:
- Output must be exactly one of: Yes, To some extent, No enclosed in <answer> tags. Example: <answer>Yes</answer>
- No explanations, punctuation, or extra lines.
    """
).strip()


# --------------------------------------
# Dataset
# --------------------------------------
class MistakeIDDataset(Dataset):
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data = [
            d
            for d in data
            if d.get("conversation_history")
            and d.get("candidate_response")
            and d.get("label") in EXPECTED
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x["conversation_history"], x["candidate_response"], x["label"]


def make_user_block(conv: str, resp: str) -> str:
    return (
        "=== DIALOGUE CONTEXT ===\n"
        f"{conv.strip()}\n\n"
        "=== CANDIDATE TUTOR RESPONSE TO EVALUATE ===\n"
        f"{resp.strip()}"
    )


def build_collate_fn(tokenizer):
    def collate_fn(batch):
        prompts, labels = [], []
        for conversation, candidate_response, gold in batch:
            chat = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": make_user_block(conversation, candidate_response),
                },
            ]
            prompt = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
            labels.append(gold)
        return prompts, labels

    return collate_fn


# --------------------------------------
# Training utils
# --------------------------------------
def get_lr(it, max_steps, warmup_frac=0.1, max_lr=1e-5, min_lr=1e-6):
    warmup_steps = max(1, int(warmup_frac * max_steps))
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


# --------------------------------------
# Main
# --------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train_path", default="dataset/train.json")
    parser.add_argument("--save_dir", default="outputs/grpo_models")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)  # per GPU
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument(
        "--max_new_tokens", type=int, default=256
    )  # generous for <think>, not insane
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--init_lr", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument(
        "--entropy_coef", type=float, default=0.003
    )  # NEW: exploration without KL
    args = parser.parse_args()

    # DDP init
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    master = rank == 0

    seed = 42 + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, "log.txt")
    if master:
        open(log_file, "w").close()

    # tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    LEAN_TARGETS = ["q_proj", "v_proj"]
    BEEFY_TARGETS = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=BEEFY_TARGETS,
        task_type="CAUSAL_LM",
        inference_mode=False,
    )

    # Base policy + LoRA adapters (trainable), DDP wrap
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    ).to(local_rank)
    policy = get_peft_model(base, lora_cfg)
    policy = DDP(
        policy,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    policy.train()

    # optimizer (LoRA params only, no weight decay typical for LoRA)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    if master:
        n_all = sum(p.numel() for p in policy.parameters())
        n_train = sum(p.numel() for p in trainable_params)
        print(f"🔧 LoRA params: {n_train / n_all:.4%} of total ({n_train}/{n_all})")
    optim = torch.optim.AdamW(
        [{"params": trainable_params, "weight_decay": 0.0}], lr=args.init_lr
    )
    optim.zero_grad(set_to_none=True)

    # data
    train_data = MistakeIDDataset(args.train_path)
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=True
    )
    collate_fn = build_collate_fn(tokenizer)
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    max_steps = len(loader)
    if master:
        print(f"⚙️ Steps per epoch: {max_steps}")

    # training loop
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", disable=not master)

        for step, (prompts, gold_labels) in enumerate(pbar):
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                padding_side="left",
            )
            input_ids = enc["input_ids"].to(local_rank)
            attn = enc["attention_mask"].to(local_rank)
            B = input_ids.shape[0]
            prompt_len = input_ids.shape[1]

            # Generate K samples per prompt
            policy.eval()
            with (
                torch.no_grad(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                generations = policy.module.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    num_return_sequences=args.K,
                    top_p=0.9,
                    temperature=1.0,
                    eos_token_id=tokenizer.eos_token_id,
                )
            policy.train()

            # compose masks & labels
            attn_full = (generations != tokenizer.pad_token_id).long()  # [B*K, T]
            action_mask = attn_full.clone()
            action_mask[:, :prompt_len] = 0
            labels = generations.clone()
            labels[action_mask == 0] = -100

            # logprobs_old under current policy (behavior policy at sampling time)
            policy.eval()
            with (
                torch.no_grad(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                out_old = policy(
                    input_ids=generations, attention_mask=attn_full, use_cache=False
                )
            policy.train()
            logits_old = out_old.logits[:, :-1, :].contiguous()
            labels_old = labels[:, 1:].contiguous()
            nll_old = F.cross_entropy(
                logits_old.view(-1, logits_old.size(-1)),
                labels_old.view(-1),
                reduction="none",
                ignore_index=-100,
            )
            logprobs_old = -nll_old.view(logits_old.size(0), -1).view(B, args.K, -1)

            # rewards/advantages
            resp_ids = generations[:, prompt_len:]  # [B*K, L]
            texts = tokenizer.batch_decode(resp_ids, skip_special_tokens=True)
            preds = [extract_xml_answer(t) for t in texts]
            goldK = [g for g in gold_labels for _ in range(args.K)]

            fmt = compute_format_score(texts)  # 1.0 if tag present
            acc = compute_reward(preds, goldK)  # 4.0 if correct label
            total_reward = torch.tensor(
                [f + a for f, a in zip(fmt, acc)],
                dtype=torch.bfloat16,
                device=local_rank,
            )
            rew = total_reward.view(B, args.K)
            adv = (rew - rew.mean(dim=-1, keepdim=True)) / rew.std(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            adv = adv.unsqueeze(2).expand_as(logprobs_old)  # match [B, K, T-1]

            # PPO update (NO KL, add entropy bonus)
            valid_mask = action_mask[:, :-1].float().view(B, args.K, -1)  # [B,K,T-1]
            for pe in range(args.ppo_epochs):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_new = policy(
                        input_ids=generations, attention_mask=attn_full, use_cache=False
                    )
                logits_new = out_new.logits[:, :-1, :].contiguous()
                labels_new = labels[:, 1:].contiguous()
                nll_new = F.cross_entropy(
                    logits_new.view(-1, logits_new.size(-1)),
                    labels_new.view(-1),
                    reduction="none",
                    ignore_index=-100,
                )
                logprobs_new = -nll_new.view(logits_new.size(0), -1).view(B, args.K, -1)

                ratio = torch.exp(logprobs_new - logprobs_old)
                ratio_clip = torch.clamp(
                    ratio, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip
                )
                ppo_term = torch.min(ratio * adv, ratio_clip * adv)  # [B,K,T-1]

                # Entropy bonus over valid tokens
                # entropy = -sum(p*log p) across vocab
                probs = F.softmax(logits_new, dim=-1)
                log_probs = F.log_softmax(logits_new, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1)  # [B*K, T-1]
                entropy = entropy.view(B, args.K, -1)

                # Aggregate objective per sequence
                masked_ppo = ppo_term * valid_mask
                masked_ent = entropy * valid_mask
                token_counts = valid_mask.sum(dim=-1)  # [B,K]
                # Avoid divide-by-zero if any weirdness
                token_counts = token_counts.clamp_min(1.0)

                seq_obj = masked_ppo.sum(dim=-1) / token_counts + args.entropy_coef * (
                    masked_ent.sum(dim=-1) / token_counts
                )
                loss = -seq_obj.mean()  # maximize objective

                # logging
                loss_val = loss.detach().clone()
                dist.all_reduce(loss_val, op=dist.ReduceOp.AVG)
                if master:
                    pbar.set_postfix({"loss": f"{loss_val.item():.4f}"})
                    with open(log_file, "a") as f:
                        f.write(
                            f"epoch {epoch} step {step} ppo {pe} loss {loss_val.item():.4f}\n"
                        )

                # optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                lr = get_lr(step, max_steps)
                for pg in optim.param_groups:
                    pg["lr"] = lr
                optim.step()
                optim.zero_grad(set_to_none=True)

            # checkpoints
            if master and (step % 50 == 0 or step == max_steps - 1):
                ckpt_dir = os.path.join(
                    args.save_dir, f"ckpt-epoch{epoch + 1}-step{step + 1}"
                )
                os.makedirs(ckpt_dir, exist_ok=True)
                policy.module.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
