# eval_mistake_id.py
import argparse, json, os, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset

EXPECTED = {"Yes", "To some extent", "No"}

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


def extract_xml_answer(text: str) -> str:
    m = re.search(
        r"<answer>\s*(Yes|To\s+some\s+extent|No)\s*</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return ""
    t = re.sub(r"\s+", " ", m.group(1)).strip().lower()
    return (
        "Yes"
        if t.startswith("yes")
        else ("To some extent" if "some extent" in t else "No")
    )


def compute_format_ok(s: str) -> bool:
    return bool(
        re.search(
            r"<answer>\s*(Yes|To\s+some\s+extent|No)\s*</answer>",
            s,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )


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
        return self.data[idx]


def make_user_block(conv: str, resp: str) -> str:
    return (
        "=== DIALOGUE CONTEXT ===\n"
        f"{conv.strip()}\n\n"
        "=== CANDIDATE TUTOR RESPONSE TO EVALUATE ===\n"
        f"{resp.strip()}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to saved checkpoint dir")
    ap.add_argument("--test_path", default="dataset/test.json")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    mdl.eval()

    ds = MistakeIDDataset(args.test_path)
    loader = DataLoader(ds, batch_size=args.batch_size)

    correct = 0
    total = 0
    fmt_ok = 0

    with torch.no_grad():
        for batch in loader:
            chats = []
            for ex in batch:
                chats.append(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": make_user_block(
                                ex["conversation_history"], ex["candidate_response"]
                            ),
                        },
                    ]
                )
            prompts = [
                tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                for c in chats
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(
                device
            )
            out = mdl.generate(
                **enc,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=32,
                eos_token_id=tok.eos_token_id,
            )
            # slice to completion
            prompt_len = enc["input_ids"].shape[1]
            comp_ids = out[:, prompt_len:]
            texts = tok.batch_decode(comp_ids, skip_special_tokens=True)
            preds = [extract_xml_answer(t) for t in texts]
            fmt_ok += sum(compute_format_ok(t) for t in texts)
            golds = [ex["label"] for ex in batch]
            correct += sum(p == g for p, g in zip(preds, golds))
            total += len(golds)

    acc = correct / total if total else 0.0
    fr = fmt_ok / total if total else 0.0
    print(f"Test size: {total}")
    print(f"Exact-match accuracy: {acc:.4f}")
    print(f"Format rate: {fr:.4f}")


if __name__ == "__main__":
    main()
