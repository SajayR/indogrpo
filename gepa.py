api_key = ""
import os
import dspy

lm = dspy.LM("gpt-5-mini-2025-08-07", temperature=1, api_key=api_key, max_tokens=32000)
dspy.configure(lm=lm)
import dspy
from datasets import load_dataset

from collections import Counter
import numpy as np


def upsample_by_ratio(
    ds,
    label_col="answer",
    base_label="Yes",
    ratios={"Yes": 1.0, "No": 1.0, "To some extent": 1.0},
    seed=3407,
    shuffle=True,
):
    """
    Upsample minority classes to a target ratio vs the base_label count.
    - ratios are multipliers relative to count(base_label).
    - We do NOT downsample any class (we keep all original rows).
    """
    labels = np.array(ds[label_col])
    rng = np.random.RandomState(seed)

    # indices for each label
    uniq = sorted(set(labels.tolist()))
    idxs = {lab: np.where(labels == lab)[0] for lab in uniq}

    base_n = len(idxs[base_label])
    all_pick = []

    for lab, lab_idx in idxs.items():
        target_n = int(round(base_n * ratios.get(lab, 1.0)))
        n = len(lab_idx)

        # Never downsample — just keep originals if target <= n
        if target_n <= n:
            chosen = lab_idx
        else:
            k = target_n // n
            r = target_n % n
            parts = [lab_idx] * k
            if r > 0:
                parts.append(rng.choice(lab_idx, size=r, replace=False))
            chosen = np.concatenate(parts)
        all_pick.append(chosen)

    pick = np.concatenate(all_pick)
    if shuffle:
        rng.shuffle(pick)
    return ds.select(pick.tolist())


def init_dataset():
    # Load your single dataset with 'train' split
    ds = load_dataset("SajayR/MI_chat_dataset")[
        "train"
    ]  # Replace with your dataset name
    dataset = upsample_by_ratio(
        ds,
        "label",
        "Yes",
        ratios={"Yes": 1.0, "No": 0.8, "To some extent": 0.8},
    )
    print("After :", Counter(dataset["label"]))
    # Convert to dspy.Example format
    dataset = [
        dspy.Example(
            {
                "problem": x["input_text"],
                "answer": x["label"],
            }
        ).with_inputs("problem")
        for x in dataset
    ]

    import random

    random.Random(0).shuffle(dataset)
    tot_num = len(dataset)

    train_end = int(0.9 * tot_num)
    val_end = train_end + int(0.05 * tot_num)  # 90% + 5% = 95%

    train_set = dataset[:train_end]  # 0 to 90% = 90% of data
    val_set = dataset[train_end:val_end]  # 90% to 95% = 5% of data
    test_set = dataset[val_end:]  # 95% to end = 5% of data

    return train_set, val_set, test_set


train_set, val_set, test_set = init_dataset()

len(train_set), len(val_set), len(test_set)
print("Problem:")
print(train_set[0]["problem"])
print("\n\nAnswer:")
print(train_set[0]["answer"])
from typing import Literal


class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""

    problem = dspy.InputField()
    # answer = dspy.OutputField()

    answer: Literal["Yes", "No", "To some extent"] = dspy.OutputField()


program = dspy.ChainOfThought(GenerateResponse)
# program.predict.signature.instructions
program.predict.signature.instructions = """You are an evaluator. Your task is to judge whether a “Candidate Tutor Response” correctly identifies a student’s mistake in a math tutoring dialogue.

Input you will receive:
- A DIALOGUE CONTEXT: prior tutor–student turns and the math problem.
- A CANDIDATE TUTOR RESPONSE TO EVALUATE: the specific tutor message you must judge.

Your output:
- Return exactly one of these labels with exact capitalization and no extra text: Yes, To some extent, or No.

Scope and focus:
- Base your judgment ONLY on the DIALOGUE CONTEXT and the CANDIDATE TUTOR RESPONSE. Use standard math reasoning; do not rely on outside knowledge.
- Evaluate mistake identification only. Do not judge whether the candidate computed a correct final answer. Minor slips or informal computation in the candidate’s response are acceptable if the mistake identification is accurate.

Core rubric (tuned to examples):
- Yes: The candidate replies in a way that accurately localizes or targets the student’s mistake. This can be explicit (naming the erroneous step, operation, or interpretation) or indirect but specific (steering the student to re-examine the exact faulty equation, step, quantity, base, unit, or requested object). It is sufficient if the candidate:
  - Points to the specific equation/step likely wrong or asks the student to reconsider that specific equation.
  - Identifies wrong operation (e.g., multiplied instead of divided, inverted operands).
  - Clarifies the correct base/quantity for a percentage or ratio, or the correct units/timeframe (e.g., per month vs per year) and directs the student to align with it.
  - Redirects to the precise requested object when the student provided the wrong type of response (e.g., asked for a formula but student gave a number; “let’s review the volume formula” is Yes).
  - Guides the student to connect a computed intermediate value to the correct interpretation (e.g., “60 ÷ 2 = 30—connect that to birthday presents” is Yes).
  - Challenges an unrealistic result while pointing to the exact step/equation producing it (e.g., “take a closer look at 0.1x = 50; does 500 students seem realistic?” is Yes).
  - Emphasizes using the correct base for an offer/discount (e.g., focusing on which item is cheaper by original price or that discounts apply to the cheaper/original price) when that’s the crux.

- To some extent: The candidate partially identifies or hints at the error but lacks clarity or precision. Examples:
  - Asks a relevant but not fully localized question (e.g., “Which pair is cheaper by original price?” without stating that the student applied the discount to the wrong pair).
  - Misidentifies the exact error yet references a closely related core concept (e.g., talking about “half of the original vs remaining” when the actual issue was a small arithmetic sum; still somewhat relevant).
  - Prompts to re-check without specifying the exact mistaken step, but in a way that nudges toward the right area.

- No: The candidate fails to identify the mistake, points to an unrelated issue, endorses incorrect logic, or contradicts the problem’s facts. Use this when the reply is off-topic or provides guidance unrelated to the student’s actual misstep.

Edge-case guidance distilled from examples:
- Incorrect operation or inverted operands (e.g., 10/5 answered as 50): “You multiplied instead of dividing” → Yes.
- When a wrong equation is set (e.g., using total instead of a component, dropping a factor): Pointing back to that equation or the plausibility of the result → Yes.
- “Half” applies to total vs remainder: Explicitly clarifying that base → Yes; if mentioned but not the actual issue → To some extent.
- Unit/base consistency (monthly vs yearly): Steering toward consistent base (e.g., include all monthly expenses when building monthly total) → Yes.
- When asked for a formula and the student gives a number: Redirecting to review the formula → Yes.
- Discount/offer applies to cheaper/original price: Asking which item is cheaper by original price can be To some extent; explicitly stating the misapplication is Yes.

Procedure:
1. Identify the student’s actual mistake(s) from the dialogue (misinterpretation, wrong base/units, arithmetic slip, wrong operation, wrong equation, answering the wrong thing).
2. Examine the candidate response for how specifically it targets that mistake:
   - Explicitly names it or clearly localizes it → Yes.
   - Indirect but clearly directs the student to the exact faulty step/quantity/goal → Yes.
   - Vague but relevant nudge → To some extent.
   - Irrelevant or incorrect identification → No.
3. Output only one label: Yes, To some extent, or No.

Formatting:
- Output must be exactly one of: Yes, To some extent, No.
- No explanations, punctuation, or extra lines.
"""
print(program.predict.signature.instructions)


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    correct_answer = example["answer"]
    try:
        llm_answer = prediction.answer
    except ValueError as e:
        return 0
    return correct_answer == llm_answer


import dspy

evaluate = dspy.Evaluate(
    devset=test_set,
    metric=metric,
    num_threads=6,
    display_table=True,
    display_progress=True,
)

evaluate(program)
print(program.predict.signature.instructions)


def metric_with_feedback(
    example, prediction, trace=None, pred_name=None, pred_trace=None
):
    correct_answer = example["answer"]
    # written_solution = example.get('solution', '')
    try:
        assert prediction.answer in ["Yes", "No", "To some extent"]
        llm_answer = prediction.answer
        # print(llm_answer)
    except ValueError as e:
        feedback_text = f"The final answer must one of the following: 'Yes', 'No', 'To some extent'. You responded with '{prediction.answer}', which was not one of the options. Please ensure your answer is one of the options without any additional text or formatting."
        feedback_text += f" The correct answer is '{correct_answer}'."

        return dspy.Prediction(score=0, feedback=feedback_text)

    score = correct_answer == llm_answer
    task = """The task is to evaluate whether the candidate tutor reply correctly identifies the student's mistake according to the Mistake Identification rubric.

Rules of engagement:
- The judgment must be based solely on the candidate tutor reply text under the header "CANDIDATE TUTOR RESPONSE TO EVALUATE" and the dialogue history.
- Use the following three categories precisely:
    - Yes: The tutor accurately identifies the mistake with high precision.
    - To some extent: The tutor partially identifies the mistake but lacks full accuracy.
    - No: The tutor fails to identify the mistake or provides an incorrect identification.
"""

    feedback_text = ""
    if score == 1:
        feedback_text = (
            f"Your answer is correct. The correct answer is '{correct_answer}'."
        )
    else:
        feedback_text = (
            f"Your answer is incorrect. The correct answer is '{correct_answer}'. "
        )

    feedback_text += f"\n\n{task}"

    return dspy.Prediction(score=score, feedback=feedback_text)


from dspy import GEPA

optimizer = GEPA(
    metric=metric_with_feedback,
    auto="heavy",
    num_threads=8,
    track_stats=True,
    reflection_minibatch_size=8,
    reflection_lm=dspy.LM(
        model="gpt-5-mini-2025-08-07",
        temperature=1.0,
        max_tokens=32000,
        api_key=api_key,
    ),  # gpt-5-2025-08-07
)

optimized_program = optimizer.compile(
    program,
    trainset=train_set,
    valset=val_set,
)
print(optimized_program.predict.signature.instructions)

with open("optimized_prompt.txt", "w") as f:
    f.write(optimized_program.predict.signature.instructions)
optimized_program.save("./dspy_program/program.json", save_program=False)
evaluate(optimized_program)
