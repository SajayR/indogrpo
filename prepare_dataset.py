# prepare_dataset.py
import json, argparse, random, os, re
from typing import List, Dict
from sklearn.model_selection import train_test_split

CANON = {"yes": "Yes", "no": "No", "to some extent": "To some extent"}


def normalize_label(s: str) -> str | None:
    if s is None:
        return None
    t = re.sub(r"\s+", " ", s.strip()).lower()
    # handle common variants
    if t.startswith("yes"):
        return "Yes"
    if t.startswith("no"):
        return "No"
    if "some extent" in t or t.startswith("partial"):
        return "To some extent"
    return CANON.get(t)


def flatten(items: List[Dict]) -> List[Dict]:
    out = []
    for item in items:
        conv = item.get("conversation_history", "").strip()
        tr = item.get("tutor_responses", {}) or {}
        for _, resp in tr.items():
            r_text = (resp or {}).get("response", "")
            ann = (resp or {}).get("annotation", {}) or {}
            label_raw = ann.get("Mistake_Identification", None)
            label = normalize_label(label_raw)
            if not conv or not r_text or label is None:
                continue
            out.append(
                {
                    "conversation_history": conv,
                    "candidate_response": r_text.strip(),
                    "label": label,
                }
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/trainset.json")
    ap.add_argument("--outdir", default="dataset")
    ap.add_argument("--test_size", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = flatten(data)
    if not examples:
        raise SystemExit("No usable examples parsed. Check input format/keys.")

    # Extract labels for stratified split
    labels = [ex["label"] for ex in examples]

    # Perform stratified split
    train_set, test_set = train_test_split(
        examples, test_size=args.test_size, stratify=labels, random_state=args.seed
    )

    with open(os.path.join(args.outdir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(train_set)} train and {len(test_set)} test examples to '{args.outdir}'."
    )


if __name__ == "__main__":
    main()
