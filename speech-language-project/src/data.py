"""Load WikiText-103 and TweetEval, normalize, split into sentences,
remove leakage, and save processed splits with stable example IDs."""
import json
from pathlib import Path

import yaml
from datasets import load_dataset

from normalize import tokenize, is_word

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
OUT = ROOT / "data" / "processed"
RES = ROOT / "results"
TERMINATORS = {".", "!", "?"}
MIN_WORDS = CFG["protocol"]["min_sentence_words"]


def n_words(tokens):
    return sum(is_word(t) for t in tokens)


def split_sentences(tokens):
    """Split at . ! ? (a run like '!!!' ends one sentence, not three)."""
    out, cur = [], []
    for i, t in enumerate(tokens):
        cur.append(t)
        last = i + 1 == len(tokens)
        if t in TERMINATORS and (last or tokens[i + 1] not in TERMINATORS):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def process_split(dataset, prefix, skip_headings=False, max_words=None):
    """Return [{'id', 'text'}] with one normalized sentence per row.
    IDs are <prefix>_<source row>_<sentence index>, so they trace back to the source."""
    rows, total_words = [], 0
    for i, ex in enumerate(dataset):
        raw = ex["text"]
        if skip_headings:
            s = raw.strip()
            if not s or (s.startswith("=") and s.endswith("=")):
                continue
        for k, sent in enumerate(split_sentences(tokenize(raw))):
            w = n_words(sent)
            if w < MIN_WORDS:
                continue
            rows.append({"id": f"{prefix}_{i}_{k}", "text": " ".join(sent)})
            total_words += w
        if max_words and total_words >= max_words:
            break
    return rows


def remove_overlap(rows, blocked):
    """Drop sentences found in `blocked` (leakage) and repeats within the split."""
    kept, seen, overlap, dups = [], set(), 0, 0
    for r in rows:
        if r["text"] in blocked:
            overlap += 1
        elif r["text"] in seen:
            dups += 1
        else:
            seen.add(r["text"])
            kept.append(r)
    return kept, overlap, dups


def save_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(rows, overlap=0, dups=0):
    words = sum(n_words(r["text"].split()) for r in rows)
    return {"sentences": len(rows), "words": words,
            "removed_train_or_earlier_overlap": overlap,
            "removed_duplicates": dups}


def main():
    d = CFG["data"]
    print("Loading datasets (first run downloads WikiText-103, a few hundred MB)...")
    wt = load_dataset(d["formal_repo"], d["formal"])
    tw = load_dataset(d["casual"], d["casual_config"])

    train = process_split(wt["train"], "wt_train", skip_headings=True,
                          max_words=d["train_subset_words"])
    train_texts = {r["text"] for r in train}
    save_jsonl(train, OUT / "train_formal.jsonl")
    stats = {"train_formal": summarize(train)}

    specs = [  # name, dataset, id prefix, skip headings, earlier splits to also block
        ("formal_val", wt["validation"], "wt_val", True, []),
        ("formal_test", wt["test"], "wt_test", True, ["formal_val"]),
        ("casual_val", tw[d["casual_tuning_split"]], "tw_val", False, []),
        ("casual_test", tw[d["casual_test_split"]], "tw_test", False, ["casual_val"]),
    ]
    kept_texts = {}
    (RES / "split_ids").mkdir(parents=True, exist_ok=True)
    for name, ds, prefix, headings, also_block in specs:
        rows = process_split(ds, prefix, headings)
        blocked = set(train_texts)
        for b in also_block:
            blocked |= kept_texts[b]
        kept, overlap, dups = remove_overlap(rows, blocked)
        kept_texts[name] = {r["text"] for r in kept}
        save_jsonl(kept, OUT / f"{name}.jsonl")
        with open(RES / "split_ids" / f"{name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(r["id"] for r in kept) + "\n")
        stats[name] = summarize(kept, overlap, dups)

    with open(RES / "data_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"{'split':<14}{'sentences':>10}{'words':>10}{'overlap':>9}{'dups':>7}")
    for name, s in stats.items():
        print(f"{name:<14}{s['sentences']:>10}{s['words']:>10}"
              f"{s['removed_train_or_earlier_overlap']:>9}{s['removed_duplicates']:>7}")


if __name__ == "__main__":
    main()