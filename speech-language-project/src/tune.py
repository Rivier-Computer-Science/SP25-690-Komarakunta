"""Tune the n-gram smoothing on the FORMAL validation set only.

Nothing here touches a test set, and the casual data is not used for tuning, so the
n-gram systems stay what they would be in practice: built and tuned on formal text.
Criterion: per-word perplexity on in-vocabulary validation words. (Under add-k the
top-3 ranking does not depend on k, so accuracy cannot be used to choose k.)
"""
import csv
import json
import random

import numpy as np
import yaml

from evaluate import bigram_logprobs, ngram_features, perplexity, trigram_logprobs
from ngram import BigramLM, NGramCounts, TrigramLM, ROOT, read_sentences

K_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
L1_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]   # unigram weight
L3_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # trigram weight


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
    random.seed(cfg["seed"])
    proc = ROOT / "data" / "processed"
    counts = NGramCounts(read_sentences(proc / "train_formal.jsonl"),
                         cfg["protocol"]["min_word_count"])
    val = read_sentences(proc / "formal_val.jsonl")
    f = ngram_features(counts, val)
    inv = f["in_vocab"]
    print(f"formal validation: {len(f['w']):,} scored words, "
          f"{inv.mean():.1%} in vocabulary (OOV rate {1 - inv.mean():.1%})")

    rows = []                                          # (model, setting, perplexity)
    for k in K_GRID:
        rows.append(("bigram", {"k": k}, perplexity(bigram_logprobs(f, counts.V, k), inv)))
    for l1 in L1_GRID:
        for l3 in L3_GRID:
            l2 = round(1 - l1 - l3, 4)
            if l2 >= 0.01:
                lam = (l1, l2, l3)
                rows.append(("trigram", {"lambdas": lam}, perplexity(trigram_logprobs(f, lam), inv)))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / "tuning_grid.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["model", "setting", "val_perplexity_in_vocab"])
        for m, s, p in rows:
            wr.writerow([m, json.dumps(s), f"{p:.3f}"])

    best = {m: min((r for r in rows if r[0] == m), key=lambda r: r[2]) for m in ("bigram", "trigram")}
    k, lam = best["bigram"][1]["k"], tuple(best["trigram"][1]["lambdas"])
    print("\nBigram add-k (validation perplexity):")
    for m, s, p in rows:
        if m == "bigram":
            print(f"  k={s['k']:<6} {p:10.1f}{'   <- best' if s['k'] == k else ''}")
    print("\nTrigram, five best interpolation weights (unigram, bigram, trigram):")
    for m, s, p in sorted((r for r in rows if r[0] == "trigram"), key=lambda r: r[2])[:5]:
        print(f"  {s['lambdas']}  {p:10.1f}")
    if k in (K_GRID[0], K_GRID[-1]):
        print("\nWARNING: best k is at the edge of the grid; extend K_GRID.")
    if lam[0] in (L1_GRID[0], L1_GRID[-1]) or lam[2] in (L3_GRID[0], L3_GRID[-1]):
        print("\nWARNING: best weights are at the edge of the grid; extend the grid.")

    # Check 1: the fast array scoring above must equal the real model classes.
    bi, tri = BigramLM(counts, k), TrigramLM(counts, lam)
    lp_bi, lp_tri = bigram_logprobs(f, counts.V, k), trigram_logprobs(f, lam)
    picks = random.sample(range(len(f["w"])), 300)
    for j in picks:
        toks, i = val[f["sid"][j]], f["pid"][j]
        assert np.isclose(bi.logprob(toks[:i], toks[i]), lp_bi[j]), "bigram mismatch"
        assert np.isclose(tri.logprob(toks[:i], toks[i]), lp_tri[j]), "trigram mismatch"
    print("\nConsistency check passed: fast scoring equals the model classes (300 positions).")

    # Check 2: a first look at top-3 accuracy on a validation SAMPLE (not a final result).
    sample = random.sample(range(len(f["w"])), min(3000, len(f["w"])))
    acc = {}
    for name, model in (("bigram", bi), ("trigram", tri)):
        hits = [val[f["sid"][j]][f["pid"][j]] in model.top_k(val[f["sid"][j]][:f["pid"][j]])
                for j in sample]
        acc[name] = float(np.mean(hits))
    print(f"Top-3 accuracy on {len(sample):,} sampled validation positions: "
          f"bigram {acc['bigram']:.3f}, trigram {acc['trigram']:.3f}")

    with open(out / "tuned_params.json", "w", encoding="utf-8") as fh:
        json.dump({"tuned_on": "formal_val (in-vocabulary perplexity)",
                   "bigram": {"k": k, "val_perplexity": best["bigram"][2]},
                   "trigram": {"lambdas": list(lam), "val_perplexity": best["trigram"][2]},
                   "val_oov_rate": float(1 - inv.mean()),
                   "val_top3_sample": acc}, fh, indent=2)
    print("Saved results/tuned_params.json and results/tuning_grid.csv")


if __name__ == "__main__":
    main()