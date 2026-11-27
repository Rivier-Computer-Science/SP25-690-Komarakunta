"""Metrics with 95% bootstrap confidence intervals, computed from saved predictions.

Usage (from the project folder):
    python src/metrics.py val --n 150      # development numbers (validation sets)
    python src/metrics.py test --n 400     # final numbers (only after the test run)

Resampling unit: the SENTENCE. Words in one sentence are not independent, so whole
sentences are drawn with replacement. Formal and casual sentences are different
sets, so they are resampled separately; the same resample is used for every model,
which makes the model-vs-model comparisons paired.

Primary metric: top-3 accuracy over ALL scored words (out-of-vocabulary targets
count as misses for the n-gram models). Diagnostics: accuracy and perplexity on
in-vocabulary targets only, and the perplexity ratio casual / formal.
"""
import argparse
import csv
import json

import numpy as np
import yaml

from ngram import ROOT

MODELS = ("bigram", "trigram", "gpt2")


def load(path):
    """Per-sentence sums, so a bootstrap resample is just a re-sum of sentences."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    ids = {}
    sent = np.array([ids.setdefault(r["sentence_id"], len(ids)) for r in rows])
    S = len(ids)
    inv = np.array([r["in_vocab"] == "True" for r in rows])

    def per_sentence(values):
        return np.bincount(sent, weights=values.astype(float), minlength=S)

    d = {"S": S, "n": per_sentence(np.ones(len(rows))), "n_inv": per_sentence(inv)}
    for m in MODELS:
        hit = np.array([r["target"] in r[f"{m}_top3"].split() for r in rows])
        lp = np.array([float(r[f"{m}_logprob"]) for r in rows])
        d[m] = {"hit": per_sentence(hit), "hit_inv": per_sentence(hit & inv),
                "lp_inv": per_sentence(np.where(inv, lp, 0.0))}
    return d


def stats(d, m, idx):
    n, n_inv = d["n"][idx].sum(), d["n_inv"][idx].sum()
    return {"acc": d[m]["hit"][idx].sum() / n,
            "acc_inv": d[m]["hit_inv"][idx].sum() / n_inv,
            "ppl": float(np.exp(-d[m]["lp_inv"][idx].sum() / n_inv))}


def ci(values, level):
    lo, hi = np.percentile(values, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["val", "test"])
    ap.add_argument("--n", type=int, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
    B, level = cfg["evaluation"]["bootstrap_resamples"], cfg["evaluation"]["confidence"]
    pred = ROOT / "results" / "predictions"
    F = load(pred / f"formal_{args.kind}_n{args.n}.csv")
    C = load(pred / f"casual_{args.kind}_n{args.n}.csv")
    print(f"{args.kind}: {F['S']} formal sentences ({int(F['n'].sum())} words), "
          f"{C['S']} casual sentences ({int(C['n'].sum())} words), {B} resamples")

    allF, allC = np.arange(F["S"]), np.arange(C["S"])
    point = {m: (stats(F, m, allF), stats(C, m, allC)) for m in MODELS}
    rng = np.random.default_rng(cfg["seed"])
    reps = []
    for _ in range(B):
        fi = rng.integers(0, F["S"], F["S"])
        ci_ = rng.integers(0, C["S"], C["S"])
        reps.append({m: (stats(F, m, fi), stats(C, m, ci_)) for m in MODELS})

    def series(m, which, key):
        return np.array([r[m][which][key] for r in reps])

    def drop(m, key):
        return series(m, 0, key) - series(m, 1, key)

    out = {"kind": args.kind, "sentences_formal": F["S"], "sentences_casual": C["S"],
           "words_formal": int(F["n"].sum()), "words_casual": int(C["n"].sum()),
           "resamples": B, "confidence": level, "models": {}, "vs_bigram": {}}
    for m in MODELS:
        pf, pc = point[m]
        out["models"][m] = {
            "acc_formal": [pf["acc"], *ci(series(m, 0, "acc"), level)],
            "acc_casual": [pc["acc"], *ci(series(m, 1, "acc"), level)],
            "drop": [pf["acc"] - pc["acc"], *ci(drop(m, "acc"), level)],
            "acc_inv_formal": [pf["acc_inv"], *ci(series(m, 0, "acc_inv"), level)],
            "acc_inv_casual": [pc["acc_inv"], *ci(series(m, 1, "acc_inv"), level)],
            "drop_inv": [pf["acc_inv"] - pc["acc_inv"], *ci(drop(m, "acc_inv"), level)],
            "ppl_formal": [pf["ppl"], *ci(series(m, 0, "ppl"), level)],
            "ppl_casual": [pc["ppl"], *ci(series(m, 1, "ppl"), level)],
            "ppl_ratio": [pc["ppl"] / pf["ppl"],
                          *ci(series(m, 1, "ppl") / series(m, 0, "ppl"), level)],
        }
    for m in ("trigram", "gpt2"):
        diff = drop(m, "acc") - drop("bigram", "acc")
        pt = out["models"][m]["drop"][0] - out["models"]["bigram"]["drop"][0]
        lo, hi = ci(diff, level)
        verdict = ("drop significantly SMALLER than the bigram's (interval excludes 0)" if hi < 0 else
                   "drop significantly LARGER than the bigram's (interval excludes 0)" if lo > 0 else
                   "no significant difference from the bigram's drop (interval includes 0)")
        out["vs_bigram"][m] = {"drop_difference": [pt, lo, hi], "verdict": verdict}

    def f3(v):
        return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"

    def f1(v):
        return f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]"

    pct = int(level * 100)
    print(f"\nTop-3 accuracy, all words ({pct}% CI)")
    print(f"{'model':<9}{'formal':<22}{'casual':<22}{'drop (formal - casual)'}")
    for m in MODELS:
        r = out["models"][m]
        print(f"{m:<9}{f3(r['acc_formal']):<22}{f3(r['acc_casual']):<22}{f3(r['drop'])}")
    print(f"\nTop-3 accuracy, in-vocabulary targets only ({pct}% CI)")
    for m in MODELS:
        r = out["models"][m]
        print(f"{m:<9}{f3(r['acc_inv_formal']):<22}{f3(r['acc_inv_casual']):<22}{f3(r['drop_inv'])}")
    print(f"\nPerplexity per word, in-vocabulary targets ({pct}% CI)")
    print(f"{'model':<9}{'formal':<24}{'casual':<26}{'ratio casual/formal'}")
    for m in MODELS:
        r = out["models"][m]
        print(f"{m:<9}{f1(r['ppl_formal']):<24}{f1(r['ppl_casual']):<26}{f3(r['ppl_ratio'])}")
    print("\nSurvival test: drop of each model minus the bigram's drop (negative = smaller drop)")
    for m in ("trigram", "gpt2"):
        v = out["vs_bigram"][m]
        print(f"{m:<9}{f3(v['drop_difference'])}  -> {v['verdict']}")

    path = ROOT / "results" / f"metrics_{args.kind}_n{args.n}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved {path.name}")


if __name__ == "__main__":
    main()