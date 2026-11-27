"""Run all three models on a sample of sentences from one split and save every prediction.

Usage (from the project folder):
    python src/run_eval.py formal_val --n 150
    python src/run_eval.py casual_val --n 150
Test splits need --final and are allowed to run only once.

Every scored word position is one row in results/predictions/<split>_n<N>.csv, so any
reported metric can be recomputed from that file. All models score the same positions.
"""
import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from evaluate import scored_positions
from gpt2_words import HFBackend, WordPredictor
from ngram import BigramLM, NGramCounts, ROOT, TrigramLM, read_sentences

FIELDS = ["pos_id", "sentence_id", "position", "target", "in_vocab", "context",
          "bigram_top3", "trigram_top3", "gpt2_top3",
          "bigram_logprob", "trigram_logprob", "gpt2_logprob"]


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [(r["id"], r["text"].split()) for r in map(json.loads, f)]


def model_revision(name):
    """Commit hash of the cached Hugging Face model files (for reproducibility)."""
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(name, "config.json")).parent.name
    except Exception:
        return None


def lib_version(name):
    try:
        return __import__(name).__version__
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["formal_val", "casual_val", "formal_test", "casual_test"])
    ap.add_argument("--n", type=int, default=150, help="number of sentences to sample")
    ap.add_argument("--final", action="store_true", help="required for test splits")
    args = ap.parse_args()

    out_dir = ROOT / "results" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_n{args.n}.csv"
    if args.split.endswith("_test"):
        if not args.final:
            sys.exit("Test splits are used once. Re-run with --final when the pipeline is frozen.")
        if list(out_dir.glob(f"{args.split}_*.csv")):
            sys.exit(f"{args.split} has already been run. Delete its file only if that run crashed.")

    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
    tuned = json.load(open(ROOT / "results" / "tuned_params.json", encoding="utf-8"))
    k, lam = tuned["bigram"]["k"], tuple(tuned["trigram"]["lambdas"])
    g = cfg["models"]["distilgpt2"]

    counts = NGramCounts(read_sentences(ROOT / "data" / "processed" / "train_formal.jsonl"),
                         cfg["protocol"]["min_word_count"])
    bigram, trigram = BigramLM(counts, k), TrigramLM(counts, lam)
    backend = HFBackend(g["model_id"])
    gpt2 = WordPredictor(backend, top_first=g["first_token_candidates"],
                         top_cont=g["continuations_per_step"], max_tokens=g["max_subtokens"])

    rows = read_rows(ROOT / "data" / "processed" / f"{args.split}.jsonl")
    rng = random.Random(cfg["seed"])
    chosen = sorted(rng.sample(range(len(rows)), min(args.n, len(rows))))
    n_pos = sum(len(scored_positions(rows[i][1])) for i in chosen)
    print(f"{args.split}: {len(chosen)} sentences, {n_pos} scored positions -> {out_path.name}")

    t0 = time.time()
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, FIELDS)
        wr.writeheader()
        bar = tqdm(total=n_pos, unit="word")
        for ri in chosen:
            sid, toks = rows[ri]
            for i in scored_positions(toks):
                ctx, tgt = toks[:i], toks[i]
                wr.writerow({
                    "pos_id": f"{sid}#{i}", "sentence_id": sid, "position": i, "target": tgt,
                    "in_vocab": tgt in counts.stoi, "context": " ".join(ctx),
                    "bigram_top3": " ".join(bigram.top_k(ctx)),
                    "trigram_top3": " ".join(trigram.top_k(ctx)),
                    "gpt2_top3": " ".join(gpt2.top_k(ctx)),
                    "bigram_logprob": round(bigram.logprob(ctx, tgt), 4),
                    "trigram_logprob": round(trigram.logprob(ctx, tgt), 4),
                    "gpt2_logprob": round(gpt2.logprob(ctx, tgt), 4),
                })
                bar.update(1)
            fh.flush()
        bar.close()
    seconds = time.time() - t0

    info = {"split": args.split, "sentences": len(chosen), "positions": n_pos, "seed": cfg["seed"],
            "bigram_k": k, "trigram_lambdas": list(lam), "distilgpt2": g,
            "distilgpt2_revision": model_revision(g["model_id"]),
            "python": platform.python_version(), "torch": lib_version("torch"),
            "transformers": lib_version("transformers"), "numpy": np.__version__,
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "runtime_seconds": round(seconds, 1)}
    with open(out_path.with_suffix(".info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    # Quick orientation only. Real metrics and confidence intervals come from metrics.py.
    data = list(csv.DictReader(open(out_path, encoding="utf-8")))
    inv = np.array([r["in_vocab"] == "True" for r in data])
    print(f"\n{len(data)} positions, OOV rate {1 - inv.mean():.1%}, runtime {seconds / 60:.1f} min")
    print(f"{'model':<10}{'top-3 (all)':>13}{'top-3 (in-vocab)':>18}{'ppl (in-vocab)':>16}")
    for name in ("bigram", "trigram", "gpt2"):
        hit = np.array([r["target"] in r[f"{name}_top3"].split() for r in data])
        lp = np.array([float(r[f"{name}_logprob"]) for r in data])
        print(f"{name:<10}{hit.mean():>13.3f}{hit[inv].mean():>18.3f}{np.exp(-lp[inv].mean()):>16.1f}")


if __name__ == "__main__":
    main()