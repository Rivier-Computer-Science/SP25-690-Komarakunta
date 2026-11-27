"""Next-word suggestions from all three models, for any typed text.

Usage (from the project folder):
    python src/infer.py "the united"      one input, then exit
    python src/infer.py                   interactive mode (type q to quit)
    python src/infer.py --demo            run representative and boundary cases and save
                                          the output to results/inference_demo.txt
The input should end with a complete word. The tool suggests the NEXT word, using the
same normalization and current-sentence rule as the evaluation.
"""
import argparse
import json

import yaml

from gpt2_words import HFBackend, WordPredictor
from ngram import BigramLM, NGramCounts, ROOT, TrigramLM, read_sentences
from normalize import tokenize

MAX_TOKENS = 100
TERMINATORS = {".", "!", "?"}

DEMO_INPUTS = [
    "the united", "he was born in", "i love", "<user> thanks for", "lol that was so",
    "",                                    # empty text
    "   ",                                 # whitespace only
    "\U0001F602\U0001F602\U0001F602",      # emoji only
    "zzzxqy qwerty",                       # unknown words
    "check this out https://t.co/abc123",  # URL
    "the war ended. the",                  # a new sentence has started
    "the war ended.",                      # sentence just ended
    "!!!",                                 # punctuation only
    "the " * 300,                          # very long input
]


def load_models():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
    tuned = json.load(open(ROOT / "results" / "tuned_params.json", encoding="utf-8"))
    g = cfg["models"]["distilgpt2"]
    counts = NGramCounts(read_sentences(ROOT / "data" / "processed" / "train_formal.jsonl"),
                         cfg["protocol"]["min_word_count"])
    gpt2 = WordPredictor(HFBackend(g["model_id"]), top_first=g["first_token_candidates"],
                         top_cont=g["continuations_per_step"], max_tokens=g["max_subtokens"])
    models = {"bigram": BigramLM(counts, tuned["bigram"]["k"]),
              "trigram": TrigramLM(counts, tuple(tuned["trigram"]["lambdas"])),
              "distilgpt2": gpt2}
    return counts, models


def suggest(text, counts, models):
    """Return the lines to print for one input."""
    shown = text if len(text) <= 70 else text[:70] + "..."
    lines = [f"Input      : {shown!r}"]
    tokens = tokenize(text)
    notes = []
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[-MAX_TOKENS:]
        notes.append(f"long input: using the last {MAX_TOKENS} tokens")
    last_end = max((i for i, t in enumerate(tokens) if t in TERMINATORS), default=-1)
    if last_end >= 0:
        context = tokens[last_end + 1:]          # only the current sentence is context
        if not context:
            notes.append("the sentence just ended: suggesting how a new sentence may start")
    else:
        context = tokens
    if not tokens:
        notes.append("nothing typed yet: suggesting how a sentence may start")
    unknown = list(dict.fromkeys(t for t in context if t not in counts.stoi))
    if unknown:
        notes.append("not in the n-gram vocabulary (treated as <unk>): " + ", ".join(unknown[:5]))
    tail = ("... " if len(context) > 12 else "") + " ".join(context[-12:])
    lines.append(f"Context    : {tail if context else '(empty)'}")
    for name, model in models.items():
        try:
            words = ", ".join(model.top_k(context)) or "(no suggestion)"
        except Exception as e:                   # one failing model must not stop the others
            words = f"(error: {type(e).__name__}: {e})"
        lines.append(f"{name:<11}: {words}")
    lines.extend(f"Note       : {n}" for n in notes)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="typed words; omit for interactive mode")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    counts, models = load_models()

    if args.demo:
        out = []
        for text in DEMO_INPUTS:
            out.extend(suggest(text, counts, models) + [""])
        print("\n".join(out))
        path = ROOT / "results" / "inference_demo.txt"
        path.write_text("\n".join(out), encoding="utf-8")
        print(f"Saved {path.name}")
    elif args.text is not None:
        print("\n".join(suggest(args.text, counts, models)))
    else:
        print("Type some words and press Enter (q to quit).")
        while True:
            try:
                text = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                break
            if text.strip().lower() == "q":
                break
            print("\n".join(suggest(text, counts, models)))


if __name__ == "__main__":
    main()