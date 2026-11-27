"""Word-level next-word prediction with distilgpt2 (used as-is, no training).

distilgpt2 predicts subword tokens, but the n-gram models predict whole words.
This module turns the neural model into a whole-word predictor so all three models
are scored on the same unit, using these rules:

  * Context: the same normalized tokens the n-gram models see, joined with spaces
    (punctuation like , . ! ? attached to the previous word), with the model's
    end-of-text token in front. The text stays lowercase, as in the protocol.
  * A word starts with a token that is a space plus lowercase letters (" the") and
    continues with tokens made only of letters and apostrophes ("ing", "'t").
    The word is finished when the next token is anything else.
  * Probability of a whole word = product of its subtoken probabilities times the
    probability that the word ENDS there (1 minus the total probability of all
    continuation tokens). Without the ending term, a prefix such as " un" would be
    credited with the probability of every word that starts with "un".
  * Top-3 words: take the most probable word-start tokens, score each as a complete
    word, and expand a candidate with its most probable continuations only while it
    could still reach the top 3. Different spellings of one word are summed.
  * Only lowercase alphabetic words are scored or suggested. Capitalized variants
    (" The") are not counted, which slightly understates the model's probabilities.
"""
import re

import numpy as np

SPACE = "\u0120"                                   # GPT-2's marker for a leading space
START_RE = re.compile(r"^\u0120[a-z]+(?:'[a-z]+)*$")   # " word" (lowercase)
CONT_RE = re.compile(r"^[A-Za-z']*[A-Za-z]$")          # "ing", "'t" (no leading space)
CLOSERS = {",", ".", "!", "?", ";", ":", ")", "]", "}", "%"}


def detokenize(tokens):
    """Join normalized tokens into text; closing punctuation attaches to the word before."""
    text = ""
    for t in tokens:
        if not text:
            text = t
        elif t in CLOSERS:
            text += t
        else:
            text += " " + t
    return text


def _logsumexp(x):
    m = x.max()
    return float(m + np.log(np.exp(x - m).sum()))


class WordPredictor:
    """Whole-word top-k and word log-probability on top of any subword model backend.

    A backend provides: vocab_strings, context_ids(tokens), word_ids(word),
    next_logprobs(list_of_id_lists) and logprobs_at(id_list, positions).
    """

    def __init__(self, backend, top_first=10, top_cont=3, max_tokens=4):
        self.b = backend
        self.top_first, self.top_cont, self.max_tokens = top_first, top_cont, max_tokens
        strs = backend.vocab_strings
        self.start_idx = np.array([i for i, s in enumerate(strs) if START_RE.match(s)])
        self.cont_idx = np.array([i for i, s in enumerate(strs) if CONT_RE.match(s)])
        self.strs = strs

    def _end_logprob(self, row):
        """log P(the word ends here) = log(1 - total probability of continuation tokens)."""
        cont_mass = np.exp(_logsumexp(row[self.cont_idx]))
        return float(np.log(max(1.0 - cont_mass, 1e-12))), cont_mass

    def logprob(self, tokens, target):
        """Natural-log probability that the next word is exactly `target`."""
        ctx = self.b.context_ids(tokens)
        w = self.b.word_ids(target)
        n = len(ctx)
        rows = self.b.logprobs_at(ctx + w, list(range(n - 1, n + len(w))))
        total = sum(float(rows[j][w[j]]) for j in range(len(w)))
        return total + self._end_logprob(rows[-1])[0]

    def _word(self, toks):
        return self.strs[toks[0]][1:] + "".join(self.strs[t] for t in toks[1:])

    def word_scores(self, tokens, n=3):
        """Candidate whole words and their probabilities (before taking the top n)."""
        ctx = self.b.context_ids(tokens)
        lp0 = self.b.next_logprobs([ctx])[0]
        first = self.start_idx[np.argsort(-lp0[self.start_idx])[: self.top_first]]
        frontier = [([int(t)], float(lp0[t])) for t in first]
        words = {}
        for depth in range(self.max_tokens):
            if not frontier:
                break
            rows = self.b.next_logprobs([ctx + toks for toks, _ in frontier])
            nxt = []
            for (toks, lp), row in zip(frontier, rows):
                log_end, cont_mass = self._end_logprob(row)
                w = self._word(toks)
                words[w] = words.get(w, 0.0) + float(np.exp(lp + log_end))
                ranked = sorted(words.values(), reverse=True)
                third = ranked[n - 1] if len(ranked) >= n else 0.0
                bound = float(np.exp(lp)) * cont_mass      # no longer word can beat this
                if depth + 1 < self.max_tokens and bound > third:
                    best = self.cont_idx[np.argsort(-row[self.cont_idx])[: self.top_cont]]
                    nxt.extend((toks + [int(c)], lp + float(row[c])) for c in best)
            frontier = nxt
        return words

    def top_k(self, tokens, n=3):
        scores = self.word_scores(tokens, n)
        return [w for w, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


class HFBackend:
    """distilgpt2 through Hugging Face transformers (CPU)."""

    def __init__(self, model_name="distilgpt2", revision=None, max_context=128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision).eval()
        self.eos = self.tok.eos_token_id
        self.max_context = max_context
        self.V = self.model.config.vocab_size
        self.vocab_strings = [s or "" for s in self.tok.convert_ids_to_tokens(list(range(self.V)))]
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def context_ids(self, tokens):
        ids = self.tok.encode(detokenize(tokens)) if tokens else []
        return [self.eos] + list(ids)[-(self.max_context - 1):]

    def word_ids(self, word):
        return list(self.tok.encode(" " + word))

    def next_logprobs(self, seqs):
        torch = self.torch
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.eos, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, : len(s)] = torch.tensor(s)
            mask[r, : len(s)] = 1
        with torch.no_grad():
            h = self.model.transformer(input_ids=ids, attention_mask=mask).last_hidden_state
            last = h[torch.arange(len(seqs)), torch.tensor([len(s) - 1 for s in seqs])]
            return torch.log_softmax(self.model.lm_head(last).float(), dim=-1).numpy()

    def logprobs_at(self, seq, positions):
        torch = self.torch
        with torch.no_grad():
            h = self.model.transformer(input_ids=torch.tensor([seq])).last_hidden_state[0]
            return torch.log_softmax(self.model.lm_head(h[positions]).float(), dim=-1).numpy()


if __name__ == "__main__":
    import time

    backend = HFBackend()
    pred = WordPredictor(backend)
    print(f"model revision: {backend.revision}")
    print("\nTop-3 whole-word suggestions from distilgpt2:")
    tests = ["the united", "he was born in", "in the", "<user> thanks for", "i love", "",
             "zzzxqy", "\U0001F602", "the quick brown fox jumped over"]
    t0 = time.time()
    for ctx in tests:
        print(f"  {ctx!r:40} {pred.top_k(ctx.split())}")
    print(f"\nsuggestion time: {(time.time() - t0) / len(tests):.2f}s per context")

    print("\nWord-level log-probabilities (higher is more likely), context 'the united':")
    for w in ("states", "kingdom", "banana", "un", "unless"):
        print(f"  {w:10} {pred.logprob('the united'.split(), w):8.2f}")

    # Self-checks on the model wrapper
    a, b = backend.context_ids("the united".split()), backend.context_ids("he was born in the".split())
    batched = backend.next_logprobs([a, b])
    single = [backend.logprobs_at(s, [len(s) - 1])[0] for s in (a, b)]
    diff = max(float(np.abs(batched[i] - single[i]).max()) for i in range(2))
    assert diff < 1e-2, f"batched and single-sequence scoring disagree ({diff})"
    print(f"\nCheck 1 passed: padded batches match single sequences (max diff {diff:.1e}).")
    ctx = "he was born in".split()
    scores = pred.word_scores(ctx)
    top = pred.top_k(ctx)[0]
    print(f"Check 2: top word {top!r}: search log-prob {np.log(scores[top]):.3f}, "
          f"direct log-prob {pred.logprob(ctx, top):.3f} (should be close).")
    print(f"Check 3: total probability of the {len(scores)} candidate words = "
          f"{sum(scores.values()):.3f} (must be at most 1).")