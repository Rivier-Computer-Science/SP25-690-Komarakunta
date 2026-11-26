"""Bigram and trigram language models, written from scratch (numpy only).

All counts live in sorted integer arrays, so lookups are fast and memory is small.
Both models share one vocabulary and one set of counts (NGramCounts).

Prediction rules shared by both models (see the evaluation protocol):
  * context = the normalized tokens typed so far (punctuation and <user>/<url>
    placeholders count as context),
  * suggestions are alphabetic in-vocabulary words only (never <unk>, never
    punctuation, never placeholders),
  * a true target that is not in the vocabulary is mapped to <unk> for scoring.
"""
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from normalize import is_word

UNK, EOS = "<unk>", "</s>"
ROOT = Path(__file__).resolve().parent.parent


def read_sentences(path):
    """Read a processed .jsonl split into a list of token lists."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"].split() for line in f]


class NGramCounts:
    """Vocabulary plus unigram, bigram and trigram counts from training sentences.

    Word ids: 0 = <unk>, 1 = </s>, then words from most to least frequent.
    BOS (sentence-start padding) gets id V. It is a context only, never a target.
    """

    def __init__(self, sentences, min_count=3):
        freq = Counter(t for s in sentences for t in s)
        kept = sorted((t for t, c in freq.items() if c >= min_count),
                      key=lambda t: (-freq[t], t))
        self.itos = [UNK, EOS] + kept
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.V = len(self.itos)          # number of possible targets
        self.BOS = self.V
        self.B = self.V + 1              # base used to pack n-grams into one integer

        # one flat stream: BOS BOS w1 w2 ... EOS  BOS BOS w1 ...
        get = self.stoi.get
        stream = []
        for sent in sentences:
            stream.append(self.BOS)
            stream.append(self.BOS)
            stream.extend(get(t, 0) for t in sent)
            stream.append(1)
        F = np.array(stream, dtype=np.int64)

        pos = np.arange(2, len(F))
        pos = pos[F[pos] != self.BOS]    # every position that is a real target
        w, v, u = F[pos], F[pos - 1], F[pos - 2]
        B = self.B

        self.N = len(w)
        self.uni = np.bincount(w, minlength=self.V).astype(np.int64)
        self.bi_keys, self.bi_counts = np.unique(v * B + w, return_counts=True)
        self.tri_keys, self.tri_counts = np.unique((u * B + v) * B + w, return_counts=True)
        self.bi_cum = np.concatenate(([0], np.cumsum(self.bi_counts)))
        self.tri_cum = np.concatenate(([0], np.cumsum(self.tri_counts)))

        self.suggestable = np.array([is_word(t) for t in self.itos])
        self.suggest_order = np.nonzero(self.suggestable)[0]  # most frequent first

    def context(self, tokens, n):
        """Ids of the last n tokens, padded on the left with BOS (OOV -> <unk>)."""
        ids = [self.stoi.get(t, 0) for t in tokens[-n:]] if n else []
        return [self.BOS] * (n - len(ids)) + ids

    def bi_range(self, v):
        lo = np.searchsorted(self.bi_keys, v * self.B)
        hi = np.searchsorted(self.bi_keys, v * self.B + self.B)
        return lo, hi

    def tri_range(self, u, v):
        base = (u * self.B + v) * self.B
        lo = np.searchsorted(self.tri_keys, base)
        hi = np.searchsorted(self.tri_keys, base + self.B)
        return lo, hi

    @staticmethod
    def lookup(keys, counts, query):
        """Counts for an array of packed n-gram keys (0 where never seen)."""
        idx = np.minimum(np.searchsorted(keys, query), len(keys) - 1)
        return np.where(keys[idx] == query, counts[idx], 0)


class BigramLM:
    """P(w | previous token) with add-k smoothing."""

    def __init__(self, counts, k):
        self.c, self.k = counts, k
        self._cache = {}

    def probs(self, tokens, cand):
        """Probability of each candidate id given the typed tokens."""
        c = self.c
        v = c.context(tokens, 1)[0]
        lo, hi = c.bi_range(v)
        n_ctx = c.bi_cum[hi] - c.bi_cum[lo]
        c2 = c.lookup(c.bi_keys, c.bi_counts, v * c.B + cand)
        return (c2 + self.k) / (n_ctx + self.k * c.V)

    def logprob(self, tokens, target):
        """Natural-log probability of the true next word (OOV scored as <unk>)."""
        cand = np.array([self.c.stoi.get(target, 0)], dtype=np.int64)
        return float(np.log(self.probs(tokens, cand)[0]))

    def top_k(self, tokens, n=3):
        c = self.c
        v = c.context(tokens, 1)[0]
        if v in self._cache:
            return self._cache[v]
        lo, hi = c.bi_range(v)
        ids = c.bi_keys[lo:hi] - v * c.B
        cnt = c.bi_counts[lo:hi]
        ok = c.suggestable[ids]
        ids, cnt = ids[ok], cnt[ok]
        # seen followers outrank unseen words under add-k; ties go to the more frequent word
        chosen = [int(i) for i in ids[np.lexsort((ids, -cnt))][:n]]
        for i in c.suggest_order:        # unseen words all tie, so fill by unigram frequency
            if len(chosen) == n:
                break
            if int(i) not in chosen:
                chosen.append(int(i))
        out = [c.itos[i] for i in chosen]
        self._cache[v] = out
        return out


class TrigramLM:
    """Linear interpolation of unigram, bigram and trigram estimates.

    lambdas = (l_unigram, l_bigram, l_trigram), summing to 1, tuned on validation.
    If a context was never seen, that order is dropped and the weights renormalized.
    """

    def __init__(self, counts, lambdas):
        assert lambdas[0] > 0, "the unigram weight must be > 0"
        self.c, self.lam = counts, lambdas
        self._cache = {}
        self._mark = np.zeros(counts.V, dtype=bool)

    def _setup(self, u, v):
        c = self.c
        lo2, hi2 = c.bi_range(v)
        lo3, hi3 = c.tri_range(u, v)
        n2 = c.bi_cum[hi2] - c.bi_cum[lo2]
        n3 = c.tri_cum[hi3] - c.tri_cum[lo3]
        l1, l2, l3 = self.lam
        w1, w2, w3 = l1, (l2 if n2 > 0 else 0.0), (l3 if n3 > 0 else 0.0)
        s = w1 + w2 + w3
        return n2, n3, w1 / s, w2 / s, w3 / s

    def _probs(self, u, v, cand):
        c = self.c
        n2, n3, w1, w2, w3 = self._setup(u, v)
        p = w1 * (c.uni[cand] + 1) / (c.N + c.V)          # add-one unigram, never zero
        if w2 > 0:
            p = p + w2 * c.lookup(c.bi_keys, c.bi_counts, v * c.B + cand) / n2
        if w3 > 0:
            p = p + w3 * c.lookup(c.tri_keys, c.tri_counts, (u * c.B + v) * c.B + cand) / n3
        return p

    def probs(self, tokens, cand):
        u, v = self.c.context(tokens, 2)
        return self._probs(u, v, cand)

    def logprob(self, tokens, target):
        cand = np.array([self.c.stoi.get(target, 0)], dtype=np.int64)
        return float(np.log(self.probs(tokens, cand)[0]))

    def top_k(self, tokens, n=3):
        c = self.c
        u, v = c.context(tokens, 2)
        if (u, v) in self._cache:
            return self._cache[(u, v)]
        # Candidates: every word seen after v, plus the most frequent words not seen
        # after v (their score is only the unigram term, so the top n of them suffice).
        lo, hi = c.bi_range(v)
        pool = c.bi_keys[lo:hi] - v * c.B
        pool = pool[c.suggestable[pool]]
        self._mark[pool] = True
        extra = []
        for i in c.suggest_order:
            if not self._mark[i]:
                extra.append(int(i))
                if len(extra) == n:
                    break
        self._mark[pool] = False
        cand = np.concatenate([pool, np.array(extra, dtype=np.int64)])
        scores = self._probs(u, v, cand)
        out = [c.itos[i] for i in cand[np.lexsort((cand, -scores))][:n]]
        self._cache[(u, v)] = out
        return out


if __name__ == "__main__":
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
    t0 = time.time()
    train = read_sentences(ROOT / "data" / "processed" / "train_formal.jsonl")
    c = NGramCounts(train, cfg["protocol"]["min_word_count"])
    print(f"trained in {time.time() - t0:.1f}s | vocab {c.V:,} | "
          f"bigram types {len(c.bi_keys):,} | trigram types {len(c.tri_keys):,}")

    bi = BigramLM(c, k=0.1)
    tri = TrigramLM(c, lambdas=(0.1, 0.3, 0.6))     # demo values, tuned in the next step

    print("\nTop-3 suggestions (demo settings, not tuned):")
    tests = ["the united", "he was born in", "in the", "<user> thanks for", "i love", "",
             "zzzxqy", "\U0001F602", "the quick brown fox jumped over"]
    for ctx in tests:
        toks = ctx.split()
        print(f"  {ctx!r:40} bigram={bi.top_k(toks)}  trigram={tri.top_k(toks)}")

    print("\nSanity check: probabilities over the whole vocabulary should sum to 1")
    allw = np.arange(c.V, dtype=np.int64)
    for ctx in (["the", "united"], ["zzzxqy"], []):
        print(f"  {ctx!r:22} bigram={bi.probs(ctx, allw).sum():.6f}  "
              f"trigram={tri.probs(ctx, allw).sum():.6f}")