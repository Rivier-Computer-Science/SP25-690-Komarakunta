"""Shared evaluation rules, used by tuning now and by the final test run later.

Scored positions (one rule for every model): in each sentence, every alphabetic
word that has at least one token typed before it. Punctuation, emoji, numbers and
<user>/<url> placeholders stay in the context but are never scored.
"""
import numpy as np

from normalize import is_word


def scored_positions(tokens):
    """Indices of the scored words in one tokenized sentence."""
    return [i for i in range(1, len(tokens)) if is_word(tokens[i])]


def ngram_features(c, sentences):
    """Gather, for every scored position, the counts both n-gram models need.

    Returns a dict of equal-length arrays:
      sid, pid   sentence index and token index (to trace each position back)
      w, u, v    ids of the true word and of the two previous tokens (OOV -> <unk>)
      in_vocab   True if the true word is in the vocabulary
      c2, n2     bigram count of (v, w) and total count of context v
      c3, n3     trigram count of (u, v, w) and total count of context (u, v)
      p1         add-one unigram probability of w
    """
    sid, pid, u, v, w, inv = [], [], [], [], [], []
    for s_i, toks in enumerate(sentences):
        ids = [c.stoi.get(t, 0) for t in toks]
        for i in scored_positions(toks):
            sid.append(s_i)
            pid.append(i)
            v.append(ids[i - 1])
            u.append(ids[i - 2] if i >= 2 else c.BOS)
            w.append(ids[i])
            inv.append(toks[i] in c.stoi)
    sid, pid, u, v, w = (np.array(a, dtype=np.int64) for a in (sid, pid, u, v, w))
    B = c.B
    n2 = c.bi_cum[np.searchsorted(c.bi_keys, v * B + B)] - c.bi_cum[np.searchsorted(c.bi_keys, v * B)]
    ctx = (u * B + v) * B
    n3 = c.tri_cum[np.searchsorted(c.tri_keys, ctx + B)] - c.tri_cum[np.searchsorted(c.tri_keys, ctx)]
    return {
        "sid": sid, "pid": pid, "u": u, "v": v, "w": w, "in_vocab": np.array(inv),
        "c2": c.lookup(c.bi_keys, c.bi_counts, v * B + w), "n2": n2,
        "c3": c.lookup(c.tri_keys, c.tri_counts, ctx + w), "n3": n3,
        "p1": (c.uni[w] + 1) / (c.N + c.V),
    }


def bigram_logprobs(f, V, k):
    """Natural-log probability of the true word under add-k bigram smoothing."""
    return np.log((f["c2"] + k) / (f["n2"] + k * V))


def trigram_logprobs(f, lam):
    """Natural-log probability under interpolation; unseen contexts drop their order."""
    l1, l2, l3 = lam
    w2 = np.where(f["n2"] > 0, l2, 0.0)
    w3 = np.where(f["n3"] > 0, l3, 0.0)
    p2 = f["c2"] / np.maximum(f["n2"], 1)
    p3 = f["c3"] / np.maximum(f["n3"], 1)
    return np.log((l1 * f["p1"] + w2 * p2 + w3 * p3) / (l1 + w2 + w3))


def perplexity(logprobs, mask):
    """Per-word perplexity over the positions selected by mask."""
    return float(np.exp(-logprobs[mask].mean()))