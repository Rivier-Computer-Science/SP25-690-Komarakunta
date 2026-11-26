"""Shared text normalization and tokenization for all three models.

Every model (bigram, trigram, distilgpt2) must see text processed by
exactly this code, so the comparison is fair.
"""
import html
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")

# A word = unicode letters, optionally with inner apostrophes (don't, it's)
WORD = r"[^\W\d_]+(?:'[^\W\d_]+)*"
TOKEN_RE = re.compile(rf"<url>|<user>|{WORD}|\d+(?:[.,]\d+)*[^\W\d_]*|\S")
WORD_RE = re.compile(WORD)


def normalize(text):
    """Lowercase, unify quotes, mask URLs and @mentions, strip '#' from hashtags."""
    if not text:
        return ""
    text = html.unescape(text).lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    # WikiText raw marks hyphens/commas/periods inside words as ' @-@ ' etc.
    text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    text = URL_RE.sub(" <url> ", text)
    text = MENTION_RE.sub(" <user> ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    """Normalize, then split into words, numbers, placeholders, and single symbols."""
    return TOKEN_RE.findall(normalize(text))


def is_word(token):
    """True only for alphabetic words. Only these are scored as targets or suggested."""
    return WORD_RE.fullmatch(token) is not None


if __name__ == "__main__":
    tests = [
        "QT @user In the original draft of the 7th book, Remus Lupin survived the Battle of Hogwarts. #HappyBirthdayRemusLupin",
        "",
        "\U0001F602\U0001F602\U0001F602",
        "Check this out https://t.co/abc123 lol",
        "He said it's well @-@ known .",
        "Don\u2019t worry!!!",
    ]
    for t in tests:
        toks = tokenize(t)
        print(repr(t))
        print("  tokens  :", toks)
        print("  scorable:", [w for w in toks if is_word(w)])