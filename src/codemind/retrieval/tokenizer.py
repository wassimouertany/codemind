"""Code-aware tokenization for sparse (BM25) retrieval.

BM25 was built for prose. Its tokenizer splits on whitespace and punctuation,
then stems English words. Handed `getUserById` it produces one opaque token that
matches nothing a human would type, so sparse search over code is close to dead
on arrival.

Splitting identifiers first fixes that:

    getUserById            -> get user by id
    OrderRepository        -> order repository
    find_by_customer_id    -> find by customer id
    PaymentTimeoutException-> payment timeout exception
    HTTPResponseCode       -> http response code

THE INVARIANT: this function runs at index time AND at query time, identically.
If the two ever diverge, sparse recall silently drops to near zero with no error
anywhere. That is why tokenization lives in one module with one public entry
point rather than being inlined at each call site.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Splits CamelCase and PascalCase, and handles acronym runs correctly:
# "HTTPResponse" -> "HTTP", "Response"  (not "H", "T", "T", "P", "Response")
_CAMEL = re.compile(
    r"""
    (?<=[a-z0-9])(?=[A-Z])      # getUser      -> get|User
  | (?<=[A-Z])(?=[A-Z][a-z])    # HTTPResponse -> HTTP|Response
  | (?<=[A-Za-z])(?=[0-9])      # user2Id      -> user|2Id
  | (?<=[0-9])(?=[A-Za-z])      # base64Encode -> base64... see note below
    """,
    re.VERBOSE,
)

# Everything that separates identifiers in real source: snake_case, kebab-case,
# dotted paths, generics, annotations, operators, punctuation.
_SEPARATORS = re.compile(r"[^A-Za-z0-9]+")

# Language keywords carry no retrieval signal — every Java file has `public`
# Language keywords carry no retrieval signal — every Java file has `public`
# and `return`. Dropping them keeps BM25 focused on domain vocabulary.
#
# SIM905 (prefer a list literal) is suppressed deliberately. Its rationale is
# avoiding a runtime split, which matters in a hot path; this runs once at
# import. A literal this long is exploded to one item per line by the
# formatter, turning 12 scannable lines into 70 — a real readability cost for
# no measurable gain. The rule and the formatter disagree here; readability wins.
_JAVA_TS_KEYWORDS = """
public private protected static final void class interface extends implements
new return this super import package throws throw try catch finally
if else for while switch case break continue
const let var function export default async await
null true false int long double float boolean string object type enum record
""".split()  # noqa: SIM905

_PYTHON_KEYWORDS = """
def self cls none elif pass raise from as lambda yield with assert
global nonlocal del and or not in is
""".split()  # noqa: SIM905

STOPWORDS: frozenset[str] = frozenset(_JAVA_TS_KEYWORDS + _PYTHON_KEYWORDS)

MIN_TOKEN_LENGTH = 2
"""Single characters (loop variables i, x, n) are noise in an inverted index."""


def split_identifier(identifier: str) -> list[str]:
    """Split one identifier into lowercase word parts.

    >>> split_identifier("getUserById")
    ['get', 'user', 'by', 'id']
    >>> split_identifier("PaymentTimeoutException")
    ['payment', 'timeout', 'exception']
    >>> split_identifier("find_by_customer_id")
    ['find', 'by', 'customer', 'id']
    """
    parts: list[str] = []
    for raw in _SEPARATORS.split(identifier):
        if not raw:
            continue
        parts.extend(piece.lower() for piece in _CAMEL.split(raw) if piece)
    return parts


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Turn source text or a natural-language query into BM25 terms.

    Both sides of retrieval call this. Index time sees code; query time sees a
    question. Feeding both through the same function is what lets "how do we get
    a user by id" match `getUserById`.
    """
    tokens: list[str] = []
    for raw in _SEPARATORS.split(text):
        if not raw:
            continue
        for piece in _CAMEL.split(raw):
            token = piece.lower()
            if len(token) < MIN_TOKEN_LENGTH:
                continue
            if drop_stopwords and token in STOPWORDS:
                continue
            tokens.append(token)
    return tokens


def tokenized_text(text: str, *, drop_stopwords: bool = True) -> str:
    """Tokenize and rejoin, for feeding a BM25 encoder that expects a string.

    fastembed's `SparseTextEmbedding` takes text, not tokens, so we hand it
    pre-split text and let its own tokenizer do the trivial whitespace split.
    """
    return " ".join(tokenize(text, drop_stopwords=drop_stopwords))


@lru_cache(maxsize=8192)
def tokenize_query(query: str) -> str:
    """Query-side entry point. Cached because queries repeat during evaluation.

    Deliberately identical to the index side, stopwords included. An earlier
    draft kept stopwords in queries to preserve words like "not" — but BM25 is a
    bag of words and cannot represent negation anyway, so those terms could only
    match index entries that no longer exist. Asymmetry here is the classic way
    sparse recall collapses silently; dense retrieval carries the semantics.
    """
    return " ".join(tokenize(query, drop_stopwords=True))
