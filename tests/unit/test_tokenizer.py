"""Tests for code-aware tokenization.

`test_index_and_query_tokenization_are_symmetric` is the load-bearing one. If
the two sides ever diverge, BM25 recall drops toward zero and nothing raises an
error — the system just quietly stops finding things.
"""

from __future__ import annotations

import pytest

from codemind.retrieval.tokenizer import (
    STOPWORDS,
    split_identifier,
    tokenize,
    tokenize_query,
    tokenized_text,
)

# --------------------------------------------------------------------------- #
# identifier splitting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        # camelCase
        ("getUserById", ["get", "user", "by", "id"]),
        ("placeOrder", ["place", "order"]),
        ("processPayment", ["process", "payment"]),
        ("callGateway", ["call", "gateway"]),
        # PascalCase
        ("OrderRepository", ["order", "repository"]),
        ("PaymentTimeoutException", ["payment", "timeout", "exception"]),
        ("UserService", ["user", "service"]),
        # snake_case
        ("find_by_customer_id", ["find", "by", "customer", "id"]),
        ("get_user_by_id", ["get", "user", "by", "id"]),
        ("deactivate_user", ["deactivate", "user"]),
        # acronym runs — the case naive regexes get wrong
        ("HTTPResponseCode", ["http", "response", "code"]),
        ("XMLHttpRequest", ["xml", "http", "request"]),
        ("IOException", ["io", "exception"]),
        ("parseJSONBody", ["parse", "json", "body"]),
        # punctuation and qualified names
        ("com.demo.service.OrderService", ["com", "demo", "service", "order", "service"]),
        ("@PostMapping", ["post", "mapping"]),
        ("JpaRepository<Order, Long>", ["jpa", "repository", "order", "long"]),
        ("kebab-case-name", ["kebab", "case", "name"]),
        ("__init__", ["init"]),
    ],
)
def test_split_identifier(identifier: str, expected: list[str]) -> None:
    assert split_identifier(identifier) == expected


def test_split_is_pure_lowercase() -> None:
    for part in split_identifier("SomeMixedCASEIdentifier"):
        assert part == part.lower()


def test_empty_and_junk_inputs() -> None:
    assert split_identifier("") == []
    assert split_identifier("   ") == []
    assert split_identifier("!!!") == []
    assert tokenize("") == []


# --------------------------------------------------------------------------- #
# the invariant
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "getUserById",
        "OrderService.placeOrder",
        "why does POST /orders return 500",
        "PaymentTimeoutException in processPayment",
        "find_by_customer_id",
    ],
)
def test_index_and_query_tokenization_are_symmetric(text: str) -> None:
    """The same input must produce the same terms on both sides.

    Index time calls `tokenized_text`, query time calls `tokenize_query`. Any
    divergence between them breaks sparse retrieval silently, so it is asserted
    rather than assumed.
    """
    assert tokenized_text(text) == tokenize_query(text)


def test_query_side_is_cached_but_not_stale() -> None:
    first = tokenize_query("getUserById")
    second = tokenize_query("getUserById")
    assert first == second == "get user by id"


# --------------------------------------------------------------------------- #
# stopwords
# --------------------------------------------------------------------------- #


def test_language_keywords_are_dropped() -> None:
    tokens = tokenize("public static void main(String[] args)")
    for keyword in ("public", "static", "void", "string"):
        assert keyword not in tokens
    assert "main" in tokens
    assert "args" in tokens


def test_domain_words_survive() -> None:
    """Stopwords must not eat the vocabulary that actually identifies code."""
    tokens = tokenize(
        "public Order placeOrder(Order order) { return orderRepository.save(order); }"
    )
    for word in ("order", "place", "repository", "save"):
        assert word in tokens


def test_stopwords_can_be_kept() -> None:
    assert "return" in tokenize("return value", drop_stopwords=False)
    assert "return" not in tokenize("return value", drop_stopwords=True)


def test_stopword_list_is_lowercase() -> None:
    assert all(word == word.lower() for word in STOPWORDS)


# --------------------------------------------------------------------------- #
# single characters
# --------------------------------------------------------------------------- #


def test_single_characters_are_dropped() -> None:
    """Loop variables are noise in an inverted index."""
    tokens = tokenize("for (int i = 0; i < n; i++) { x = y; }")
    for noise in ("i", "n", "x", "y"):
        assert noise not in tokens


def test_two_character_tokens_survive() -> None:
    """`io` in IOException and `id` in getUserById must not be dropped."""
    assert "io" in tokenize("IOException")
    assert "id" in tokenize("getUserById")


# --------------------------------------------------------------------------- #
# realistic retrieval behaviour
# --------------------------------------------------------------------------- #


def test_natural_question_matches_the_method_name() -> None:
    """This is the entire point of the module."""
    query = set(tokenize_query("how do we get a user by id").split())
    method = set(tokenized_text("public User getUserById(Long id)").split())
    assert {"get", "user", "by", "id"} <= query & method


def test_exception_name_is_findable_from_prose() -> None:
    query = set(tokenize_query("payment timeout when placing an order").split())
    code = set(tokenized_text("throw new PaymentTimeoutException(...)").split())
    assert {"payment", "timeout"} <= query & code


def test_annotation_route_is_findable() -> None:
    query = set(tokenize_query("the POST mapping for orders").split())
    code = set(
        tokenized_text("@PostMapping public Order createOrder(@RequestBody Order o)").split()
    )
    assert {"post", "mapping"} <= query & code


def test_this_module_splits_but_does_not_stem() -> None:
    """Plural/singular is NOT handled here, by design.

    "orders" does not become "order". Stemming belongs to the BM25 encoder
    downstream (fastembed's Bm25 runs a Snowball stemmer), and doing it in both
    places risks double-stemming, which mangles identifiers: "processing" ->
    "process" is fine, but "caching" -> "cach" then "cach" -> "cach" is not.

    Documented as a test so the boundary is explicit rather than folklore. If
    `sparse.py` ever stops relying on fastembed's stemmer, this is the test that
    tells you what you now have to add here.
    """
    assert tokenize_query("orders") == "orders"
    assert tokenized_text("Order") == "order"
    assert tokenize_query("orders") != tokenized_text("Order")


def test_repeated_terms_are_preserved_for_bm25() -> None:
    """BM25 scores on term frequency, so duplicates must not be deduplicated."""
    tokens = tokenize("order order order")
    assert tokens.count("order") == 3


def test_tokenized_text_roundtrip() -> None:
    text = "OrderService.placeOrder"
    assert tokenized_text(text).split() == tokenize(text)
