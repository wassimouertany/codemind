"""Tests for BM25 sparse encoding.

The asymmetry tests are the important ones. `encode_documents` must weight
terms and `encode_query` must not — if both ever route to the same fastembed
call, ranking degrades in a way no test of "did I get a vector back" catches.
"""

from __future__ import annotations

import pytest
from qdrant_client import models

from codemind.retrieval.sparse import SparseEncoder, get_sparse_encoder

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def encoder() -> SparseEncoder:
    """One model load for the whole module — construction is the expensive part."""
    return SparseEncoder()


# --------------------------------------------------------------------------- #
# shape and type contract
# --------------------------------------------------------------------------- #


def test_encode_documents_returns_one_vector_per_input(encoder: SparseEncoder) -> None:
    texts = ["class OrderService {}", "def get_user_by_id(self, user_id): pass", "int x = 1;"]
    vectors = encoder.encode_documents_sync(texts)
    assert len(vectors) == len(texts)


def test_empty_input_returns_empty_list(encoder: SparseEncoder) -> None:
    assert encoder.encode_documents_sync([]) == []


def test_vectors_are_qdrant_native_python_scalars(encoder: SparseEncoder) -> None:
    """numpy ints would pass here but be rejected by Qdrant's strict models."""
    vector = encoder.encode_documents_sync(["placeOrder processPayment"])[0]
    assert isinstance(vector, models.SparseVector)
    assert all(type(index) is int for index in vector.indices)
    assert all(type(value) is float for value in vector.values)
    assert len(vector.indices) == len(vector.values)


# --------------------------------------------------------------------------- #
# the tokenizer is actually applied
# --------------------------------------------------------------------------- #


def test_camel_case_identifier_matches_spaced_query(encoder: SparseEncoder) -> None:
    """The whole point of the module: `getUserById` must reach "get user by id".

    Without the tokenizer these two share zero index positions.
    """
    document = encoder.encode_documents_sync(["getUserById"])[0]
    query = encoder.encode_query_sync("get user by id")
    assert set(document.indices) & set(query.indices)


def test_snake_case_and_camel_case_spellings_collide(encoder: SparseEncoder) -> None:
    """`find_by_customer_id` and `findByCustomerId` are the same symbol to BM25."""
    snake = encoder.encode_documents_sync(["find_by_customer_id"])[0]
    camel = encoder.encode_documents_sync(["findByCustomerId"])[0]
    assert set(snake.indices) == set(camel.indices)


def test_dotted_path_is_split(encoder: SparseEncoder) -> None:
    document = encoder.encode_documents_sync(["com.demo.repository.OrderRepository"])[0]
    query = encoder.encode_query_sync("order repository")
    assert set(query.indices) <= set(document.indices)


# --------------------------------------------------------------------------- #
# document / query asymmetry
# --------------------------------------------------------------------------- #


def test_query_weights_are_uniform(encoder: SparseEncoder) -> None:
    """Query side contributes each term once. Repetition must not re-weight it."""
    vector = encoder.encode_query_sync("order service find customer find find")
    assert vector.values
    assert all(value == pytest.approx(1.0) for value in vector.values)


def test_document_weights_reflect_term_frequency(encoder: SparseEncoder) -> None:
    """Document side applies BM25 saturation, so a repeated term outweighs a rare one."""
    vector = encoder.encode_documents_sync(["order service find customer find find"])[0]
    assert len(set(vector.values)) > 1, "all-equal weights means query_embed leaked in"


def test_document_and_query_cover_the_same_terms(encoder: SparseEncoder) -> None:
    """Different weights, same vocabulary. Divergence here means split tokenization."""
    text = "processPayment gateway timeout"
    document = encoder.encode_documents_sync([text])[0]
    query = encoder.encode_query_sync(text)
    assert set(document.indices) == set(query.indices)


# --------------------------------------------------------------------------- #
# degenerate input
# --------------------------------------------------------------------------- #


def test_stopword_only_query_returns_empty_vector_without_raising(
    encoder: SparseEncoder,
) -> None:
    """`public static void return` is all keywords. Empty is correct, a crash is not."""
    vector = encoder.encode_query_sync("public static void return")
    assert vector.indices == []
    assert vector.values == []


def test_empty_query_returns_empty_vector(encoder: SparseEncoder) -> None:
    assert encoder.encode_query_sync("").indices == []


def test_punctuation_only_document_does_not_break_alignment(encoder: SparseEncoder) -> None:
    """A junk chunk still occupies its slot, or every later vector misaligns."""
    vectors = encoder.encode_documents_sync(["{ } ; ( )", "OrderRepository"])
    assert len(vectors) == 2
    assert vectors[0].indices == []
    assert vectors[1].indices


# --------------------------------------------------------------------------- #
# async surface and lifetime
# --------------------------------------------------------------------------- #


async def test_async_matches_sync(encoder: SparseEncoder) -> None:
    texts = ["OrderController placeOrder", "PaymentTimeoutException"]
    assert [v.indices for v in await encoder.encode_documents(texts)] == [
        v.indices for v in encoder.encode_documents_sync(texts)
    ]
    assert (await encoder.encode_query("place order")).indices == encoder.encode_query_sync(
        "place order"
    ).indices


def test_get_sparse_encoder_is_a_singleton() -> None:
    """Heavy models load once. A second load would double RSS for nothing."""
    assert get_sparse_encoder() is get_sparse_encoder()
