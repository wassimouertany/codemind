#!/usr/bin/env python
"""Verify the installed stack empirically (v2).

Fixes two bugs in v1 that produced false failures:
  * helper `_walk` was defined after the check that used it
  * `from __future__ import annotations` broke TypedDict resolution inside a
    function scope, so the LangGraph check failed before LangGraph ran

Changes one real thing: dense embeddings now load through fastembed (ONNX)
instead of sentence-transformers, because jina-v2-base-code's trust_remote_code
modeling file is incompatible with transformers 5.x.

Run:  uv run python scripts/verify_stack.py
Needs Qdrant running (`make up`) and Ollama on the Windows side.
"""

# NOTE: deliberately NO `from __future__ import annotations` — it breaks
# TypedDict annotation resolution in the LangGraph check below.

import importlib.metadata as md
import sys
import traceback
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


class CheckWarning(Exception):
    """Raised when a check runs but the result looks wrong."""


def check(name: str) -> Callable[[Callable[[], str]], None]:
    def decorator(fn: Callable[[], str]) -> None:
        try:
            results.append((name, PASS, fn()))
        except CheckWarning as exc:
            results.append((name, WARN, str(exc)))
        except Exception as exc:  # noqa: BLE001
            line = traceback.extract_tb(sys.exc_info()[2])[-1].lineno
            results.append((name, FAIL, f"{type(exc).__name__}: {exc} (line {line})"))

    return decorator


# --------------------------------------------------------------------------- #
# helpers — MUST be defined before any check that uses them
# --------------------------------------------------------------------------- #
def walk_nodes(node: Any):
    yield node
    for child in node.children:
        yield from walk_nodes(child)


# --------------------------------------------------------------------------- #
# LangGraph state — MUST be module level so annotations resolve
# --------------------------------------------------------------------------- #
def _concat(a: list[str], b: list[str]) -> list[str]:
    return a + b


class GraphState(TypedDict):
    steps: Annotated[list[str], _concat]


# --------------------------------------------------------------------------- #
# 1. tree-sitter
# --------------------------------------------------------------------------- #
@check("tree-sitter parse")
def _treesitter() -> str:
    from tree_sitter_language_pack import get_parser

    ok: list[str] = []
    for grammar, src, expected in (
        ("python", b"def hello(x):\n    return x + 1\n", "function_definition"),
        ("java", b"class A { void run() { int x = 1; } }", "method_declaration"),
        ("typescript", b"function hi(a: number): number { return a; }", "function_declaration"),
    ):
        tree = get_parser(grammar).parse(src)
        types = {n.type for n in walk_nodes(tree.root_node)}
        if expected not in types:
            raise RuntimeError(f"{grammar}: no {expected!r}; saw {sorted(types)[:10]}")
        ok.append(grammar)
    return f"grammars ok: {', '.join(ok)}"


# --------------------------------------------------------------------------- #
# 2. which dense models can fastembed actually serve?
# --------------------------------------------------------------------------- #
@check("fastembed catalogue")
def _catalogue() -> str:
    from fastembed import TextEmbedding

    names = [m["model"] for m in TextEmbedding.list_supported_models()]
    interesting = [
        n for n in names
        if any(k in n.lower() for k in ("code", "jina", "bge", "gte", "nomic", "e5"))
    ]
    return f"{len(names)} models; candidates={interesting[:12]}"


# --------------------------------------------------------------------------- #
# 3. dense embeddings via fastembed ONNX (no transformers involved)
# --------------------------------------------------------------------------- #
@check("dense embeddings (fastembed)")
def _dense() -> str:
    import numpy as np
    from fastembed import TextEmbedding

    available = {m["model"] for m in TextEmbedding.list_supported_models()}
    preferred = [
        "jinaai/jina-embeddings-v2-base-code",
        "jinaai/jina-embeddings-v2-base-en",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-small-en-v1.5",
    ]
    chosen = next((m for m in preferred if m in available), None)
    if chosen is None:
        raise RuntimeError(f"none of {preferred} available; have {sorted(available)[:10]}")

    model = TextEmbedding(model_name=chosen)
    vectors = list(
        model.embed(
            [
                "public Order placeOrder(Order order) { paymentService.processPayment(); }",
                "def get_user_by_id(self, user_id: int) -> dict: ...",
                "the weather in Tunis is warm today",
            ]
        )
    )
    dim = len(vectors[0])

    def cos(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    code_code = cos(vectors[0], vectors[1])
    code_prose = cos(vectors[0], vectors[2])
    verdict = f"model={chosen} dim={dim} code~code={code_code:.3f} code~prose={code_prose:.3f}"
    if code_code <= code_prose:
        raise CheckWarning(f"{verdict}  <-- code pairs should score higher than prose")
    return verdict


# --------------------------------------------------------------------------- #
# 4. reranker, with real precision
# --------------------------------------------------------------------------- #
@check("cross-encoder rerank")
def _rerank() -> str:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("BAAI/bge-reranker-base", device="cpu", max_length=512)
    query = "why does placing an order sometimes return HTTP 500"
    pairs = [
        (query, "public Order placeOrder(Order o) { paymentService.processPayment(o.getTotal()); "
                "return orderRepository.save(o); }  // PaymentTimeoutException is not caught"),
        (query, "public class Order { private Long id; public Long getId() { return id; } }"),
        (query, "README.md: to build the project run mvn clean install"),
    ]
    scores = [float(s) for s in model.predict(pairs)]
    spread = max(scores) - min(scores)
    verdict = f"scores={[round(s, 6) for s in scores]} spread={spread:.6f}"
    if scores[0] != max(scores):
        raise CheckWarning(f"{verdict}  <-- the relevant chunk did not rank first")
    if spread < 1e-4:
        raise CheckWarning(f"{verdict}  <-- scores are flat, reranker adds no signal")
    return verdict


# --------------------------------------------------------------------------- #
# 5. LangGraph (state defined at module level, so this actually tests LangGraph)
# --------------------------------------------------------------------------- #
@check("langgraph StateGraph")
def _langgraph() -> str:
    from langgraph.graph import END, START, StateGraph

    def one(state: GraphState) -> dict:
        return {"steps": ["one"]}

    def two(state: GraphState) -> dict:
        return {"steps": ["two"]}

    g = StateGraph(GraphState)
    g.add_node("one", one)
    g.add_node("two", two)
    g.add_edge(START, "one")
    g.add_edge("one", "two")
    g.add_edge("two", END)
    out = g.compile().invoke({"steps": []})
    if out["steps"] != ["one", "two"]:
        raise RuntimeError(f"unexpected state: {out}")
    return f"v{md.version('langgraph')} compiled + ran, state={out['steps']}"


@check("langgraph conditional edges")
def _conditional() -> str:
    from langgraph.graph import END, START, StateGraph

    def router(state: GraphState) -> dict:
        return {"steps": ["routed"]}

    def pick(state: GraphState) -> str:
        return "bug" if "routed" in state["steps"] else "other"

    def bug(state: GraphState) -> dict:
        return {"steps": ["bug_agent"]}

    def other(state: GraphState) -> dict:
        return {"steps": ["other_agent"]}

    g = StateGraph(GraphState)
    g.add_node("router", router)
    g.add_node("bug", bug)
    g.add_node("other", other)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", pick, {"bug": "bug", "other": "other"})
    g.add_edge("bug", END)
    g.add_edge("other", END)
    out = g.compile().invoke({"steps": []})
    if out["steps"] != ["routed", "bug_agent"]:
        raise RuntimeError(f"routing wrong: {out}")
    return f"conditional routing ok: {out['steps']}"


# --------------------------------------------------------------------------- #
# 6. Langfuse 4.x — discover the real decorator/client surface
# --------------------------------------------------------------------------- #
@check("langfuse surface")
def _langfuse() -> str:
    import langfuse

    notes = []
    notes.append(f"observe={'observe' in dir(langfuse)}")
    notes.append(f"Langfuse={'Langfuse' in dir(langfuse)}")
    notes.append(f"get_client={'get_client' in dir(langfuse)}")
    try:
        from langfuse import observe  # noqa: F401

        notes.append("import_observe=ok")
    except ImportError as exc:
        notes.append(f"import_observe=FAILED({exc})")
    try:
        import langfuse.openai  # noqa: F401

        notes.append("openai_wrapper=ok")
    except Exception:  # noqa: BLE001
        notes.append("openai_wrapper=absent")
    return f"v{md.version('langfuse')} " + " ".join(notes)


# --------------------------------------------------------------------------- #
# 7. RAGAS — informational only, this is week 4
# --------------------------------------------------------------------------- #
@check("ragas (week 4, non-blocking)")
def _ragas() -> str:
    try:
        from ragas import metrics
    except ModuleNotFoundError as exc:
        raise CheckWarning(
            f"v{md.version('ragas')} broken import: {exc} — deferred to week 4, "
            "will be isolated into its own dependency group"
        ) from exc
    names = [n for n in dir(metrics) if not n.startswith("_")]
    return f"v{md.version('ragas')} metrics={names[:10]}"


# --------------------------------------------------------------------------- #
def main() -> int:
    width = max(len(n) for n, _, _ in results)
    print("\n" + "=" * 110)
    print("  CodeMind stack verification (v2)")
    print("=" * 110)
    for name, status, detail in results:
        colour = {PASS: "\033[92m", FAIL: "\033[91m", WARN: "\033[93m"}[status]
        print(f"  {colour}{status}\033[0m  {name:<{width}}  {detail}")
    print("=" * 110)

    failed = [n for n, s, _ in results if s == FAIL]
    warned = [n for n, s, _ in results if s == WARN]
    if failed:
        print(f"\n  {len(failed)} FAILED: {', '.join(failed)}")
    if warned:
        print(f"  {len(warned)} warnings: {', '.join(warned)}")
    if not failed and not warned:
        print("\n  All green.\n")
    else:
        print("  Paste the whole block back into the chat.\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
