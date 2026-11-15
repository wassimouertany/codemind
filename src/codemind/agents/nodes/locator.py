"""Locator: list where the code is, with `file:line`. No LLM.

"Where is X?" is a lookup, not a question that needs analysis. Locator scored
Recall@10 1.000 in every Gate 1 configuration, with the gold file at rank 1 for
7 of 8 questions, so the retriever has already done the work. The path is a
short circuit — retrieve -> locator -> END — and costs no generation at all.
"""

from __future__ import annotations

from codemind.agents.state import AnalysisUpdate, CodeMindState, Evidence, Node
from codemind.core.config import Settings, get_settings
from codemind.core.types import CodeChunk

NODE_NAME = "locator"


def make_locator(settings: Settings | None = None) -> Node[AnalysisUpdate]:
    top_k = (settings or get_settings()).locator_top_k

    async def locator(state: CodeMindState) -> AnalysisUpdate:
        hits = (state.get("retrieved") or [])[:top_k]
        if not hits:
            return {
                "errors": [f"{NODE_NAME}: no retrieved context"],
                "messages": [{"node": NODE_NAME, "content": "No matching code was found."}],
            }
        lines = [f"{rank}. {format_hit(hit.chunk)}" for rank, hit in enumerate(hits, start=1)]
        evidence: list[Evidence] = [
            {
                "file_path": hit.chunk.relative_path,
                "start_line": hit.chunk.start_line,
                "end_line": hit.chunk.end_line,
                "claim": f"{hit.chunk.kind} {qualified_name(hit.chunk)}",
            }
            for hit in hits
        ]
        return {
            "evidence": evidence,
            "messages": [{"node": NODE_NAME, "content": "\n".join(lines)}],
        }

    return locator


def qualified_name(chunk: CodeChunk) -> str:
    if chunk.parent_symbol and chunk.parent_symbol != chunk.symbol_name:
        return f"{chunk.parent_symbol}.{chunk.symbol_name}"
    return chunk.symbol_name


def format_hit(chunk: CodeChunk) -> str:
    return (
        f"`{chunk.relative_path}:{chunk.start_line}-{chunk.end_line}` — "
        f"{qualified_name(chunk)} ({chunk.kind})"
    )
