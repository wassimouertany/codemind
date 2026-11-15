"""Router node: classify the question into a `Route`.

The decision comes back through `generate_structured`, so the route is an enum
value enforced by the backend's grammar and re-validated by Pydantic. Nothing
here parses free text.

The router never fails the graph. Any `LLMError` — an unreachable backend, a
truncated or off-schema answer — routes to `DEFAULT_ROUTE` and is recorded in
`errors`, so the fallback is visible in the trace instead of looking like a
confident classification.
"""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel

from codemind.agents.state import DEFAULT_ROUTE, CodeMindState, Route, RouterUpdate
from codemind.core.exceptions import LLMError
from codemind.llm.base import LLMClient
from codemind.llm.prompts import render_prompt

logger = logging.getLogger(__name__)

NODE_NAME = "router"


class RouteDecision(BaseModel):
    """The router's structured output. One enum field, nothing to parse."""

    route: Route


class RouterNode(Protocol):
    """A node LangGraph can call as `node(state=...)`; a bare `Callable` cannot promise that."""

    async def __call__(self, state: CodeMindState) -> RouterUpdate: ...


def make_router(llm: LLMClient) -> RouterNode:
    """Bind the router to an `LLMClient`; the graph registers the result as a node."""

    async def router(state: CodeMindState) -> RouterUpdate:
        prompt = render_prompt("router", question=state["question"])
        try:
            decision = await llm.generate_structured(prompt, RouteDecision)
        except LLMError as exc:
            logger.warning("router fell back to %s: %s", DEFAULT_ROUTE, exc)
            return {
                "route": [DEFAULT_ROUTE],
                "errors": [f"{NODE_NAME}: unclassifiable, defaulted to {DEFAULT_ROUTE}: {exc}"],
                "messages": [{"node": NODE_NAME, "content": f"route={DEFAULT_ROUTE} (fallback)"}],
            }
        return {
            "route": [decision.route],
            "messages": [{"node": NODE_NAME, "content": f"route={decision.route}"}],
        }

    return router
