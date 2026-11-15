"""Code analyst: root cause for bug questions, design for architecture questions.

Both routes need the same thing — reasoning over the bodies of the retrieved
chunks — and differ only in what the answer explains, so one prompt,
`code_analyst.jinja2`, serves both and branches on the route.

The graph still registers one node per route. A trace span is named after its
node, and `code_analyst` on an architecture question would make the Day 13
latency breakdown unreadable; `bug_analyst` and `architecture_analyst` say
which question each span answered.
"""

from __future__ import annotations

from codemind.agents.nodes.specialist import make_specialist
from codemind.agents.state import AnalysisUpdate, Node, Route
from codemind.llm.base import LLMClient

PROMPT = "code_analyst"

NODE_NAMES: dict[Route, str] = {
    Route.BUG: "bug_analyst",
    Route.ARCHITECTURE: "architecture_analyst",
}


def make_code_analyst(llm: LLMClient, route: Route) -> Node[AnalysisUpdate]:
    return make_specialist(NODE_NAMES[route], llm, prompt=PROMPT)
