"""Code analyst: root cause for bug questions, design for architecture questions.

Both routes need the same thing — reasoning over the bodies of the retrieved
chunks — and differ only in what the answer explains, so one node serves both
and `code_analyst.jinja2` branches on the route.
"""

from __future__ import annotations

from codemind.agents.nodes.specialist import make_specialist
from codemind.agents.state import AnalysisUpdate, Node
from codemind.llm.base import LLMClient

NODE_NAME = "code_analyst"


def make_code_analyst(llm: LLMClient) -> Node[AnalysisUpdate]:
    return make_specialist(NODE_NAME, llm)
