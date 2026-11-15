"""Dependency agent: who uses, calls, imports or inherits from what, and the impact.

Its prompt shows each chunk's imports alongside the body — coupling lives in
import lists as much as in call sites.
"""

from __future__ import annotations

from codemind.agents.nodes.specialist import make_specialist
from codemind.agents.state import AnalysisUpdate, Node
from codemind.llm.base import LLMClient

NODE_NAME = "dependency_agent"


def make_dependency_agent(llm: LLMClient) -> Node[AnalysisUpdate]:
    return make_specialist(NODE_NAME, llm)
