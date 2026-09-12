---
description: Scaffold a new LangGraph node — usage: /node grounding
argument-hint: <node_name>
---
Create the LangGraph node `$1` following the conventions in CLAUDE.md:
- File: `src/codemind/agents/nodes/$1.py`
- Signature: `async def $1_node(state: CodeMindState) -> dict[str, Any]`
- Prompt template in `configs/prompts/$1.jinja2` — no inline f-string prompts
- Wrap the LLM call in a Langfuse span named `node.$1`
- Return only the state keys this node owns; never overwrite the whole state
- Add a unit test in `tests/unit/test_$1_node.py` with a stubbed LLMClient
Then register it in `src/codemind/agents/graph.py` and show me the edge changes.
