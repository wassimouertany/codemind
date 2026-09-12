---
description: Audit the codebase for GPU memory violations
---
This project runs on a 4GB VRAM card. Audit for violations:
1. Any hardcoded `.cuda()`, `.to("cuda")`, or `device="cuda"` outside of
   `training/` — these must read from Settings instead.
2. Any model loaded at import time rather than in the FastAPI lifespan.
3. Any code path that could hold the embedder, the reranker and the LLM
   on GPU at the same time.
4. Any `max_length` / `num_ctx` that exceeds what the config declares.
Report findings as a table. Fix only what I confirm.
