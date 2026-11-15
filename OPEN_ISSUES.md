# Open issues

Tracked defects with a known fix, deliberately not fixed yet. Each entry says
what blocks it, because "deferred" without a reason becomes "forgotten".

---

## 1. Files with no chunkable node are unretrievable

**Opened:** 2026-10-05 · **Severity:** medium · **Area:** `ingestion/ast_chunker.py`

A file that parses successfully but contains no chunkable node yields zero
chunks, so nothing in it can ever be retrieved.

**Measured** on `spring-petclinic` (main sources only): 30 files load, 25
produce chunks.

| Case | Chunks | Verdict |
|---|---|---|
| Java `package-info.java` (5 files) | 0 | **Correct** — declares no symbol; only a package javadoc |
| Python module of top-level statements | 0 | **Wrong** — `settings.py`, constants, Django config are real code |
| TypeScript file of consts/types | 1 | Escapes it only because `type_alias_declaration` is a container node |

The Python case is the defect: on a Python repository this silently drops
configuration and constants rather than javadoc.

**Fix:** emit a file-level fallback chunk when a parsed file produces none —
path + package + leading comment as the context header, file body as the chunk.
That also makes package-level javadoc retrievable for architecture questions.

**Why deferred:** it changes chunking, so it invalidates the corpus the Gate 1
numbers were measured against (25 files / 99 chunks). Scheduled immediately
after the Gate 1 commits; the ablation is then re-run and **both** results
reported, since a corpus change is exactly the kind of thing that must not be
folded silently into a metric.

**Verify the fix with:** a Python fixture of only top-level statements, a Java
`package-info.java`, and a re-run of `make eval-retrieval`.

---

## 2. The 3B model cites by walking chunks in order, not by relevance

**Opened:** 2026-11-15 · **Severity:** high · **Area:** fine-tuning dataset (Week 3)

Asked for 1-6 claims over 10 retrieved chunks, the base
`qwen2.5-coder:3b-instruct` fills the cap by describing chunks `[1], [2],
[3] ...` in retrieval order, whether or not the question needs them.

**Measured** on the fixture's reference question ("Why does POST /orders
sometimes return 500?"): claims [1]-[3] are the correct call path and root
cause, each tied to the right chunk. Claims [4]-[6] describe `callGateway`,
`findById` and a constructor — accurate about their chunks, irrelevant to the
question. On "Which classes call PaymentService?" claim [1] is the whole
answer; [2]-[6] are padding, and two of them are false statements about the
chunks they cite. The prompt rule "stop when it is answered; do not describe
every chunk" did not change this.

**Fix:** the Week 3 QLoRA dataset must teach *selective* citation. Include
examples where the correct answer uses 2 of 10 chunks, where the relevant
chunks are not the first ones in retrieval order, and where the irrelevant
chunks are deliberately plausible neighbours (same class, same layer) — so the
model learns that position and proximity are not relevance.

**Why deferred:** it is a model behaviour, not a code defect; prompting
alone did not move it, and the training data is the lever. Until then the
grounding gate (Day 12) must check each claim against its cited chunk's text
and drop unsupported claims, rather than trusting that a correctly cited claim
is a true one.

**Verify the fix with:** the same two fixture questions in
`tests/integration/test_graph_live.py`, comparing claims per answer and the
share of claims the grounding gate keeps, base model vs QLoRA, identical
contexts.
