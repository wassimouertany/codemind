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
