# Building the retrieval gold set — Gate 1

30 questions, hand-labelled by you. This file explains the format and the rules.
Output goes to `evaluation/datasets/retrieval_goldset.jsonl`.

---

## Why you label this by hand

If an LLM writes questions from the code, you are measuring whether retrieval
can find code that was *described from that code*. The vocabulary overlap is
guaranteed, every configuration scores well, and the ablation flattens — which
is the one thing it must not do, since its whole purpose is to show which
component earns its place.

Write the questions the way a developer who has never seen the repo would ask
them. Then find the answer yourself and record the files.

**This is the most valuable two hours in the whole project.** Every number in
your README traces back to these 30 rows.

---

## Format

One JSON object per line:

```json
{"id": "bug-01", "category": "bug", "question": "Why might saving a new owner fail silently?", "gold_files": ["src/main/java/.../OwnerController.java", "src/main/java/.../OwnerRepository.java"], "notes": "validation happens in the controller, not the repository"}
```

| Field | Rule |
|---|---|
| `id` | `<category>-NN`. Stable — the per-query regression report keys on it. |
| `category` | `bug` / `architecture` / `dependency` / `locator` |
| `question` | Natural language. No class names unless a user would plausibly say them. |
| `gold_files` | Paths exactly as `file_path` appears in Qdrant. 1–4 files. |
| `notes` | Why these files. For you, in three weeks, when a number moves. |

---

## The four categories — roughly 8 of each

**`bug`** — diagnostic, the reference shape.
> "Why would a request for a non-existent owner return a 500 instead of a 404?"

The hardest category and the one that justifies hybrid search: the question
shares almost no vocabulary with the code that answers it.

**`architecture`** — how something is structured or flows.
> "How does pagination work when listing owners?"

**`dependency`** — what touches what.
> "What would break if I changed the Vet entity?"

Tests the symbol graph and expansion more than raw retrieval.

**`locator`** — find a specific thing.
> "Where is the form for adding a visit to a pet?"

The easy category. Sparse should dominate here. If BM25-only *doesn't* beat
dense-only on locator questions, something is wrong in `sparse.py`.

Keeping categories separate matters: reporting Recall@10 per category is far
more informative than one average, and it tells you *which kind* of question
each component helps with.

---

## Labelling rules

1. **A file is gold if a competent engineer would need to read it** to answer.
   Not "mentions the topic" — *needed*.
2. **Prefer 2–3 gold files over 1.** A single-file answer makes Recall@10 a
   coin flip and adds noise to the average.
3. **Never exceed 4.** Beyond that the question is too broad to be diagnostic.
4. **Exclude test files** unless the question is about tests. They duplicate
   production vocabulary and will mask a retrieval failure.
5. **Verify every path exists in the index**, not just on disk. A typo silently
   becomes a permanent zero that no configuration can fix.
6. **Write the question before you look for the answer.** Looking first leaks
   the code's vocabulary into the question, which is the circularity you are
   trying to avoid.

---

## Procedure

```bash
# 1. index the repo
make ingest REPO=https://github.com/spring-projects/spring-petclinic

# 2. dump the indexed file list to label against
uv run python -c "
import asyncio
from codemind.retrieval.qdrant_store import QdrantStore
async def main():
    store = QdrantStore()
    points, _ = await store.client.scroll(store.collection_name, limit=10000, with_payload=['file_path'])
    for path in sorted({p.payload['file_path'] for p in points}):
        print(path)
    await store.close()
asyncio.run(main())
" > evaluation/datasets/indexed_files.txt

wc -l evaluation/datasets/indexed_files.txt

# 3. write the 30 questions, copying paths from that file

# 4. validate before running the ablation
uv run python evaluation/validate_goldset.py
```

Step 4 is not optional. A path that does not exist in the index is a silent
permanent zero, and you will spend an hour blaming the retriever.

---

## Seed rows

Replace the paths with real ones from `indexed_files.txt`. These show the shape
of each category; the remaining 26 are yours.

```json
{"id": "bug-01", "category": "bug", "question": "Why would submitting the owner form with a blank last name not show an error?", "gold_files": ["REPLACE/OwnerController.java", "REPLACE/Owner.java"], "notes": "bean validation annotations on the entity, binding result handled in the controller"}
{"id": "arch-01", "category": "architecture", "question": "How are database tables created when the application starts?", "gold_files": ["REPLACE/schema.sql", "REPLACE/PetClinicApplication.java"], "notes": "schema init is declarative, not code"}
{"id": "dep-01", "category": "dependency", "question": "What code would I have to change if I renamed the Vet entity?", "gold_files": ["REPLACE/Vet.java", "REPLACE/VetController.java", "REPLACE/VetRepository.java"], "notes": "tests the symbol graph, not just lexical matching"}
{"id": "loc-01", "category": "locator", "question": "Where is the pet type dropdown populated?", "gold_files": ["REPLACE/PetTypeFormatter.java"], "notes": "sparse should win this one outright"}
```

---

## What Gate 1 must produce

An ablation table with a row per configuration:

| Configuration | Recall@5 | Recall@10 | MRR | nDCG@10 | Misses |
|---|---|---|---|---|---|
| Dense only | | | | | |
| BM25 only | | | | | |
| Hybrid (RRF) | | | | | |
| Hybrid (DBSF) | | | | | |
| Hybrid + rerank | | | | | |
| Hybrid + rerank + expansion | | | | | |

Plus, per configuration, **which questions regressed** against the previous row.
A mean can rise while individual questions break, and naming those cases is what
separates a measured result from a claimed one.

**Hybrid must beat dense-only.** If it does not, do not proceed to the agents —
something in `sparse.py`, the tokenizer, or the IDF modifier is wrong, and
building on it will cost far more to debug later.
