"""Create the CodeMind Qdrant collection. Safe to run any number of times.

    uv run python scripts/bootstrap_qdrant.py            # create if absent
    uv run python scripts/bootstrap_qdrant.py --recreate # drop and rebuild (destroys data)

Everything — URL, collection name, dimension, quantization, on_disk — comes
from Settings, so this and the API can never disagree about the schema.
Exits non-zero if Qdrant is unreachable or the existing collection does not
match the current settings (e.g. EMBEDDING_DIM changed).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from codemind.core.config import get_settings
from codemind.core.exceptions import RetrievalError
from codemind.retrieval.qdrant_store import QdrantStore


async def bootstrap(*, recreate: bool) -> int:
    settings = get_settings()
    store = QdrantStore(settings)
    try:
        created = await store.ensure_collection(recreate=recreate)
        count = await store.count()
    except (RetrievalError, UnexpectedResponse, ResponseHandlingException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        await store.close()

    verb = "created" if created else "already exists"
    print(
        f"collection {settings.qdrant_collection!r} {verb} at {settings.qdrant_url} "
        f"(dim={settings.embedding_dim}, int8={settings.qdrant_quantization}, "
        f"on_disk={settings.qdrant_on_disk}, points={count})"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--recreate", action="store_true", help="drop the collection first (destroys all points)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")
    sys.exit(asyncio.run(bootstrap(recreate=args.recreate)))


if __name__ == "__main__":
    main()
