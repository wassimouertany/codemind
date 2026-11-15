"""Shared fixtures for tests that need a live Qdrant server."""

from __future__ import annotations

import httpx
import pytest

from codemind.core.config import get_settings


def _qdrant_reachable(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="session")
def live_qdrant_url() -> str:
    """The configured Qdrant URL, or skip the requesting test when nothing answers there.

    One check and one skip reason for every live-server test, so the suite passes
    with or without a server and verifies the real thing whenever one is up.
    """
    url = get_settings().qdrant_url
    if not _qdrant_reachable(url):
        pytest.skip(f"no Qdrant reachable at {url}")
    return url
