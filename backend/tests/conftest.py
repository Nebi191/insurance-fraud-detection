"""Shared test fixtures.

The `client` fixture uses `TestClient` as a CONTEXT MANAGER, which is mandatory:
Starlette only runs `lifespan` inside a `with` block. Had we written
`TestClient(app)` directly, `app.state.bundle` would never be set up and every
request would return a 500 with an AttributeError. So this fixture is also the
first answer to "does the artifact really load?".

Scope `session`: loading pipeline.pkl + the SHAP TreeExplainer is expensive and
there is no point repeating it per test.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi.testclient import TestClient

# `app = create_app()` executes at import time (see `app/main.py`), reading
# `RATE_LIMIT_PER_MINUTE` right then. The shared, SESSION-scoped `client`
# fixture below serves the entire suite — hundreds of `/predict` calls,
# including a 50-way concurrent burst in `test_concurrent_predictions_are_
# bitwise_identical` — so the DEFAULT limit (30/min) would make unrelated tests
# fail with 429 depending on run order. `setdefault` only supplies a practical
# "unlimited" ceiling for THIS shared app if the variable is not already set
# (e.g. by a developer probing the real default locally); tests that exercise
# the rate limiter itself build their OWN app via `create_app()` with their own
# small `RATE_LIMIT_PER_MINUTE`, so they are unaffected by this line.
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "100000")

# Must follow the setdefault() above.
from app.main import app
from app.model import ModelBundle


@pytest.fixture(scope="session")
def client() -> Any:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def bundle(client: TestClient) -> ModelBundle:
    """The very artifact the application loaded during lifespan.

    We do not call a separate `ModelBundle.load()`: the object the test verifies
    must be the SAME object the requests use.
    """
    return client.app.state.bundle


@pytest.fixture(scope="session")
def metadata(bundle: ModelBundle) -> dict[str, Any]:
    return bundle.metadata
