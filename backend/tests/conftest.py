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

from typing import Any

import pytest
from fastapi.testclient import TestClient

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
