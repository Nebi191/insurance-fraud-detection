"""Ortak test fixture'ları.

`client` fixture'ı `TestClient`'ı CONTEXT MANAGER olarak kullanır — bu şart:
Starlette `lifespan`'i yalnızca `with` bloğunda çalıştırır. Doğrudan
`TestClient(app)` yazsaydık `app.state.bundle` hiç kurulmaz ve her istek
AttributeError ile 500 dönerdi. Yani bu fixture aynı zamanda "artefakt gerçekten
yükleniyor mu?" sorusunun ilk cevabıdır.

Scope `session`: pipeline.pkl + SHAP TreeExplainer yüklemesi pahalıdır, test
başına tekrarlamanın anlamı yok.
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
    """Uygulamanın lifespan sırasında yüklediği artefaktın ta kendisi.

    Ayrı bir `ModelBundle.load()` çağırmıyoruz: testin doğruladığı nesne,
    isteklerin kullandığı nesneyle AYNI olmalı.
    """
    return client.app.state.bundle


@pytest.fixture(scope="session")
def metadata(bundle: ModelBundle) -> dict[str, Any]:
    return bundle.metadata
