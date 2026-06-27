"""
Basic smoke tests — no CLIP model or Chroma needed.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def client():
    # Patch vector store so tests don't need Chroma on disk
    mock_vs = MagicMock()
    mock_vs.count.return_value = 0
    mock_vs.list_cameras.return_value = []

    # Use `new=` with a lambda to avoid Mock objects exposing a (*args, **kwargs)
    # signature which FastAPI would interpret as required query params.
    with patch("backend.app.repositories.vector_store.get_vector_store", new=lambda: mock_vs):
        from backend.app.main import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


def test_collections(client):
    r = client.get("/api/v1/collections")
    assert r.status_code == 200
    data = r.json()
    assert "total_alerts" in data
    assert "cameras" in data
