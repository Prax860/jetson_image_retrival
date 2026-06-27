from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

mock_vs = MagicMock()
mock_vs.count.return_value = 0
mock_vs.list_cameras.return_value = []

with patch("backend.app.repositories.vector_store.get_vector_store", return_value=mock_vs):
    from backend.app.main import app

    with TestClient(app) as c:
        r = c.get("/api/v1/health")
        print("status:", r.status_code)
        try:
            print("json:", r.json())
        except Exception as exc:
            print("content:", r.content)
            print("error parsing json:", exc)
