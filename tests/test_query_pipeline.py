from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from backend.app.models.intent import IntentFilter
from backend.app.services import query_pipeline
from backend.app.utils.camera_ids import CameraIdFormat, normalize_camera_id, parse_camera_id_from_query


def test_camera_id_parser_supports_common_query_forms() -> None:
    assert parse_camera_id_from_query("cam 1") == "1"
    assert parse_camera_id_from_query("camera 1") == "1"
    assert parse_camera_id_from_query("CAM_01") == "1"
    assert parse_camera_id_from_query("cam01") == "1"
    assert parse_camera_id_from_query("CAM01") == "1"
    assert parse_camera_id_from_query("01") == "01"
    assert parse_camera_id_from_query("1") == "1"


def test_normalize_camera_id_matches_numeric_storage() -> None:
    storage = CameraIdFormat()
    assert normalize_camera_id("cam 1", storage) == "1"
    assert normalize_camera_id("camera 1", storage) == "1"
    assert normalize_camera_id("CAM_01", storage) == "1"
    assert normalize_camera_id("cam01", storage) == "1"
    assert normalize_camera_id("CAM01", storage) == "1"
    assert normalize_camera_id("01", storage) == "1"
    assert normalize_camera_id("1", storage) == "1"


def test_run_query_normalizes_camera_id_before_building_where(monkeypatch) -> None:
    vector_store = SimpleNamespace(normalize_camera_id=lambda value: "1")
    captured = {}

    monkeypatch.setattr(
        query_pipeline,
        "extract_intent",
        lambda query: IntentFilter(camera_id="CAM_01", label="person", semantic_query="person"),
    )

    def fake_search_alerts(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(query_pipeline, "search_alerts", fake_search_alerts)

    result = query_pipeline.run_query(
        query="show person from camera 1",
        vector_store=vector_store,
        top_k=5,
        min_score=0.0,
    )

    assert result.where_clause == {"$and": [{"camera_id": {"$eq": "1"}}, {"label": {"$eq": "person"}}]}
    assert captured["where"] == result.where_clause
    assert captured["query"] == "person"


def test_run_query_skips_trivial_greeting(monkeypatch) -> None:
    vector_store = SimpleNamespace(normalize_camera_id=lambda value: value)
    search_mock = MagicMock(return_value=[])

    monkeypatch.setattr(
        query_pipeline,
        "extract_intent",
        lambda query: IntentFilter(semantic_query=query),
    )
    monkeypatch.setattr(query_pipeline, "search_alerts", search_mock)

    result = query_pipeline.run_query(
        query="hi",
        vector_store=vector_store,
        top_k=5,
        min_score=0.0,
    )

    assert result.results == []
    assert result.where_clause is None
    search_mock.assert_not_called()