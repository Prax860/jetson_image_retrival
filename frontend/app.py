"""
Imagify — Streamlit chatbot frontend.

Run with:
    streamlit run frontend/app.py

Requires the FastAPI backend to be running at IMAGIFY_API_URL (default: http://localhost:8000).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

API_URL = os.getenv("IMAGIFY_API_URL", "http://localhost:8000")
API_BASE = f"{API_URL}/api/v1"

st.set_page_config(
    page_title="Imagify — Alert Search",
    page_icon="🎥",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_stats() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE}/collections", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _search(
    query: str,
    top_k: int,
    camera_id: Optional[str],
    alert_type: Optional[str],
    min_score: float,
) -> Optional[Dict[str, Any]]:
    payload = {
        "query": query,
        "top_k": top_k,
        "min_score": min_score,
    }
    if camera_id:
        payload["camera_id"] = camera_id
    if alert_type:
        payload["alert_type"] = alert_type

    try:
        r = requests.post(f"{API_BASE}/search", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as exc:
        st.error(f"Search failed: {exc.response.text}")
        return None
    except Exception as exc:
        st.error(f"Cannot reach backend: {exc}")
        return None


def _format_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


def _render_results(results: List[Dict[str, Any]]) -> None:
    if not results:
        st.info("No matching alerts found.")
        return

    cols_per_row = 3
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, item in enumerate(results[i : i + cols_per_row]):
            with cols[j]:
                b64 = item.get("image_b64", "")
                if b64:
                    st.image(
                        f"data:image/jpeg;base64,{b64}",
                        width="stretch",
                        caption=f"Rank #{item['rank']} · score {item['score']:.3f}",
                    )
                else:
                    st.warning("Image not available")

                st.caption(
                    f"📷 **{item['camera_id']}**  \n"
                    f"🕐 {_format_ts(item['timestamp'])}  \n"
                    + (f"🏷 {item['alert_type']}  \n" if item.get("alert_type") else "")
                    + (f"🎯 confidence {item['confidence']:.2f}" if item.get("confidence") is not None and item['confidence'] >= 0 else "")
                )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎥 Imagify")
    st.caption("Jetson Alert Visual Search")
    st.divider()

    stats = _get_stats()
    if stats:
        st.metric("Indexed Alerts", stats.get("total_alerts", 0))
        cameras = stats.get("cameras", [])
        if cameras:
            st.caption(f"Cameras: {', '.join(cameras)}")
    else:
        st.warning("Backend offline or no data yet.")

    st.divider()
    st.subheader("Search Filters")
    top_k = st.slider("Max results", 1, 50, 10)
    min_score = st.slider("Min similarity", 0.0, 1.0, 0.0, step=0.05)

    # Camera filter (populated from stats)
    cam_options = ["(all cameras)"] + (stats.get("cameras", []) if stats else [])
    cam_sel = st.selectbox("Camera", cam_options)
    camera_filter = None if cam_sel == "(all cameras)" else cam_sel

    alert_type_input = st.text_input("Alert type filter", placeholder="person / motion / vehicle")
    alert_type_filter = alert_type_input.strip() or None

    st.divider()
    st.caption("Backend: " + API_URL)


# ── Main chat area ────────────────────────────────────────────────────────────

st.header("Ask about your alert footage")
st.caption("Examples: *person in a red jacket*, *vehicle near gate at night*, *motion on camera 3*")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role", "content", "results"}

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("results"):
            _render_results(msg["results"])


# Chat input
if stats is None:
    st.info("Backend is offline. You can still type a query, but search will fail until the API is reachable.")
elif stats.get("total_alerts", 0) == 0:
    st.info("No alerts are indexed yet. You can still type a query now and search once data is ingested.")

user_query = st.chat_input(
    "Describe what you're looking for…",
)

if user_query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query, "results": None})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Run search
    with st.chat_message("assistant"):
        with st.spinner("Searching alert footage…"):
            data = _search(
                query=user_query,
                top_k=top_k,
                camera_id=camera_filter,
                alert_type=alert_type_filter,
                min_score=min_score,
            )

        if data is not None:
            results = data.get("results", [])
            total = data.get("total", 0)
            reply = (
                f"Found **{total}** alert{'s' if total != 1 else ''} matching *\"{user_query}\"*."
                if total > 0
                else f"No alerts matched *\"{user_query}\"*. Try broader terms or lower the similarity threshold."
            )
            st.markdown(reply)
            _render_results(results)

            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "results": results,
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Search failed — see error above.",
                "results": [],
            })
