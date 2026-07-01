"""
Intent model.

Structured output produced by the LLM intent-extraction step.
All metadata fields are Optional; None means "no filter on this field".
`semantic_query` is always present and is passed to CLIP for vector search.
"""

from __future__ import annotations

from typing import Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator


class IntentFilter(BaseModel):
    """
    Structured representation of a user's natural-language surveillance query.

    Produced by intent.py and consumed by query_pipeline.py.

    Metadata fields
    ---------------
    camera_id       : Chroma field "camera_id"  — exact match
    label           : Chroma field "label"       — exact match (person/vehicle/…)
    alert_type      : Chroma field "alert_type"  — exact match
    date            : Calendar date in YYYY-MM-DD format
    time_after      : Lower bound on time-of-day in HH:MM (24-h)
    time_before     : Upper bound on time-of-day in HH:MM (24-h)
    min_confidence  : Minimum detection confidence in [0.0, 1.0]

    Semantic field
    --------------
    semantic_query  : Cleaned natural-language string passed to CLIP.
                      Always required; falls back to the original user query
                      if no other filters were identified.

    Extensibility
    -------------
    Add new optional fields here as the metadata schema grows (e.g. zone,
    object_count, ocr_text).  query_pipeline.py is the only place that maps
    these fields to Chroma `where` clauses, so adding a field here + a
    mapping branch there is sufficient.
    """

    # ── Metadata filters ──────────────────────────────────────────────────────

    camera_id: Optional[str] = Field(
        default=None,
        description="Camera identifier, e.g. '2' or 'CAM_05'.",
    )
    label: Optional[str] = Field(
        default=None,
        description="Object class label, e.g. 'person', 'vehicle', 'car'.",
    )
    alert_type: Optional[str] = Field(
        default=None,
        description="Alert classification, e.g. 'motion', 'intrusion'.",
    )
    date: Optional[str] = Field(
        default=None,
        description="Specific calendar date in YYYY-MM-DD format.",
    )
    time_after: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("time_after", "timestamp_start"),
        description="Lower bound on time-of-day in HH:MM (24-h), e.g. '15:00'.",
    )
    time_before: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("time_before", "timestamp_end"),
        description="Upper bound on time-of-day in HH:MM (24-h), e.g. '18:00'.",
    )
    min_confidence: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("min_confidence", "confidence_min"),
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence in [0, 1].",
    )

    # ── Semantic field ────────────────────────────────────────────────────────

    semantic_query: Optional[str] = Field(
        default=None,
        description=(
            "Cleaned description passed to CLIP for vector similarity search. "
            "If omitted by the LLM, it will be derived from the label or original query."
        ),
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("camera_id", "label", "alert_type", mode="before")
    @classmethod
    def _empty_to_none(cls, v: object) -> Optional[str]:
        """Convert empty strings to None so downstream code can use `if field`."""
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator("time_after", "time_before", mode="before")
    @classmethod
    def _normalise_time(cls, v: object) -> Optional[str]:
        """Accept 'HH:MM', 'HH:MM:SS', or None; return 'HH:MM' or None."""
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        parts = s.split(":")
        if len(parts) < 2:  # noqa: PLR2004
            raise ValueError(f"Invalid time format: {s!r}. Expected HH:MM.")
        return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}"

    @field_validator("date", mode="before")
    @classmethod
    def _normalise_date(cls, v: object) -> Optional[str]:
        """
        Accept any recognisable date string and normalise to YYYY-MM-DD.

        Previously this validator only accepted strings already in YYYY-MM-DD
        format — anything else silently became None because the ValueError was
        swallowed by Pydantic.  Now we call parse_date() first so natural
        language like "27 june" or "yesterday" is resolved before the
        structural check.
        """
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None

        # Fast path: already in YYYY-MM-DD format
        if len(s) == 10 and s[4] == "-" and s[7] == "-":  # noqa: PLR2004
            return s

        # Slow path: try to parse natural-language date
        from backend.app.utils.date_parser import parse_date  # local import avoids circularity
        parsed = parse_date(s)
        if parsed:
            return parsed

        # If nothing matched, raise so Pydantic sets the field to None
        # (consistent with the original behaviour for truly unrecognisable values)
        raise ValueError(f"Unrecognisable date: {s!r}. Expected YYYY-MM-DD or natural language.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def has_metadata_filters(self) -> bool:
        """Return True if at least one metadata filter is set."""
        return any(
            v is not None
            for v in (
                self.camera_id,
                self.label,
                self.alert_type,
                self.date,
                self.time_after,
                self.time_before,
                self.min_confidence,
            )
        )

    def __repr__(self) -> str:  # pragma: no cover
        fields = {k: v for k, v in self.model_dump().items() if v is not None}
        return f"IntentFilter({fields})"