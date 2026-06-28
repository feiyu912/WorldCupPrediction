"""Agent enrichment interfaces.

These are deterministic, mock-friendly interfaces that downstream
LLM-powered tools can implement. The interfaces are deliberately
narrow and structured; no LLM may alter probabilities directly.

Each method returns a structured payload plus explicit evidence
metadata so every claim is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass
class AgentEvidence:
    """Evidence metadata for an agent-produced claim."""

    source_url: str | None
    observed_at: datetime
    published_at: datetime | None
    raw_excerpt: str
    confidence: float = 0.5


@dataclass
class AgentOutput:
    """Standard structured output of an agent interface.

    The fields are domain-specific data extracted from text, not
    probabilities. The agent NEVER returns a probability directly.
    """

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    evidence: list[AgentEvidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class AvailabilityExtractor(Protocol):
    """Convert free-form text into structured availability records."""

    name: str

    def extract(
        self, *, raw_text: str, observed_at: datetime, source_url: str | None = None
    ) -> AgentOutput:
        ...


@runtime_checkable
class NewsTimestampValidator(Protocol):
    """Confirm whether a piece of news is published before a cutoff."""

    name: str

    def validate(
        self, *, item: dict[str, Any], cutoff: datetime
    ) -> AgentOutput:
        ...


@runtime_checkable
class SourceReliabilityScorer(Protocol):
    """Score the reliability of a source URL."""

    name: str

    def score(self, *, source_url: str) -> AgentOutput:
        ...


@runtime_checkable
class MatchReviewAssistant(Protocol):
    """Produce a post-match error analysis (factual)."""

    name: str

    def review(
        self,
        *,
        prediction_id: str,
        actual_outcome: dict[str, Any],
    ) -> AgentOutput:
        ...


# ---------------------------------------------------------------------------
# Deterministic mock implementations
# ---------------------------------------------------------------------------


class MockAvailabilityExtractor:
    """Trivial deterministic extractor.

    Useful for tests and as a default implementation. Production code
    should swap in an LLM-backed implementation behind the same
    protocol.
    """

    name = "mock_availability_extractor"

    def extract(
        self, *, raw_text: str, observed_at: datetime, source_url: str | None = None
    ) -> AgentOutput:
        text = raw_text.lower()
        data: dict[str, Any] = {}
        if "out" in text and "confirmed" in text:
            data["availability_status"] = "confirmed_out"
        elif "doubt" in text:
            data["availability_status"] = "doubtful"
        elif "lineup" in text and "confirmed" in text:
            data["availability_status"] = "lineup_confirmed"
        else:
            data["availability_status"] = "available"
        return AgentOutput(
            success=True,
            data=data,
            evidence=[
                AgentEvidence(
                    source_url=source_url,
                    observed_at=observed_at,
                    published_at=observed_at,
                    raw_excerpt=raw_text[:200],
                )
            ],
        )


class MockNewsTimestampValidator:
    """Validates that an item's observed_at is strictly before the cutoff."""

    name = "mock_news_timestamp_validator"

    def validate(self, *, item: dict[str, Any], cutoff: datetime) -> AgentOutput:
        observed_at = item.get("observed_at")
        success = False
        warnings: list[str] = []
        if observed_at is None:
            warnings.append("Missing observed_at")
        else:
            from football_advance_predictor.core.time import to_utc

            success = to_utc(observed_at) < to_utc(cutoff)
        return AgentOutput(
            success=success,
            data={"valid": success},
            evidence=[
                AgentEvidence(
                    source_url=item.get("source_url"),
                    observed_at=observed_at or cutoff,
                    published_at=item.get("published_at"),
                    raw_excerpt=str(item.get("raw_text", ""))[:200],
                )
            ],
            warnings=warnings,
        )


class MockSourceReliabilityScorer:
    """Assigns a reliability score by URL suffix heuristics."""

    name = "mock_source_reliability_scorer"

    def score(self, *, source_url: str) -> AgentOutput:
        score = 0.5
        if not source_url:
            return AgentOutput(success=False, data={"score": score})
        lowered = source_url.lower()
        if any(s in lowered for s in ("reuters", "apnews", "bbc", "uefa.com", "fifa.com")):
            score = 0.9
        elif any(s in lowered for s in ("twitter", "x.com", "reddit")):
            score = 0.3
        return AgentOutput(success=True, data={"score": score})


class MockMatchReviewAssistant:
    """Produces a placeholder post-match error summary."""

    name = "mock_match_review_assistant"

    def review(
        self,
        *,
        prediction_id: str,
        actual_outcome: dict[str, Any],
    ) -> AgentOutput:
        return AgentOutput(
            success=True,
            data={
                "prediction_id": prediction_id,
                "actual": actual_outcome,
                "summary": "Deterministic mock review; replace with LLM in production.",
            },
        )
