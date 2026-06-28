"""Time utilities for cutoff-aware feature computation.

Centralizing timestamp parsing and comparison keeps the anti-leakage
contract explicit throughout the codebase.
"""

from __future__ import annotations

from datetime import UTC, datetime


def to_utc(value: datetime | str) -> datetime:
    """Convert a datetime or ISO 8601 string into a timezone-aware UTC datetime.

    Args:
        value: Either a ``datetime`` (naive or aware) or an ISO 8601 string.

    Returns:
        A timezone-aware ``datetime`` in UTC.

    Raises:
        ValueError: If the value cannot be parsed.
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid ISO 8601 timestamp: {value!r}") from exc
    else:
        raise ValueError(f"Unsupported timestamp value: {value!r}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt


def assert_cutoff_before(cutoff: datetime, kickoff: datetime, *, label: str) -> None:
    """Assert that ``cutoff`` is strictly before ``kickoff``.

    Args:
        cutoff: The cutoff timestamp.
        kickoff: The match kickoff timestamp.
        label: Human-readable label for the assertion (used in error message).

    Raises:
        ValueError: If ``cutoff >= kickoff``.
    """
    cutoff_utc = to_utc(cutoff)
    kickoff_utc = to_utc(kickoff)
    if cutoff_utc >= kickoff_utc:
        raise ValueError(
            f"Cutoff time for {label!r} must be strictly before kickoff "
            f"(cutoff={cutoff_utc.isoformat()}, kickoff={kickoff_utc.isoformat()})."
        )


def assert_observed_before(observed_at: datetime, cutoff: datetime, *, label: str) -> None:
    """Assert that a piece of data was observed strictly before the cutoff.

    Args:
        observed_at: The observed/published timestamp of the data point.
        cutoff: The cutoff timestamp for a prediction.
        label: Human-readable label used in the error message.

    Raises:
        ValueError: If ``observed_at >= cutoff``.
    """
    obs_utc = to_utc(observed_at)
    cut_utc = to_utc(cutoff)
    if obs_utc >= cut_utc:
        raise ValueError(
            f"Data point {label!r} was observed at {obs_utc.isoformat()} "
            f"which is not strictly before cutoff {cut_utc.isoformat()}."
        )
