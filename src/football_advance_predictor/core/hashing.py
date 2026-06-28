"""Hashing utilities used for immutable feature snapshots and lineage."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(payload: Any) -> str:
    """Compute a stable SHA-256 hex digest for a JSON-serializable payload.

    Args:
        payload: Any value that is JSON-serializable (dicts, lists, primitives).

    Returns:
        Hex-encoded SHA-256 digest.
    """
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
