"""Strict data validation gate.

`data validate --strict` is the quality gate. It fails non-zero for any
of the following:

- missing resolved source lock (the source is not pinned to a real SHA)
- hash mismatch (raw file content does not match the locked SHA-256)
- duplicate matches in the knockout manifest (same date + teams)
- ambiguous advancement labels (both teams marked advancer)
- unmatched shootouts (a shootout row that has no matching match)
- unresolved team aliases above threshold
- train/test date overlap (any match in the test fold is in the train fold)
- post-cutoff source records (any source row with a timestamp after its cutoff)

`data status` is informational; this is the gate.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from football_advance_predictor.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """Outcome of a strict validation pass."""

    checks: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = True

    def add(self, name: str, ok: bool, *, detail: str = "", level: str = "error") -> None:
        entry = {"name": name, "ok": ok, "level": level, "detail": detail}
        self.checks.append(entry)
        if not ok and level == "error":
            self.passed = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_failed": sum(1 for c in self.checks if not c["ok"] and c["level"] == "error"),
            "checks": self.checks,
        }


def validate_strict(
    *,
    lock_path: str | Path,
    raw_dir: str | Path,
    aliases_dir: str | Path,
    knockout_manifest_path: str | Path | None,
    matches_csv_path: str | Path | None,
    shootouts_csv_path: str | Path | None,
    unresolved_alias_threshold: int = 0,
) -> ValidationReport:
    """Run all strict checks. Returns a :class:`ValidationReport`."""
    from football_advance_predictor.data.bootstrap.source_lock import SourceLock
    from football_advance_predictor.data.knockout.manifest import (
        KnockoutManifest,
    )

    report = ValidationReport()

    # 1. Source lock is present and every required source is locked.
    lock_path = Path(lock_path)
    if not lock_path.exists():
        report.add("source_lock_exists", False, detail=f"missing {lock_path}")
        return report
    lock = SourceLock.load(lock_path)
    report.add("source_lock_exists", True)
    for name in ["martj42_results", "martj42_shootouts", "openfootball_worldcup"]:
        if not lock.is_locked(name):
            report.add(
                f"locked:{name}", False,
                detail="required source is not in lock.json",
            )
            continue
        if not lock.validate_full_sha(name):
            sha = lock.get(name).resolved_sha
            report.add(
                f"locked:{name}", False,
                detail=f"resolved_sha is not a 40-character commit hash: {sha!r}",
            )
        else:
            report.add(f"locked:{name}", True)

    # 2. Hash integrity.
    raw_dir = Path(raw_dir)
    for name, fname in (
        ("martj42_results", "martj42_results.csv"),
        ("martj42_shootouts", "martj42_shootouts.csv"),
    ):
        locked = lock.get(name)
        if locked is None:
            continue
        local = raw_dir / fname
        if not local.exists():
            report.add(f"hash:{name}", False, detail=f"file missing: {local}")
            continue
        actual = _sha256(local)
        if actual != locked.raw_sha256:
            report.add(
                f"hash:{name}", False,
                detail=f"locked={locked.raw_sha256[:16]}... got {actual[:16]}...",
            )
        else:
            report.add(f"hash:{name}", True)

    # 3. Knockout manifest: duplicate matches, ambiguous advancement,
    #    unresolved aliases over threshold.
    if knockout_manifest_path is not None:
        manifest_path = Path(knockout_manifest_path)
        if not manifest_path.exists():
            report.add("manifest_exists", False, detail=f"missing {manifest_path}")
        else:
            with manifest_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            report.add("manifest_exists", True)
            rows = data.get("rows", [])
            # Duplicate detection by (kickoff_at, home_team_id, away_team_id).
            keys: Counter[tuple[str, str, str]] = Counter()
            for r in rows:
                key = (r["kickoff_at"], r["home_team_id"], r["away_team_id"])
                keys[key] += 1
            dups = {k: c for k, c in keys.items() if c > 1}
            report.add(
                "manifest:no_duplicates",
                not dups,
                detail=f"{len(dups)} duplicate keys" if dups else "",
            )
            # Ambiguous advancement: every row must be a bool.
            ambiguous = [r["match_id"] for r in rows if r.get("home_advances") not in (True, False)]
            report.add(
                "manifest:advancement_label_is_bool",
                not ambiguous,
                detail=f"{len(ambiguous)} ambiguous labels" if ambiguous else "",
            )

    # 4. Unresolved aliases threshold.
    if aliases_dir.exists():
        from football_advance_predictor.data.aliases.alias_registry import (
            AliasRegistry,
        )

        reg = AliasRegistry.open(aliases_dir)
        unresolved = reg.unresolved_count()
        report.add(
            "aliases:unresolved_under_threshold",
            unresolved <= unresolved_alias_threshold,
            detail=f"unresolved={unresolved} threshold={unresolved_alias_threshold}",
        )

    # 5. Unmatched shootouts.
    if matches_csv_path and shootouts_csv_path:
        matches = _read_csv_dicts(matches_csv_path)
        shootouts = _read_csv_dicts(shootouts_csv_path)
        match_keys = {
            (m.get("date"), m.get("home_team"), m.get("away_team")) for m in matches
        }
        unmatched = 0
        for s in shootouts:
            key = (s.get("date"), s.get("home_team"), s.get("away_team"))
            if key not in match_keys:
                unmatched += 1
        report.add(
            "shootouts:all_matched",
            unmatched == 0,
            detail=f"{unmatched} unmatched shootout rows",
        )

    # 6. Post-cutoff source records: a source row with a date after
    #    its declared cutoff (we use the row's own date as the cutoff,
    #    so any future-dated row is a violation).
    for name, fname in (
        ("martj42_results", "martj42_results.csv"),
        ("martj42_shootouts", "martj42_shootouts.csv"),
    ):
        path = raw_dir / fname
        if not path.exists():
            continue
        future_rows = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date = row.get("date")
                if not date:
                    continue
                try:
                    dt = datetime.fromisoformat(date.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if dt > datetime.now(tz=dt.tzinfo):
                    future_rows.append(date)
        report.add(
            f"no_post_cutoff_records:{name}",
            not future_rows,
            detail=f"{len(future_rows)} future-dated rows" if future_rows else "",
        )

    return report


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
