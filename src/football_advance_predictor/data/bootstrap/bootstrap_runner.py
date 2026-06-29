"""End-to-end bootstrap orchestrator.

`BootstrapRunner` runs the full data pipeline:

1. Download all required (and optional) pinned sources.
2. Validate the raw files.
3. Open the alias registry (auto-seeded with built-in defaults).
4. Seed the alias registry with observed source names.
5. Build the knockout manifest across all providers.
6. Optionally print a report and write artifacts.

It is intentionally side-effecting only within the configured
directories. It never reaches out to the network unless
``offline=False`` and a cache miss occurs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.data.aliases.alias_registry import AliasRegistry
from football_advance_predictor.data.bootstrap.sha_resolver import (
    parse_github_repo_from_url,
    resolve_repo_head_sha,
)
from football_advance_predictor.data.bootstrap.source_downloader import (
    DownloadResult,
    SourceDownloader,
)
from football_advance_predictor.data.bootstrap.source_lock import SourceLock
from football_advance_predictor.data.bootstrap.source_registry import SourceRegistry
from football_advance_predictor.data.knockout.manifest import (
    KnockoutManifest,
    KnockoutManifestBuilder,
)
from football_advance_predictor.data.sources.martj42 import MartJ42ResultsProvider
from football_advance_predictor.data.sources.openfootball import OpenFootballTournamentProvider
from football_advance_predictor.data.sources.statsbomb_open_data import (
    StatsBombOpenDataProvider,
)

logger = get_logger(__name__)


@dataclass
class BootstrapReport:
    """The output of a single bootstrap run."""

    generated_at: datetime
    required_sources: list[DownloadResult] = field(default_factory=list)
    optional_sources: list[DownloadResult] = field(default_factory=list)
    alias_registry_size: int = 0
    unresolved_aliases: list[str] = field(default_factory=list)
    knockout_manifest: KnockoutManifest | None = None
    statsbomb_available: bool = False
    feature_coverage: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "required_sources": [
                {
                    "name": r.source.name,
                    "local_path": str(r.local_path),
                    "schema_valid": r.schema_valid,
                    "cache_hit": r.cache_hit,
                    "sha256": r.sha256[:16] + "...",
                }
                for r in self.required_sources
            ],
            "optional_sources": [
                {"name": r.source.name, "cache_hit": r.cache_hit, "schema_valid": r.schema_valid}
                for r in self.optional_sources
            ],
            "alias_registry_size": self.alias_registry_size,
            "unresolved_aliases_count": len(self.unresolved_aliases),
            "unresolved_aliases_sample": self.unresolved_aliases[:25],
            "knockout_manifest": {
                "total": self.knockout_manifest.total if self.knockout_manifest else 0,
                "tournament_coverage": (
                    self.knockout_manifest.tournament_coverage if self.knockout_manifest else {}
                ),
                "quarantined_count": (
                    len(self.knockout_manifest.quarantined) if self.knockout_manifest else 0
                ),
            },
            "statsbomb_available": self.statsbomb_available,
            "feature_coverage": self.feature_coverage,
            "errors": self.errors,
        }


class BootstrapRunner:
    """Run the full self-bootstrap pipeline.

    Args:
        registry_path: Path to the source registry JSON.
        raw_dir: Directory where raw downloaded files are cached.
        aliases_dir: Directory where the alias registry is persisted.
        artifacts_dir: Directory where generated artifacts (manifest
            CSV/JSON) are written.
        offline: If True, never reach the network.
    """

    def __init__(
        self,
        registry_path: str | Path,
        raw_dir: str | Path,
        aliases_dir: str | Path,
        artifacts_dir: str | Path,
        *,
        offline: bool = False,
        allow_first_run_resolution: bool = True,
    ) -> None:
        self.registry = SourceRegistry(registry_path)
        self.raw_dir = Path(raw_dir)
        self.aliases_dir = Path(aliases_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.raw_dir / "lock.json"
        self.lock = SourceLock.load(self.lock_path)
        self.downloader = SourceDownloader(
            self.registry, self.lock, self.raw_dir, offline=offline,
        )
        self._allow_first_run_resolution = allow_first_run_resolution

    def update_sources(self) -> BootstrapReport:
        """Explicitly re-resolve every source's HEAD SHA and update the lock.

        This is the only path that intentionally fetches a newer HEAD.
        Ordinary ``bootstrap()`` uses the locked SHAs.
        """
        from datetime import UTC, datetime

        report = BootstrapReport(generated_at=datetime.now(tz=UTC))
        # Refresh required sources.
        for spec in self.registry.all_required():
            try:
                # Force re-resolution by temporarily clearing the lock entry.
                if self.lock.is_locked(spec.name):
                    del self.lock.sources[spec.name]
                result = self.downloader.download(spec.name)
                report.required_sources.append(result)
            except Exception as exc:
                report.errors.append(
                    f"Failed to refresh {spec.name!r}: {exc}"
                )
                return report
        for spec in self.registry.all_optional():
            try:
                if self.lock.is_locked(spec.name):
                    del self.lock.sources[spec.name]
                result = self.downloader.download(spec.name)
                report.optional_sources.append(result)
            except Exception as exc:
                logger.warning("Optional source refresh failed", extra={"source": spec.name, "error": str(exc)})
        # Persist the new lock.
        self.lock.save(self.lock_path)
        # Re-run the rest of the bootstrap.
        report = self._finalize_report(report)
        return report

    def run(self) -> BootstrapReport:
        from datetime import UTC, datetime

        report = BootstrapReport(generated_at=datetime.now(tz=UTC))

        # 1. Download required sources. Once locked, the downloader will
        # refuse to fall back to HEAD.
        required_specs = self.registry.all_required()
        for spec in required_specs:
            try:
                result = self.downloader.download(
                    spec.name,
                    allow_first_run_resolution=self._allow_first_run_resolution,
                )
                report.required_sources.append(result)
            except Exception as exc:
                report.errors.append(
                    f"Failed to download required source {spec.name!r}: {exc}"
                )
                logger.error("Required source download failed", extra={"source": spec.name, "error": str(exc)})
                return report  # Cannot proceed without required sources.
        # 2. Best-effort optional sources.
        for spec in self.registry.all_optional():
            try:
                result = self.downloader.download(
                    spec.name,
                    allow_first_run_resolution=self._allow_first_run_resolution,
                )
                report.optional_sources.append(result)
            except Exception as exc:
                logger.warning(
                    "Optional source skipped",
                    extra={"source": spec.name, "error": str(exc)},
                )

        # 3. Persist the (possibly first-run) lock.
        self.lock.save(self.lock_path)
        return self._finalize_report(report)

    def _finalize_report(self, report: BootstrapReport) -> BootstrapReport:

        # Alias registry.
        aliases = AliasRegistry.open(self.aliases_dir)
        # Seed the registry with names observed in the canonical result file.
        results_csv = self._resolve_path("martj42_results", "martj42_results.csv")
        if results_csv is not None:
            observed = self._observed_team_names(results_csv)
            added = aliases.seed_from_observed(observed, source="martj42_results")
            logger.info("Seeded alias registry", extra={"added": added, "size": aliases.size()})
        report.alias_registry_size = aliases.size()

        # 4. Build the knockout manifest. The openfootball providers come
        #    FIRST so that per-match round labels are used; the martj42
        #    provider (which has no per-row stage) is processed last
        #    and any duplicates are dropped by the dedup key.
        builder = KnockoutManifestBuilder(aliases)
        for source_name, default_name in (
            ("openfootball_worldcup_1990", "FIFA World Cup 1990"),
            ("openfootball_worldcup_1994", "FIFA World Cup 1994"),
            ("openfootball_worldcup_1998", "FIFA World Cup 1998"),
            ("openfootball_worldcup_2002", "FIFA World Cup 2002"),
            ("openfootball_worldcup_2006", "FIFA World Cup 2006"),
            ("openfootball_worldcup_2010", "FIFA World Cup 2010"),
            ("openfootball_worldcup_2014", "FIFA World Cup 2014"),
            ("openfootball_worldcup_2018", "FIFA World Cup 2018"),
            ("openfootball_worldcup_2022", "FIFA World Cup 2022"),
        ):
            try:
                spec = self.registry.get(source_name)
            except KeyError:
                continue
            target = self.raw_dir / (spec.local_filename or "")
            if not target.exists():
                continue
            provider = OpenFootballTournamentProvider(
                path=target, alias_registry=aliases, tournament_name=default_name
            )
            builder.add_provider(source_name, provider)
        results_csv = self._resolve_path("martj42_results", "martj42_results.csv")
        shootouts_csv = self._resolve_path("martj42_shootouts", "martj42_shootouts.csv")
        if results_csv is not None and shootouts_csv is not None:
            provider = MartJ42ResultsProvider(
                results_path=results_csv,
                shootouts_path=shootouts_csv,
                alias_registry=aliases,
            )
            provider.tournament_name = "international_friendly_and_competitive"
            builder.add_provider("martj42_results", provider)
        manifest = builder.build()
        report.knockout_manifest = manifest
        report.unresolved_aliases = aliases.unresolved_names()

        # 5. StatsBomb coverage (only if the local clone exists).
        sb_root = self._resolve_path("statsbomb_open_data", None)
        if sb_root is not None and sb_root.is_dir():
            report.statsbomb_available = (sb_root / "matches").is_dir()
        report.feature_coverage = {
            "statsbomb_events": report.statsbomb_available,
            "historical_odds": False,
            "lineups": False,
        }

        # 6. Persist artifacts.
        self._write_manifest_artifact(manifest)
        # Manifest download summary.
        manifest_path = self.artifacts_dir / "bootstrap_report.json"
        manifest_path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
        self.downloader.write_manifest(
            [*report.required_sources, *report.optional_sources],
            self.artifacts_dir / "source_manifest.json",
        )
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, source_name: str, filename: str | None) -> Path | None:
        spec = self.registry.get(source_name)
        if spec.local_filename and filename:
            return self.raw_dir / spec.local_filename
        if spec.local_path:
            return self.raw_dir / spec.local_path
        if filename:
            return self.raw_dir / filename
        return None

    @staticmethod
    def _observed_team_names(csv_path: Path) -> list[str]:
        import csv

        names: list[str] = []
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in ("home_team", "away_team"):
                    val = row.get(col)
                    if val:
                        names.append(val)
        return names

    def _write_manifest_artifact(self, manifest: KnockoutManifest) -> None:
        path = self.artifacts_dir / "knockout_match_manifest.json"
        path.write_text(
            json.dumps(manifest.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Wrote knockout manifest artifact", extra={"path": str(path)})
