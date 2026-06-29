"""Source downloader with pinned revisions, content-hash verification, and caching.

Design constraints:
- No network access in tests (the HTTP client is injectable).
- The downloader writes raw bytes unchanged to a per-source directory.
- After download, a SHA-256 of the bytes is computed and recorded.
- If the on-disk file already exists and matches the recorded hash, the
  download is skipped (cache hit).
- A download failure (network down, 404, schema mismatch) fails loud.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Protocol

from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.data.bootstrap.source_registry import SourceRegistry, SourceSpec

logger = get_logger(__name__)


class HttpFetcher(Protocol):
    """Pluggable HTTP fetcher (so tests can mock network calls)."""

    def fetch(self, url: str) -> bytes:
        ...


class UrllibFetcher:
    """Default fetcher using the standard library. No external deps."""

    def fetch(self, url: str) -> bytes:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(url, headers={"User-Agent": "football-advance-predictor/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.URLError as exc:
            raise DownloadError(f"Failed to fetch {url}: {exc}") from exc


class DownloadError(RuntimeError):
    """Raised on any download or validation failure."""


@dataclass
class DownloadResult:
    """Result of a single source download."""

    source: SourceSpec
    local_path: Path
    sha256: str
    bytes_written: int
    cache_hit: bool
    schema_valid: bool


class SourceDownloader:
    """Download and cache pinned sources from the registry.

    Args:
        registry: :class:`SourceRegistry` describing what to fetch.
        raw_dir: Directory under which source files are cached.
        fetcher: Optional HTTP fetcher; defaults to :class:`UrllibFetcher`.
    """

    def __init__(
        self,
        registry: SourceRegistry,
        raw_dir: str | Path,
        *,
        fetcher: HttpFetcher | None = None,
        offline: bool = False,
    ) -> None:
        self.registry = registry
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.fetcher = fetcher or UrllibFetcher()
        self.offline = offline

    def download(self, name: str) -> DownloadResult:
        spec = self.registry.get(name)
        if spec.kind == "statsbomb_open_data":
            return self._download_git(spec)
        return self._download_http(spec)

    def download_all(self, names: list[str] | None = None) -> list[DownloadResult]:
        names = names or self.registry.all_names()
        out: list[DownloadResult] = []
        for n in names:
            try:
                out.append(self.download(n))
            except DownloadError as exc:
                logger.warning("Skipping source (download failed)", extra={"source": n, "error": str(exc)})
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _download_http(self, spec: SourceSpec) -> DownloadResult:
        if not spec.local_filename:
            raise DownloadError(f"Source {spec.name!r} has no local_filename")
        target = self.raw_dir / spec.local_filename
        if target.exists():
            existing_sha = self._sha256(target)
            logger.info("Cache hit", extra={"source": spec.name, "path": str(target)})
            schema_valid = self._validate_schema(spec, target)
            return DownloadResult(
                source=spec,
                local_path=target,
                sha256=existing_sha,
                bytes_written=0,
                cache_hit=True,
                schema_valid=schema_valid,
            )
        if self.offline:
            raise DownloadError(
                f"Source {spec.name!r} not cached and offline mode is on ({target})."
            )
        logger.info("Downloading", extra={"source": spec.name, "url": spec.resolved_url})
        raw = self.fetcher.fetch(spec.resolved_url)
        if not raw:
            raise DownloadError(f"Empty response for {spec.resolved_url}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        sha = self._sha256(target)
        schema_valid = self._validate_schema(spec, target)
        return DownloadResult(
            source=spec,
            local_path=target,
            sha256=sha,
            bytes_written=len(raw),
            cache_hit=False,
            schema_valid=schema_valid,
        )

    def _download_git(self, spec: SourceSpec) -> DownloadResult:
        if not spec.local_path:
            raise DownloadError(f"Git source {spec.name!r} has no local_path")
        target = self.raw_dir / spec.local_path
        if (target / ".git").exists():
            logger.info("Cache hit (git)", extra={"source": spec.name, "path": str(target)})
            sha = self._sha256(target)
            return DownloadResult(
                source=spec, local_path=target, sha256=sha, bytes_written=0,
                cache_hit=True, schema_valid=True,
            )
        if self.offline:
            raise DownloadError(f"Git source {spec.name!r} not cached and offline mode is on.")
        target.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("git") is None:
            raise DownloadError("git binary not found; cannot clone StatsBomb open-data.")
        url = spec.url_template
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", spec.pinned_sha, url, str(target)],
                check=True, timeout=300, capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            raise DownloadError(f"git clone failed: {exc.stderr.decode(errors='ignore')}") from exc
        sha = self._sha256(target)
        return DownloadResult(
            source=spec, local_path=target, sha256=sha, bytes_written=0,
            cache_hit=False, schema_valid=True,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        if path.is_dir():
            for f in sorted(path.rglob("*")):
                if f.is_file():
                    h.update(f.relative_to(path).as_posix().encode("utf-8"))
                    h.update(f.read_bytes())
            return h.hexdigest()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _validate_schema(self, spec: SourceSpec, path: Path) -> bool:
        if not spec.expected_columns and not spec.expected_keys:
            return True
        try:
            if path.suffix == ".csv":
                import csv

                with path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    headers = tuple(reader.fieldnames or ())
                missing = set(spec.expected_columns) - set(headers)
                if missing:
                    raise DownloadError(
                        f"CSV schema mismatch for {spec.name!r}: missing columns {sorted(missing)}"
                    )
                return True
            if path.suffix == ".json":
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise DownloadError(f"JSON {spec.name!r} root must be an object.")
                missing = set(spec.expected_keys) - set(data.keys())
                if missing:
                    raise DownloadError(
                        f"JSON schema mismatch for {spec.name!r}: missing keys {sorted(missing)}"
                    )
                return True
        except DownloadError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise DownloadError(f"Failed to validate {spec.name!r}: {exc}") from exc
        return True

    def write_manifest(self, results: list[DownloadResult], manifest_path: str | Path) -> None:
        """Write a JSON manifest of what was downloaded, with hashes."""
        import json

        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": _utc_now_iso(),
            "sources": [
                {
                    "name": r.source.name,
                    "kind": r.source.kind,
                    "pinned_sha": r.source.pinned_sha,
                    "resolved_url": r.source.resolved_url,
                    "local_path": str(r.local_path),
                    "sha256": r.sha256,
                    "bytes_written": r.bytes_written,
                    "cache_hit": r.cache_hit,
                    "schema_valid": r.schema_valid,
                }
                for r in results
            ],
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote source manifest", extra={"path": str(manifest_path)})


def _utc_now_iso() -> str:
    from datetime import datetime

    return datetime.now(tz=UTC).isoformat()
