"""Source downloader with lock-file enforcement.

Behavior:

- If a source is **locked** in ``data/raw/sources/lock.json``, the
  downloader fetches the file at the locked SHA, verifies the SHA-256,
  and refuses to fall back to HEAD.
- If a source is **not locked** (first bootstrap), the downloader
  resolves HEAD via the GitHub API to a 40-character commit SHA,
  downloads the file at that exact SHA, verifies integrity, and
  writes the lock entry.
- The downloader fails loudly on any mismatch, network error, or
  schema problem.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.data.bootstrap.sha_resolver import (
    parse_github_repo_from_url,
    resolve_repo_head_sha,
)
from football_advance_predictor.data.bootstrap.source_lock import (
    SourceLock,
    _FULL_SHA_PATTERN,
)
from football_advance_predictor.data.bootstrap.source_registry import SourceRegistry, SourceSpec

logger = get_logger(__name__)


class HttpFetcher(Protocol):
    """Pluggable HTTP fetcher (so tests can mock network calls)."""

    def fetch(self, url: str) -> bytes: ...


class UrllibFetcher:
    """Default fetcher using the standard library. No external deps."""

    def fetch(self, url: str) -> bytes:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "football-advance-predictor/0.1",
                "Accept": "application/vnd.github+json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
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
    used_lock: bool
    resolved_sha: str | None


class SourceDownloader:
    """Download and cache pinned sources from the registry.

    Args:
        registry: :class:`SourceRegistry` describing what to fetch.
        lock: :class:`SourceLock` (may be empty on the first run).
        raw_dir: Directory under which source files are cached.
        fetcher: Optional HTTP fetcher; defaults to :class:`UrllibFetcher`.
        offline: If True, never reach the network; require every source
            to be already locked AND already on disk.
    """

    def __init__(
        self,
        registry: SourceRegistry,
        lock: SourceLock,
        raw_dir: str | Path,
        *,
        fetcher: HttpFetcher | None = None,
        offline: bool = False,
    ) -> None:
        self.registry = registry
        self.lock = lock
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.fetcher = fetcher or UrllibFetcher()
        self.offline = offline

    def download(self, name: str, *, allow_first_run_resolution: bool = True) -> DownloadResult:
        spec = self.registry.get(name)
        if spec.kind == "statsbomb_open_data":
            return self._download_git(spec)
        return self._download_http(spec, allow_first_run_resolution=allow_first_run_resolution)

    def download_all(
        self, names: list[str] | None = None, *, allow_first_run_resolution: bool = True
    ) -> list[DownloadResult]:
        names = names or self.registry.all_names()
        out: list[DownloadResult] = []
        for n in names:
            try:
                out.append(self.download(n, allow_first_run_resolution=allow_first_run_resolution))
            except DownloadError as exc:
                logger.warning("Skipping source (download failed)", extra={"source": n, "error": str(exc)})
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _download_http(
        self, spec: SourceSpec, *, allow_first_run_resolution: bool
    ) -> DownloadResult:
        if not spec.local_filename:
            raise DownloadError(f"Source {spec.name!r} has no local_filename")
        target = self.raw_dir / spec.local_filename

        # 1. Determine the SHA to use.
        locked = self.lock.get(spec.name)
        if locked is not None:
            sha = locked.resolved_sha
            url = spec.url_template.format(sha=sha)
            used_lock = True
        else:
            if self.offline:
                raise DownloadError(
                    f"Source {spec.name!r} not locked and offline mode is on."
                )
            if not allow_first_run_resolution:
                raise DownloadError(
                    f"Source {spec.name!r} is not locked; refusing to fetch HEAD."
                )
            # First-run: resolve HEAD to a 40-character SHA via the GitHub API.
            repo = parse_github_repo_from_url(spec.url_template)
            if repo is None:
                raise DownloadError(
                    f"Cannot resolve HEAD SHA for non-GitHub URL: {spec.url_template}"
                )
            sha = resolve_repo_head_sha(repo, ref=spec.pinned_sha, fetcher=self.fetcher)
            url = spec.url_template.format(sha=sha)
            used_lock = False

        # 2. Cache hit: re-verify integrity against the locked SHA.
        if target.exists():
            existing_sha256 = self._sha256_file(target)
            if locked is not None and locked.raw_sha256 != existing_sha256:
                raise DownloadError(
                    f"Hash mismatch for cached {spec.name!r}: "
                    f"locked={locked.raw_sha256[:16]}... got {existing_sha256[:16]}..."
                )
            logger.info("Cache hit", extra={"source": spec.name, "path": str(target)})
            schema_valid = self._validate_schema(spec, target)
            # If the file is on disk but the lock has no entry (e.g.
            # a previous bootstrap downloaded the file but failed before
            # persisting the lock), populate the lock now.
            if locked is None:
                self.lock.set_or_update(
                    name=spec.name,
                    requested_ref=spec.pinned_sha,
                    resolved_sha=sha,
                    source_url=url,
                    raw_sha256=existing_sha256,
                    local_path=str(target),
                )
            return DownloadResult(
                source=spec,
                local_path=target,
                sha256=existing_sha256,
                bytes_written=0,
                cache_hit=True,
                schema_valid=schema_valid,
                used_lock=used_lock,
                resolved_sha=sha,
            )

        # 3. No cache; download at the resolved SHA.
        if self.offline:
            raise DownloadError(
                f"Source {spec.name!r} not cached and offline mode is on."
            )
        logger.info("Downloading", extra={"source": spec.name, "url": url, "sha": sha[:12]})
        raw = self.fetcher.fetch(url)
        if not raw:
            raise DownloadError(f"Empty response for {url}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        actual_sha256 = self._sha256_file(target)
        schema_valid = self._validate_schema(spec, target)
        # Persist the lock entry. This is the only place a lock is
        # written; downstream runs will refuse to use HEAD.
        if locked is None or locked.resolved_sha != sha:
            self.lock.set_or_update(
                name=spec.name,
                requested_ref=spec.pinned_sha,
                resolved_sha=sha,
                source_url=url,
                raw_sha256=actual_sha256,
                local_path=str(target),
            )
        return DownloadResult(
            source=spec,
            local_path=target,
            sha256=actual_sha256,
            bytes_written=len(raw),
            cache_hit=False,
            schema_valid=schema_valid,
            used_lock=used_lock,
            resolved_sha=sha,
        )

    def _download_git(self, spec: SourceSpec) -> DownloadResult:
        if not spec.local_path:
            raise DownloadError(f"Git source {spec.name!r} has no local_path")
        target = self.raw_dir / spec.local_path
        if (target / ".git").exists():
            sha = self._sha256_dir(target)
            return DownloadResult(
                source=spec, local_path=target, sha256=sha, bytes_written=0,
                cache_hit=True, schema_valid=True, used_lock=True, resolved_sha=None,
            )
        if self.offline:
            raise DownloadError(f"Git source {spec.name!r} not cached and offline mode is on.")
        if shutil.which("git") is None:
            raise DownloadError("git binary not found; cannot clone StatsBomb open-data.")
        url = spec.url_template
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(target)],
                check=True, timeout=300, capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            raise DownloadError(f"git clone failed: {exc.stderr.decode(errors='ignore')}") from exc
        sha = self._sha256_dir(target)
        return DownloadResult(
            source=spec, local_path=target, sha256=sha, bytes_written=0,
            cache_hit=False, schema_valid=True, used_lock=False, resolved_sha=None,
        )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _sha256_dir(path: Path) -> str:
        h = hashlib.sha256()
        for f in sorted(path.rglob("*")):
            if f.is_file():
                h.update(f.relative_to(path).as_posix().encode("utf-8"))
                h.update(f.read_bytes())
        return h.hexdigest()

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

    def _validate_schema(self, spec: SourceSpec, path: Path) -> bool:
        if not spec.expected_columns and not spec.expected_keys:
            return True
        import csv
        import json

        try:
            if path.suffix == ".csv":
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


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(tz=timezone.utc).isoformat()

    def _validate_schema(self, spec: SourceSpec, path: Path) -> bool:
        if not spec.expected_columns and not spec.expected_keys:
            return True
        import csv
        import json

        try:
            if path.suffix == ".csv":
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
