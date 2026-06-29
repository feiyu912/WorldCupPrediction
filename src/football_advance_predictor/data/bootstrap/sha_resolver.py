"""Resolve a Git ref to a full 40-character commit SHA via the GitHub API.

Only used during the **first** bootstrap (or after an explicit
``data update-sources``) to upgrade a placeholder ref like
``HEAD`` / ``master`` to a pinned commit SHA.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

from football_advance_predictor.core.logging import get_logger

logger = get_logger(__name__)

_GITHUB_API = "https://api.github.com"
_FULL_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ShaResolutionError(RuntimeError):
    """Raised when a Git ref cannot be resolved to a commit SHA."""


def resolve_repo_head_sha(repo: str, ref: str = "master", *, fetcher=None) -> str:
    """Resolve the head SHA of a GitHub repo's ref.

    Args:
        repo: GitHub repo in ``owner/name`` form (e.g. ``martj42/international-results``).
        ref: The branch or ref to resolve.
        fetcher: Optional HTTP fetcher (defaults to urllib).

    Returns:
        The 40-character commit SHA.

    Behavior:
    - If ``ref`` already looks like a 40-character SHA, return it as-is.
    - If ``ref`` is a placeholder like ``PIN_ME_TO_REAL_COMMIT_SHA`` or
      HEAD, fall back to the repo's default branch (typically
      ``master``) and resolve that.
    """
    if fetcher is None:
        from football_advance_predictor.data.bootstrap.source_downloader import (
            UrllibFetcher,
        )

        fetcher = UrllibFetcher()
    # If ref is already a 40-char SHA, return it.
    if _FULL_SHA_PATTERN.match(ref):
        return ref
    # If ref is a placeholder, fall back to default branch.
    fallback_refs = [ref]
    if ref not in {"master", "main"}:
        fallback_refs.extend(["master", "main"])
    last_exc: ShaResolutionError | None = None
    for candidate in fallback_refs:
        url = f"{_GITHUB_API}/repos/{repo}/branches/{candidate}"
        try:
            body = fetcher.fetch(url)
        except Exception as exc:
            last_exc = ShaResolutionError(
                f"Failed to fetch {url}: {exc}"
            )
            continue
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            last_exc = ShaResolutionError(
                f"Failed to parse SHA response for {repo}@{candidate}"
            )
            continue
        commit = data.get("commit", {}).get("sha")
        if commit and len(commit) == 40:
            logger.info(
                "Resolved SHA", extra={"repo": repo, "ref": candidate, "sha": commit[:12]}
            )
            return commit
    if last_exc is not None:
        raise last_exc
    raise ShaResolutionError(
        f"Could not resolve {repo} (tried {fallback_refs}): unexpected response shape"
    )


def resolve_blob_sha(
    repo: str, ref: str, path: str, *, fetcher=None
) -> str:
    """Resolve the blob SHA of a single file in a GitHub repo at ``ref``."""
    if fetcher is None:
        from football_advance_predictor.data.bootstrap.source_downloader import (
            UrllibFetcher,
        )

        fetcher = UrllibFetcher()
    url = f"{_GITHUB_API}/repos/{repo}/contents/{path}?ref={ref}"
    body = fetcher.fetch(url)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ShaResolutionError(f"Failed to parse blob response for {repo}:{path}@{ref}") from exc
    sha = data.get("sha")
    if not sha:
        raise ShaResolutionError(f"Could not resolve blob {repo}:{path}@{ref}")
    return sha


def parse_github_repo_from_url(url: str) -> str | None:
    """Extract ``owner/name`` from a ``github.com`` URL.

    The URL may include a path segment that is a ``{sha}`` placeholder
    (e.g. ``https://raw.githubusercontent.com/owner/repo/{sha}/file``)
    or it may be a clone URL (``https://github.com/owner/repo.git``).
    In both cases the first two non-empty path segments are owner/name.
    """
    from urllib.parse import unquote

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if not (host == "github.com" or host.endswith(".github.com") or host == "raw.githubusercontent.com"):
        return None
    decoded = unquote(parsed.path)
    parts = [p for p in decoded.split("/") if p]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"
