"""Real-data baseline + audit report.

Run after ``uv run football data bootstrap``. This script:

1. Rebuilds the knockout manifest from raw sources in offline mode.
2. Reports the manifest reconciliation (expected vs found per tournament).
3. Cross-checks OpenFootball penalty outcomes against martj42/shootouts.csv.
4. Classifies alias resolutions into resolved-default / resolved-curated
   / unresolved.
5. Trains Elo-only, constant-prevalence, and unweighted Logistic
   Regression baselines with mirrored training examples kept in the
   same fold. Home advantage is forced to 0 for knockout matches.
6. Verifies complementarity: p(A wins tie) + p(B wins tie) = 1.
7. Emits a per-match prediction audit with log-loss / Brier
   contributions and feature completeness.
8. Marks calibration as insufficient-data when evaluation examples
   are too few.

Output: data/processed/bootstrap/baseline_report.json plus a
human-readable summary printed to stdout.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from football_advance_predictor.backtesting.metrics.evaluation import (  # noqa: E402
    brier_score,
    log_loss,
    roc_auc,
)
from football_advance_predictor.backtesting.symmetry import symmetry_test  # noqa: E402
from football_advance_predictor.data.aliases.alias_registry import (  # noqa: E402
    AliasRegistry,
    _DEFAULT_ALIASES,
)
from football_advance_predictor.data.bootstrap.source_lock import SourceLock  # noqa: E402
from football_advance_predictor.data.knockout.manifest import (  # noqa: E402
    KnockoutManifestBuilder,
    has_downstream_bracket,
    is_knockout_stage,
)
from football_advance_predictor.data.sources.martj42 import MartJ42ResultsProvider  # noqa: E402
from football_advance_predictor.data.sources.openfootball import (  # noqa: E402
    OpenFootballTournamentProvider,
)
from football_advance_predictor.features.elo.elo_engine import (  # noqa: E402
    DynamicEloEngine,
    EloConfig,
)
from football_advance_predictor.models.catboost_gate import (  # noqa: E402
    CatBoostGateConfig,
    evaluate_gate,
)
from football_advance_predictor.models.logistic_baseline import (  # noqa: E402
    LogisticRegressionBaseline,
    LogisticRegressionBaselineConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HOME_ADVANTAGE_KNOCKOUT = 0.0  # World Cup knockout is neutral
EXPECTED_PER_EDITION = 15  # 8 R16 + 4 QF + 2 SF + 1 F
MIN_EVAL_EXAMPLES_FOR_CALIBRATION = 30


@dataclass
class MirroredExample:
    """A training example with optional mirrored partner."""

    match_id: str
    kickoff: datetime
    home_team_id: str
    away_team_id: str
    home_wins_tie: int
    mirrored: bool
    fold: str


def _resolve_knockout_via_openfootball(raw_dir: Path, aliases: AliasRegistry) -> dict[str, list[dict]]:
    """Return per-tournament list of matches with explicit knockout round
    metadata from the openfootball per-year JSONs.

    The openfootball per-year file is ``worldcup.json/<year>/worldcup.json``.
    We pin to the registry entries (openfootball_worldcup_2014, ...).
    """
    out: dict[str, list[dict]] = {}
    for source_name, default_name, year in (
        ("openfootball_worldcup_1990", "FIFA World Cup 1990", 1990),
        ("openfootball_worldcup_1994", "FIFA World Cup 1994", 1994),
        ("openfootball_worldcup_1998", "FIFA World Cup 1998", 1998),
        ("openfootball_worldcup_2002", "FIFA World Cup 2002", 2002),
        ("openfootball_worldcup_2006", "FIFA World Cup 2006", 2006),
        ("openfootball_worldcup_2010", "FIFA World Cup 2010", 2010),
        ("openfootball_worldcup_2014", "FIFA World Cup 2014", 2014),
        ("openfootball_worldcup_2018", "FIFA World Cup 2018", 2018),
        ("openfootball_worldcup_2022", "FIFA World Cup 2022", 2022),
    ):
        path = raw_dir / f"{source_name}.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        matches: list[dict] = []
        for m in doc.get("matches", []) or []:
            if not isinstance(m, dict):
                continue
            round_name = (m.get("round") or "").strip()
            if not is_knockout_stage(round_name):
                continue
            team1 = (m.get("team1") or "").strip()
            team2 = (m.get("team2") or "").strip()
            if not team1 or not team2:
                continue
            score = m.get("score") or {}
            ft = score.get("ft") or []
            if not isinstance(ft, (list, tuple)) or len(ft) < 2:
                continue
            try:
                hg = int(ft[0])
                ag = int(ft[1])
            except Exception:
                continue
            date = (m.get("date") or "").strip()
            if not date:
                continue
            pen = score.get("p") or score.get("pen") or []
            try:
                ph = int(pen[0]) if len(pen) >= 2 else None
                pa = int(pen[1]) if len(pen) >= 2 else None
            except Exception:
                ph = pa = None
            matches.append(
                {
                    "year": year,
                    "tournament_name": default_name,
                    "date": date,
                    "round": round_name,
                    "team1_raw": team1,
                    "team2_raw": team2,
                    "team1_id": aliases.resolve(team1, source=source_name),
                    "team2_id": aliases.resolve(team2, source=source_name),
                    "score_ft": (hg, ag),
                    "score_pen": (ph, pa),
                    "winner_field": m.get("winner"),
                }
            )
        out[default_name] = matches
    return out


def _cross_check_penalties(openfootball_matches: list[dict], shootouts_csv: Path) -> dict[str, Any]:
    """Verify OpenFootball penalty outcomes against martj42/shootouts.csv.

    For every openfootball knockout draw that has ``score.p``, look up
    the matching shootout row by (date, team1, team2) and report
    agreement.
    """
    if not shootouts_csv.exists():
        return {"enabled": False, "reason": f"shootouts.csv missing: {shootouts_csv}"}
    matches: dict[tuple[str, str, str], dict[str, str]] = {}
    with shootouts_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("date", "").strip(), row.get("home_team", "").strip().lower(), row.get("away_team", "").strip().lower())
            matches[key] = row
    agreement = {"enabled": True, "n_compared": 0, "n_agree": 0, "n_disagree": 0, "details": []}
    for m in openfootball_matches:
        if m["score_ft"][0] != m["score_ft"][1]:
            continue  # only check drawn matches that went to penalties
        if m["score_pen"][0] is None or m["score_pen"][1] is None:
            continue
        key = (m["date"], m["team1_raw"].lower(), m["team2_raw"].lower())
        s = matches.get(key)
        if not s:
            agreement["details"].append(
                {
                    "match": f'{m["date"]} {m["team1_raw"]} vs {m["team2_raw"]}',
                    "outcome": "no_martj42_shootout_row",
                }
            )
            continue
        agreement["n_compared"] += 1
        # Determine the shootout winner per martj42.
        winner_field = s.get("winner", "").strip().lower()
        of_winner_idx = 0 if m["score_pen"][0] > m["score_pen"][1] else 1
        of_winner_name = (
            m["team1_raw"] if of_winner_idx == 0 else m["team2_raw"]
        )
        agree = (
            (winner_field == m["team1_raw"].lower() and of_winner_idx == 0)
            or (winner_field == m["team2_raw"].lower() and of_winner_idx == 1)
        )
        if agree:
            agreement["n_agree"] += 1
        else:
            agreement["n_disagree"] += 1
        agreement["details"].append(
            {
                "match": f'{m["date"]} {m["team1_raw"]} vs {m["team2_raw"]}',
                "openfootball_pen_score": f'{m["score_pen"][0]}-{m["score_pen"][1]}',
                "openfootball_winner": of_winner_name,
                "martj42_shootout_winner": winner_field,
                "agree": agree,
            }
        )
    return agreement


def _classify_aliases(aliases: AliasRegistry, observed_names: list[str]) -> dict[str, Any]:
    """Split aliases into resolved-default / resolved-curated / unresolved.

    - resolved-default: matched against the built-in default table.
    - resolved-curated: matched against the versioned registry (entries
      added via the seed-and-extend flow beyond the defaults).
    - unresolved: not matched anywhere; recorded in unresolved.jsonl.
    """
    default_keys = {_DEFAULT_ALIASES[k].lower() if False else k.lower() for k in _DEFAULT_ALIASES}
    # _DEFAULT_ALIASES keys are already in display form; canonicalise
    # by lowercasing for comparison.
    default_canonical = {k.lower() for k in _DEFAULT_ALIASES}

    # canonical_key used by the registry is in alias_registry.canonical_key
    from football_advance_predictor.data.aliases.alias_registry import canonical_key

    default_canonical_keys = {canonical_key(k) for k in _DEFAULT_ALIASES}

    resolved_default = 0
    resolved_curated = 0
    unresolved = 0
    ambiguous = 0
    sample_unresolved: list[str] = []
    for name in observed_names:
        if not name:
            continue
        key = canonical_key(name)
        if key in aliases._entries:
            entry = aliases._entries[key]
            if entry.source == "builtin_default":
                resolved_default += 1
            else:
                resolved_curated += 1
        elif key in default_canonical_keys:
            resolved_default += 1
        else:
            unresolved += 1
            if len(sample_unresolved) < 25:
                sample_unresolved.append(name)
    return {
        "resolved_default": resolved_default,
        "resolved_curated": resolved_curated,
        "unresolved": unresolved,
        "ambiguous": ambiguous,
        "sample_unresolved": sample_unresolved,
    }


def _build_mirrored_examples(
    rows: list,
    *,
    fold_fn,
) -> tuple[list[MirroredExample], list[MirroredExample], list[MirroredExample]]:
    """Split rows into train/val/test folds and add mirrored examples.

    Mirrored examples are added IN THE SAME FOLD as their originals to
    prevent leakage. The mirrored example swaps (home, away) and
    inverts ``home_wins_tie``.
    """
    train: list[MirroredExample] = []
    val: list[MirroredExample] = []
    test: list[MirroredExample] = []
    for r in sorted(rows, key=lambda r: r.kickoff_at):
        fold = fold_fn(r)
        original = MirroredExample(
            match_id=r.match_id,
            kickoff=r.kickoff_at,
            home_team_id=r.home_team_id,
            away_team_id=r.away_team_id,
            home_wins_tie=int(r.home_wins_tie),
            mirrored=False,
            fold=fold,
        )
        mirror = MirroredExample(
            match_id=f"{r.match_id}__mirror",
            kickoff=r.kickoff_at,
            home_team_id=r.away_team_id,
            away_team_id=r.home_team_id,
            home_wins_tie=int(not r.home_wins_tie),
            mirrored=True,
            fold=fold,
        )
        bucket = {"train": train, "val": val, "test": test}[fold]
        bucket.append(original)
        bucket.append(mirror)
    return train, val, test


def _feature_matrix(examples: list[MirroredExample]) -> tuple[Any, np.ndarray, list[str]]:
    """Build feature matrix using signed difference features.

    The baseline has only one feature (Elo probability at kickoff).
    Signed differences are computed as (home - away) so that swapping
    the teams flips the sign.
    """
    import pandas as pd

    if not examples:
        return pd.DataFrame(columns=["elo_advantage"]), np.array([], dtype=int), []
    rows = []
    y = []
    for ex in examples:
        rows.append(
            {
                "match_id": ex.match_id,
                "mirrored": ex.mirrored,
                "elo_advantage": 0.0,
            }
        )
        y.append(ex.home_wins_tie)
    df = pd.DataFrame(rows).set_index("match_id")
    return df, np.array(y, dtype=int), ["elo_advantage"]


def _elo_probability_for(
    elo_engine: DynamicEloEngine, ex: MirroredExample
) -> float:
    """Compute Elo-based P(home wins tie) with home_advantage = 0
    (knockout matches are treated as neutral).
    """
    return elo_engine.predict_home_advance_probability(
        ex.home_team_id, ex.away_team_id, ex.kickoff, neutral_venue=True
    )


def _elo_probability_mirror(
    elo_engine: DynamicEloEngine, ex: MirroredExample
) -> float:
    """P(away team wins) = 1 - P(home wins). Used in the symmetry check
    on the Elo engine.

    With home_advantage = 0 (neutral), the Elo probability of B beating
    A at the same cutoff is exactly 1 - P(A beats B). This must hold
    within numerical tolerance.
    """
    p = _elo_probability_for(elo_engine, ex)
    return 1.0 - p


def main() -> int:
    raw_dir = ROOT / "data" / "raw" / "sources"
    aliases_dir = ROOT / "data" / "aliases"
    artifacts_dir = ROOT / "data" / "processed" / "bootstrap"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    aliases = AliasRegistry.open(aliases_dir)

    # ---------- 1. Source lock report --------------------------------
    lock_path = raw_dir / "lock.json"
    lock = SourceLock.load(lock_path)
    source_lock_report = {
        "lock_present": lock_path.exists(),
        "schema_version": lock.schema_version,
        "created_at": lock.created_at,
        "updated_at": lock.updated_at,
        "n_locked_sources": len(lock.sources),
        "all_full_sha": all(lock.validate_full_sha(n) for n in lock.names()),
        "sources": {
            n: {
                "requested_ref": s.requested_ref,
                "resolved_sha": s.resolved_sha,
                "raw_sha256": s.raw_sha256,
                "source_url": s.source_url,
                "retrieved_at": s.retrieved_at,
            }
            for n, s in lock.sources.items()
        },
    }

    # ---------- 2. Manifest reconciliation ---------------------------
    of_matches = _resolve_knockout_via_openfootball(raw_dir, aliases)
    expected_vs_found: dict[str, dict[str, Any]] = {}
    reconciliation_rows: list[dict[str, Any]] = []
    third_place_excluded: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    knockout_rows: list[Any] = []

    builder = KnockoutManifestBuilder(aliases)
    for default_name, matches in sorted(of_matches.items()):
        # The provider-level manifest still uses the existing builder
        # for cross-checking; we ALSO compute the expected-vs-found
        # diff manually to be transparent.
        found = sum(
            1
            for m in matches
            if has_downstream_bracket(m["round"])
        )
        third = sum(
            1 for m in matches if is_knockout_stage(m["round"]) and not has_downstream_bracket(m["round"])
        )
        expected_vs_found[default_name] = {
            "expected": EXPECTED_PER_EDITION,
            "found": found,
            "passes": found == EXPECTED_PER_EDITION,
            "delta": found - EXPECTED_PER_EDITION,
        }
        for m in matches:
            row = {
                "tournament_name": default_name,
                "year": m["year"],
                "date": m["date"],
                "round": m["round"],
                "team1_raw": m["team1_raw"],
                "team2_raw": m["team2_raw"],
                "team1_id": m["team1_id"],
                "team2_id": m["team2_id"],
                "score_ft": m["score_ft"],
                "score_pen": m["score_pen"],
                "winner_field": m["winner_field"],
                "home_wins_tie": None,
                "exclusion": None,
            }
            if not is_knockout_stage(m["round"]):
                row["exclusion"] = "not_knockout_stage"
                quarantined.append(row)
                continue
            if not has_downstream_bracket(m["round"]):
                row["exclusion"] = "third_place_no_downstream_bracket"
                third_place_excluded.append(row)
                continue
            if m["score_pen"][0] is not None and m["score_pen"][1] is not None:
                if m["score_pen"][0] != m["score_pen"][1]:
                    row["home_wins_tie"] = bool(m["score_pen"][0] > m["score_pen"][1])
            elif m["winner_field"] in (1, "1"):
                row["home_wins_tie"] = True
            elif m["winner_field"] in (2, "2"):
                row["home_wins_tie"] = False
            elif m["score_ft"][0] != m["score_ft"][1]:
                row["home_wins_tie"] = bool(m["score_ft"][0] > m["score_ft"][1])
            else:
                row["exclusion"] = "no_advancer_on_draw"
                quarantined.append(row)
                continue
            reconciliation_rows.append(row)
            knockout_rows.append(
                type(
                    "R",
                    (),
                    {
                        "match_id": f"{default_name.lower().replace(' ', '_')}_{m['date'].replace('-', '')}_{m['team1_id']}_{m['team2_id']}",
                        "kickoff_at": datetime.fromisoformat(m["date"] + "T00:00:00+00:00"),
                        "competition_id": "world_cup",
                        "competition_name": default_name,
                        "stage": m["round"],
                        "season_or_year": str(m["year"]),
                        "home_team_id": m["team1_id"],
                        "away_team_id": m["team2_id"],
                        "home_goals_90": m["score_ft"][0],
                        "away_goals_90": m["score_ft"][1],
                        "home_wins_tie": row["home_wins_tie"],
                        "source": f"openfootball_worldcup_{m['year']}",
                    },
                )()
            )

    manifest_reconciliation = {
        "expected_per_edition": EXPECTED_PER_EDITION,
        "expected_vs_found": expected_vs_found,
        "reconciliation_rows": reconciliation_rows,
        "third_place_excluded": third_place_excluded,
        "quarantined": quarantined,
        "all_editions_pass": all(
            v["passes"] for v in expected_vs_found.values()
        ),
    }

    # ---------- 3. Alias classification -----------------------------
    observed: list[str] = []
    for r in reconciliation_rows:
        observed.extend([r["team1_raw"], r["team2_raw"]])
    alias_classification = _classify_aliases(aliases, observed)

    # ---------- 4. Penalty cross-check ------------------------------
    of_all = [m for matches in of_matches.values() for m in matches]
    penalty_cross_check = _cross_check_penalties(
        of_all, raw_dir / "martj42_shootouts.csv"
    )

    # ---------- 5. Baselines: Elo + constant + logistic --------------
    elo_cfg = EloConfig(
        base_k_factor=20.0,
        home_advantage=0.0,  # knockout treated as neutral
        tie_resolution="draw_treated_as_50_50",
    )
    elo_engine = DynamicEloEngine(elo_cfg)
    for r in knockout_rows:
        elo_engine._update(
            {
                "kickoff_at": r.kickoff_at,
                "home_team_id": r.home_team_id,
                "away_team_id": r.away_team_id,
                "home_goals": r.home_goals_90,
                "away_goals": r.away_goals_90,
                "neutral_venue": True,
                "home_advances": r.home_wins_tie,
                "competition_importance": 2.0,
            }
        )

    n = len(knockout_rows)
    train_end = max(1, int(n * 0.5))
    val_end = max(train_end + 1, int(n * 0.75))

    def _fold_fn(idx: int, row: Any) -> str:
        if idx < train_end:
            return "train"
        if idx < val_end:
            return "val"
        return "test"

    sorted_rows = sorted(
        [
            (i, r)
            for i, r in enumerate(knockout_rows)
        ],
        key=lambda x: x[1].kickoff_at,
    )
    examples: list[MirroredExample] = []
    for idx, r in sorted_rows:
        fold = _fold_fn(idx, r)
        examples.append(
            MirroredExample(
                match_id=r.match_id,
                kickoff=r.kickoff_at,
                home_team_id=r.home_team_id,
                away_team_id=r.away_team_id,
                home_wins_tie=int(r.home_wins_tie),
                mirrored=False,
                fold=fold,
            )
        )
        examples.append(
            MirroredExample(
                match_id=f"{r.match_id}__mirror",
                kickoff=r.kickoff_at,
                home_team_id=r.away_team_id,
                away_team_id=r.home_team_id,
                home_wins_tie=int(not r.home_wins_tie),
                mirrored=True,
                fold=fold,
            )
        )
    train_ex = [e for e in examples if e.fold == "train"]
    val_ex = [e for e in examples if e.fold == "val"]
    test_ex = [e for e in examples if e.fold == "test"]

    def _elo_p_from_dict(d: dict[str, Any]) -> float:
        return elo_engine.predict_home_advance_probability(
            d["home_team_id"], d["away_team_id"], d["cutoff_time"], neutral_venue=True
        )

    def _elo_p_mirror_from_dict(d: dict[str, Any]) -> float:
        """P(away team wins the tie) when the dict is the SWAPPED match.

        The symmetry test swaps the IDs in the dict before calling this.
        ``d["home_team_id"]`` here is the ORIGINAL away team. We want
        P(original away wins), which is exactly
        ``predict_home_advance_probability(away, home)`` because
        ``predict_home_advance_probability(away, home)`` is "probability
        the team listed first advances when they are treated as home" =
        "probability the original away team advances".

        With home_advantage=0 this is exactly 1 - p_home(orig home,
        orig away) and the symmetry sum is exactly 1.
        """
        return elo_engine.predict_home_advance_probability(
            d["home_team_id"], d["away_team_id"], d["cutoff_time"], neutral_venue=True
        )

    def _elo_p_from_example(ex: MirroredExample) -> float:
        return elo_engine.predict_home_advance_probability(
            ex.home_team_id, ex.away_team_id, ex.kickoff, neutral_venue=True
        )

    # Elo probabilities on training examples (with mirrors).
    elo_train = np.array([_elo_p_from_example(e) for e in train_ex])
    elo_val = np.array([_elo_p_from_example(e) for e in val_ex])
    elo_test = np.array([_elo_p_from_example(e) for e in test_ex])
    y_train = np.array([e.home_wins_tie for e in train_ex], dtype=int)
    y_val = np.array([e.home_wins_tie for e in val_ex], dtype=int)
    y_test = np.array([e.home_wins_tie for e in test_ex], dtype=int)

    # Symmetry check (Elo with home_advantage=0 must be exactly symmetric).
    sym_input = [
        {
            "home_team_id": e.home_team_id,
            "away_team_id": e.away_team_id,
            "cutoff_time": e.kickoff,
        }
        for e in test_ex
        if not e.mirrored
    ]
    symmetry = symmetry_test(
        _elo_p_from_dict, sym_input, mirror_predict_fn=_elo_p_mirror_from_dict, tolerance=0.005
    )
    symmetry_payload = symmetry.to_dict()

    def _metrics(probs, y) -> dict[str, Any]:
        if len(probs) == 0:
            return {"n": 0}
        return {
            "n": int(len(probs)),
            "log_loss": float(log_loss(y, np.clip(probs, 1e-6, 1 - 1e-6))),
            "brier": float(brier_score(probs, y)),
            "roc_auc": float(roc_auc(probs, y)) if not np.isnan(roc_auc(probs, y)) else None,
            "prevalence": float(np.mean(y)),
            "calibration_insufficient_data": len(probs) < MIN_EVAL_EXAMPLES_FOR_CALIBRATION,
        }

    elo_summary = {
        "train": _metrics(elo_train, y_train),
        "val": _metrics(elo_val, y_val),
        "test": _metrics(elo_test, y_test),
        "symmetry_test": symmetry_payload,
    }

    # Constant-prevalence baseline (predict mean(y_train) for every test).
    if len(y_train) > 0:
        const_p = float(np.mean(y_train))
    else:
        const_p = 0.5
    const_test_pred = np.full(len(y_test), const_p, dtype=float)
    const_summary = {
        "train_prevalence": const_p,
        "test": _metrics(const_test_pred, y_test),
    }

    # Logistic regression on signed difference feature.
    def _build_X(examples: list[MirroredExample]) -> tuple[Any, np.ndarray]:
        import pandas as pd

        X = pd.DataFrame(
            {
                "match_id": [e.match_id for e in examples],
                "mirrored": [e.mirrored for e in examples],
                "elo_signed": [
                    _elo_p_from_example(e) - (1.0 - _elo_p_from_example(e)) for e in examples
                ],
                "home_wins_tie": [e.home_wins_tie for e in examples],
            }
        ).set_index("match_id")
        return X, X["home_wins_tie"].to_numpy(dtype=int)

    X_train, y_train_log = _build_X(train_ex)
    X_val, y_val_log = _build_X(val_ex)
    X_test, y_test_log = _build_X(test_ex)
    X_train_features = X_train[["elo_signed"]].copy()
    X_val_features = X_val[["elo_signed"]].copy()
    X_test_features = X_test[["elo_signed"]].copy()
    baseline_cfg = LogisticRegressionBaselineConfig(class_weight=None)
    model = LogisticRegressionBaseline(baseline_cfg).fit(
        X_train_features, y_train_log, X_val_features, y_val_log
    )
    log_train = model.predict_proba(X_train_features)
    log_val = model.predict_proba(X_val_features) if len(X_val_features) else np.array([])
    log_test = model.predict_proba(X_test_features) if len(X_test_features) else np.array([])

    def _log_metrics(probs, y) -> dict[str, Any]:
        if len(probs) == 0:
            return {"n": 0}
        return {
            "n": int(len(probs)),
            "log_loss": float(log_loss(y, np.clip(probs, 1e-6, 1 - 1e-6))),
            "brier": float(brier_score(probs, y)),
            "roc_auc": float(roc_auc(probs, y)) if not np.isnan(roc_auc(probs, y)) else None,
            "prevalence": float(np.mean(y)),
            "calibration_insufficient_data": len(probs) < MIN_EVAL_EXAMPLES_FOR_CALIBRATION,
        }

    logistic_summary = {
        "config": {
            "class_weight": baseline_cfg.class_weight,
            "C": baseline_cfg.C,
            "min_samples_required": baseline_cfg.min_samples_required,
        },
        "feature_columns": model.feature_columns,
        "feature_importance_top10": model.feature_importance()[:10],
        "train": _log_metrics(log_train, y_train_log),
        "val": _log_metrics(log_val, y_val_log),
        "test": _log_metrics(log_test, y_test_log),
        "training_prevalence": float(np.mean(y_train_log)) if len(y_train_log) else None,
    }

    # Symmetry check on logistic too.
    log_sym_input = [
        {"home_team_id": e.home_team_id, "away_team_id": e.away_team_id, "cutoff_time": e.kickoff}
        for e in test_ex
        if not e.mirrored
    ]

    def _log_predict_one(d: dict[str, Any]) -> float:
        import pandas as pd

        feat = pd.DataFrame(
            {
                "elo_signed": [
                    _elo_p_from_dict(d) - (1.0 - _elo_p_from_dict(d))
                ],
            }
        )
        return float(model.predict_proba(feat)[0])

    log_sym = symmetry_test(_log_predict_one, log_sym_input, tolerance=0.05)
    log_sym_payload = log_sym.to_dict()

    # ---------- 6. CatBoost gate decision (still disabled) ---------
    gate = evaluate_gate([], CatBoostGateConfig(enabled_in_models_yaml=False))
    gate_payload = gate.to_dict()

    # ---------- 7. Per-match prediction audit (test fold only, originals)
    audit_rows: list[dict[str, Any]] = []
    for ex, prob in zip(test_ex, log_test):
        if ex.mirrored:
            continue
        # Find the corresponding reconciliation row.
        match_row = next(
            (r for r in reconciliation_rows
             if r["team1_id"] == ex.home_team_id
             and r["team2_id"] == ex.away_team_id
             and r["date"] in ex.match_id),
            None,
        )
        actual = ex.home_wins_tie
        log_loss_contrib = (
            -(actual * np.log(max(prob, 1e-12))
              + (1 - actual) * np.log(max(1 - prob, 1e-12)))
        )
        brier_contrib = (prob - actual) ** 2
        elo_p_test = _elo_p_from_example(ex)
        audit_rows.append(
            {
                "match_id": ex.match_id,
                "kickoff": ex.kickoff.isoformat(),
                "actual_winner": "home" if actual else "away",
                "elo_probability": float(elo_p_test),
                "logistic_probability": float(prob),
                "predicted_class": "home" if prob >= 0.5 else "away",
                "log_loss_contribution": float(log_loss_contrib),
                "brier_contribution": float(brier_contrib),
                "feature_completeness": {
                    "elo_probability_present": True,
                    "logistic_probability_present": True,
                    "statsbomb_available": False,
                    "statsbomb_missingness_pct": 100.0,
                },
            }
        )
    audit_rows.sort(key=lambda r: -r["log_loss_contribution"])

    # ---------- 8. Compile output -----------------------------------
    output = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "source_lock_report": source_lock_report,
        "alias_classification": alias_classification,
        "penalty_cross_check": penalty_cross_check,
        "manifest_reconciliation": manifest_reconciliation,
        "elo_baseline": elo_summary,
        "constant_prevalence_baseline": const_summary,
        "logistic_baseline": {
            **logistic_summary,
            "symmetry_test": log_sym_payload,
        },
        "catboost_gate": gate_payload,
        "per_match_prediction_audit": audit_rows[:20],
    }
    output_path = artifacts_dir / "baseline_report.json"
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {output_path}")

    # ---------- 9. Print human-readable summary ---------------------
    print("\n=== Reconciliation ===")
    for tn, v in expected_vs_found.items():
        flag = "PASS" if v["passes"] else "FAIL"
        print(f"  {tn}: expected={v['expected']}, found={v['found']}, delta={v['delta']} [{flag}]")
    print(f"  All editions pass: {manifest_reconciliation['all_editions_pass']}")
    print(f"  Third-place excluded: {len(third_place_excluded)}")
    print(f"  Quarantined: {len(quarantined)}")
    print("\n=== Alias classification ===")
    for k, v in alias_classification.items():
        if k == "sample_unresolved":
            continue
        print(f"  {k}: {v}")
    print("\n=== Penalty cross-check ===")
    if penalty_cross_check.get("enabled"):
        print(f"  Compared: {penalty_cross_check['n_compared']}")
        print(f"  Agree:    {penalty_cross_check['n_agree']}")
        print(f"  Disagree: {penalty_cross_check['n_disagree']}")
    else:
        print(f"  Skipped: {penalty_cross_check.get('reason')}")
    print("\n=== Baselines (test fold) ===")
    print(f"  Elo n={elo_summary['test']['n']}, log_loss={elo_summary['test'].get('log_loss', float('nan')):.3f}, brier={elo_summary['test'].get('brier', float('nan')):.3f}, prevalence={elo_summary['test'].get('prevalence', float('nan')):.3f}")
    print(f"  Const-prevalence (predicts {const_p:.3f}): log_loss={const_summary['test'].get('log_loss', float('nan')):.3f}, brier={const_summary['test'].get('brier', float('nan')):.3f}")
    print(f"  Logistic n={logistic_summary['test']['n']}, log_loss={logistic_summary['test'].get('log_loss', float('nan')):.3f}, brier={logistic_summary['test'].get('brier', float('nan')):.3f}, roc_auc={logistic_summary['test'].get('roc_auc')}, calibration_insufficient_data={logistic_summary['test']['calibration_insufficient_data']}")
    print(f"  Symmetry (Elo)   : {symmetry_payload}")
    print(f"  Symmetry (Logistic): {log_sym_payload}")
    print(f"\n=== CatBoost gate ===")
    print(f"  becomes_default={gate_payload['catboost_becomes_default']}, reason={gate_payload['reason']}")
    print(f"\n=== Worst 5 matches by log-loss contribution ===")
    for r in audit_rows[:5]:
        print(f"  {r['match_id']:60s} p={r['logistic_probability']:.3f} actual={r['actual_winner']} log_loss={r['log_loss_contribution']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())