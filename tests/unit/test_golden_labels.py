"""Golden-label tests for known World Cup knockout matches.

These tests guard against future regressions in the openfootball
parser, the alias registry, and the manifest reconciliation logic.
They assert the actual advancer for four well-known fixtures:

- 2018 World Cup QF: Brazil 1 - 2 Belgium (Belgium advances)
- 2014 World Cup SF: Netherlands 0 - 0 Argentina (Argentina wins on penalties)
- 2022 World Cup QF: Brazil 1 - 1 Croatia (Croatia wins on penalties)
- 2022 World Cup Final: Argentina 3 - 3 France (Argentina wins on penalties)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from football_advance_predictor.data.aliases.alias_registry import (
    AliasRegistry,
    canonical_key,
)
from football_advance_predictor.data.knockout.manifest import (
    KnockoutManifestBuilder,
    has_downstream_bracket,
    is_knockout_stage,
    stage_canonical,
)


GOLDEN_LABELS = [
    {
        "date": "2018-07-06",
        "tournament": "FIFA World Cup 2018",
        "t1": "Brazil",
        "t2": "Belgium",
        "ft": (1, 2),
        "advancer": "Belgium",
        "stage_canonical": "quarter_final",
    },
    {
        "date": "2014-07-09",
        "tournament": "FIFA World Cup 2014",
        "t1": "Netherlands",
        "t2": "Argentina",
        "ft": (0, 0),
        "advancer": "Argentina",
        "stage_canonical": "semi_final",
    },
    {
        "date": "2022-12-09",
        "tournament": "FIFA World Cup 2022",
        "t1": "Croatia",  # openfootball team1
        "t2": "Brazil",  # openfootball team2
        "ft": (0, 0),
        "advancer": "Croatia",
        "stage_canonical": "quarter_final",
    },
    {
        "date": "2022-12-18",
        "tournament": "FIFA World Cup 2022",
        "t1": "Argentina",  # openfootball team1
        "t2": "France",  # openfootball team2
        "ft": (2, 2),  # 90-minute score; Argentina wins on penalties
        "advancer": "Argentina",
        "stage_canonical": "final",
    },
]


def test_openfootball_raw_json_contains_golden_labels(tmp_path: Path) -> None:
    """The raw openfootball JSON files must contain the four matches.

    This guards against upstream schema changes that would silently
    drop fixtures.
    """
    raw_dir = Path("data/raw/sources")
    assert raw_dir.exists(), "raw sources not downloaded; run `data bootstrap` first"
    expected_by_year = {
        2018: {("brazil", "belgium"): ("2018-07-06", "Quarter-final")},
        2014: {("netherlands", "argentina"): ("2014-07-09", "Semi-final")},
        2022: {
            ("brazil", "croatia"): ("2022-12-09", "Quarter-final"),
            ("argentina", "france"): ("2022-12-18", "Final"),
        },
    }
    for year, expected in expected_by_year.items():
        path = raw_dir / f"openfootball_worldcup_{year}.json"
        if not path.exists():
            pytest.skip(f"{path} not available; run data bootstrap")
        with path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        for (t1, t2), (date, round_substr) in expected.items():
            found = False
            for m in doc.get("matches", []):
                if m.get("date") == date and round_substr in m.get("round", ""):
                    teams = {m.get("team1", "").lower(), m.get("team2", "").lower()}
                    if teams == {t1, t2}:
                        found = True
                        break
            assert found, (
                f"Missing golden match {t1} vs {t2} ({date}, ~{round_substr}) in {year}"
            )


def test_golden_labels_in_manifest(tmp_path: Path) -> None:
    """The KnockoutManifest must produce the correct advancer and stage
    for each of the four golden matches."""
    raw_dir = Path("data/raw/sources")
    if not (raw_dir / "openfootball_worldcup_2018.json").exists():
        pytest.skip("bootstrap not run")
    aliases = AliasRegistry.open(Path("data/aliases"))
    builder = KnockoutManifestBuilder(aliases)
    for year in (2014, 2018, 2022):
        provider = _open_provider(raw_dir, year, aliases)
        builder.add_provider(f"openfootball_worldcup_{year}", provider)
    manifest = builder.build()
    # Build a key that is unique: (date, home_team_id, away_team_id).
    by_full_key = {
        (
            r.kickoff_at.date().isoformat(),
            r.home_team_id,
            r.away_team_id,
        ): r
        for r in manifest.rows
    }
    for label in GOLDEN_LABELS:
        t1_id = aliases.resolve(label["t1"], source="golden_label")
        t2_id = aliases.resolve(label["t2"], source="golden_label")
        key = (label["date"], t1_id, t2_id)
        row = by_full_key.get(key) or by_full_key.get(
            (label["date"], t2_id, t1_id)
        )
        assert row is not None, f"Missing manifest row for {label}"
        assert is_knockout_stage(row.stage), f"Stage not knockout: {row.stage}"
        assert has_downstream_bracket(row.stage), f"No downstream bracket: {row.stage}"
        assert stage_canonical(row.stage) == label["stage_canonical"], (
            f"Stage canonical mismatch: {stage_canonical(row.stage)} != {label['stage_canonical']}"
        )
        advancer_id = t1_id if label["advancer"].lower() == t1_id else t2_id
        actual_advancer = row.home_team_id if row.home_wins_tie else row.away_team_id
        assert actual_advancer == advancer_id, (
            f"Wrong advancer for {label['date']} {label['t1']} vs {label['t2']}: "
            f"got {actual_advancer}, expected {advancer_id}"
        )
        assert (row.home_goals_90, row.away_goals_90) == label["ft"], (
            f"Scores mismatch for {label['date']}: got "
            f"({row.home_goals_90}, {row.away_goals_90}) != {label['ft']}"
        )


def test_golden_label_via_csv_report(tmp_path: Path) -> None:
    """The CSV audit file written by the baseline report must include
    the golden matches with the correct advancer (for matches that
    fall into the test fold)."""
    report_path = Path("data/processed/bootstrap/per_match_audit.csv")
    if not report_path.exists():
        pytest.skip("baseline report CSV not generated; run scripts/real_data_baseline_report.py")
    aliases = AliasRegistry.open(Path("data/aliases"))
    with report_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Match by (date, stage, home_team_id) for uniqueness — multiple
    # matches can share date+stage (e.g. 2022-12-09 has 2 QFs).
    by_key = {
        (
            r["kickoff_at"][:10],
            r["stage_canonical"],
            aliases.resolve(r["source_home_team_id"], source="csv_audit"),
        ): r
        for r in rows
    }
    test_only = [
        g for g in GOLDEN_LABELS
        if (g["date"], g["stage_canonical"], aliases.resolve(g["t1"], source="golden_label")) in by_key
    ]
    if not test_only:
        pytest.skip("no golden matches landed in the test fold")
    for label in test_only:
        key = (label["date"], label["stage_canonical"], aliases.resolve(label["t1"], source="golden_label"))
        row = by_key[key]
        adv_id = row.get("actual_advancer_id")
        assert adv_id is not None, f"Missing actual_advancer_id for {key}"
        expected_adv = aliases.resolve(label["advancer"], source="golden_label")
        assert adv_id == expected_adv or label["advancer"].lower() in adv_id.lower(), (
            f"CSV advancer mismatch for {key}: got {adv_id}, expected {label['advancer']}"
        )


def _open_provider(raw_dir: Path, year: int, aliases: AliasRegistry):
    from football_advance_predictor.data.sources.openfootball import (
        OpenFootballTournamentProvider,
    )
    return OpenFootballTournamentProvider(
        path=raw_dir / f"openfootball_worldcup_{year}.json",
        alias_registry=aliases,
        tournament_name=f"FIFA World Cup {year}",
    )
