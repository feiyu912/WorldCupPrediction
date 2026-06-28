"""Generate a slightly larger synthetic dataset for stress-testing the pipeline."""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main(output: Path = Path("data/fixtures/synthetic_matches.csv")) -> None:
    rng = random.Random(42)
    teams = [
        "France", "Germany", "Spain", "Italy", "England",
        "Brazil", "Argentina", "Netherlands", "Portugal", "Belgium",
        "Croatia", "Mexico", "USA", "Japan", "South Korea",
        "Sweden", "Switzerland", "Poland", "Denmark", "Uruguay",
    ]
    start = datetime(2010, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(400):
        home, away = rng.sample(teams, 2)
        hg = rng.randint(0, 4)
        ag = rng.randint(0, 4)
        ko = start + timedelta(days=i * 7)
        if hg == ag:
            adv = rng.choice([home, away])
        else:
            adv = home if hg > ag else away
        rows.append(
            {
                "match_id": f"SYN_{i:04d}",
                "kickoff_at": ko.isoformat().replace("+00:00", "Z"),
                "tournament": rng.choice(["Friendly", "FIFA World Cup", "Euro"]),
                "stage": rng.choice(["Group", "Round of 16", "Quarter-final"]),
                "season": str(ko.year),
                "home_team": home,
                "away_team": away,
                "home_score": hg,
                "away_score": ag,
                "winner": adv if hg != ag else "Draw",
                "advancing_team": adv,
                "neutral": str(rng.choice([True, False])),
                "city": "City",
                "country": "Country",
                "source": "synthetic",
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} synthetic matches to {output}")


if __name__ == "__main__":
    main()
