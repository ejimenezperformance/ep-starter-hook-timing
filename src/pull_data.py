"""Download regular-season Statcast pitch data for the seasons used in this repo.

Descarga datos de Statcast (solo temporada regular) para las temporadas del repo.

Usage / Uso:
    python src/pull_data.py            # 2025 and 2026
    python src/pull_data.py 2026       # one season

Output: data/statcast_<season>.csv.gz  (not committed; see .gitignore)
"""
from __future__ import annotations

import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Only the columns the analysis needs / Solo las columnas necesarias
COLUMNS = [
    "game_pk", "game_date", "game_type", "game_year", "home_team", "away_team",
    "inning", "inning_topbot", "at_bat_number", "pitch_number",
    "pitcher", "player_name", "p_throws", "batter", "stand",
    "pitch_type", "release_speed", "release_spin_rate", "release_extension",
    "release_pos_x", "release_pos_z", "plate_x", "plate_z", "zone",
    "description", "events", "estimated_woba_using_speedangle",
    "woba_value", "woba_denom", "n_thruorder_pitcher",
    "pitcher_days_since_prev_game", "pitcher_days_until_next_game",
]

SEASON_WINDOWS = {
    2025: (date(2025, 3, 15), date(2025, 10, 1)),
    2026: (date(2026, 3, 15), date(2026, 10, 5)),
}

URL = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details"
       "&player_type=pitcher&hfGT=R%7C&game_date_gt={start}&game_date_lt={end}")


def _fetch(start: date, end: date, tries: int = 4) -> pd.DataFrame:
    url = URL.format(start=start.isoformat(), end=end.isoformat())
    for attempt in range(tries):
        try:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            text = r.content.decode("utf-8-sig")
            if not text.strip():
                return pd.DataFrame(columns=COLUMNS)
            df = pd.read_csv(io.StringIO(text), low_memory=False)
            keep = [c for c in COLUMNS if c in df.columns]
            return df[keep]
        except Exception as exc:  # network hiccups / cortes de red
            wait = 5 * (attempt + 1)
            print(f"  retry {start}..{end} ({exc}); waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed {start}..{end}")


def pull_season(season: int, chunk_days: int = 2, workers: int = 4) -> Path:
    start, end = SEASON_WINDOWS[season]
    chunks = []
    d = start
    while d <= end:
        chunks.append((d, min(d + timedelta(days=chunk_days - 1), end)))
        d += timedelta(days=chunk_days)
    print(f"{season}: {len(chunks)} requests")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(lambda c: _fetch(*c), chunks))
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    df = df[df["game_type"] == "R"]  # regular season only / solo temporada regular
    out = DATA / f"statcast_{season}.csv.gz"
    DATA.mkdir(exist_ok=True)
    df.to_csv(out, index=False, compression="gzip", encoding="utf-8")
    print(f"{season}: {len(df):,} pitches, {df['game_pk'].nunique():,} games -> {out}")
    return out


if __name__ == "__main__":
    seasons = [int(s) for s in sys.argv[1:]] or [2025, 2026]
    for s in seasons:
        pull_season(s)
