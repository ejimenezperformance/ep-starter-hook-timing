"""Known-answer tests in both directions / Pruebas con respuesta conocida.

Scenario STABLE  : each starter has his own fatigue slope that repeats 2025->2026,
                   and lost velocity raises xwOBA  -> H1, H3 (velo) and H4 must be CONFIRMED.
Scenario RANDOM  : the slope is redrawn every season and velo has no effect on
                   xwOBA -> H1, H3 and H4 must NOT be confirmed.
Run: python tests/test_synthetic.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_analysis  # noqa: E402

TEAMS = [f"T{i:02d}" for i in range(30)]


def simulate(stable: bool, seed: int, n_pitchers: int = 120) -> dict[int, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    base_velo = rng.normal(94, 2, n_pitchers)
    slope_fixed = rng.normal(-0.9, 0.7, n_pitchers)          # mph lost from pitch 1-25 to 76-100
    xw_per_mph = 0.030 if stable else 0.0                     # xwOBA added per mph lost
    out = {}
    for season in (2025, 2026):
        slope = slope_fixed if stable else rng.normal(-0.9, 0.7, n_pitchers)
        rows = []
        gpk = season * 100000
        for p in range(n_pitchers):
            team = TEAMS[p % 30]
            day0 = pd.Timestamp(f"{season}-03-28")
            for s in range(30):
                gpk += 1
                date = day0 + pd.Timedelta(days=5 * s + rng.integers(0, 2))
                n_p = int(rng.integers(88, 112))
                s_slope = slope[p] + rng.normal(0, 0.8)   # start-to-start fatigue variation
                ab, pc, batter_i = 1, 0, 0
                while pc < n_p:
                    tto = batter_i // 9 + 1
                    pitches_in_pa = int(rng.integers(2, 7))
                    for k in range(1, pitches_in_pa + 1):
                        pc += 1
                        fb = rng.random() < 0.5
                        frac = min(max(pc - 25, 0) / 63.0, 1.3)  # 0 early, ~1 at pitch 88
                        velo = base_velo[p] + s_slope * frac + rng.normal(0, 0.8)
                        last = k == pitches_in_pa
                        lost = max(0.0, -s_slope * frac)
                        xw = np.clip(rng.normal(0.31 + 0.012 * (tto - 1) + xw_per_mph * lost, 0.35), 0, 2) if last else np.nan
                        rows.append((gpk, date.date().isoformat(), "R", season, team, "OPP", 1, "Top", ab, k,
                                     1000 + p, f"Pitcher {p}", "R", 5000 + batter_i % 9, "L" if batter_i % 2 else "R",
                                     "FF" if fb else "SL", velo if fb else velo - 8,
                                     "field_out" if last else np.nan, xw if last else np.nan,
                                     xw if last else np.nan, 1 if last else np.nan, tto))
                    ab += 1
                    batter_i += 1
                # a reliever pitch so starter detection is exercised
                rows.append((gpk, date.date().isoformat(), "R", season, team, "OPP", 9, "Top", ab, 1,
                             9999, "Reliever", "R", 5000, "R", "FF", 97.0, "field_out", 0.3, 0.3, 1, 1))
        cols = ["game_pk", "game_date", "game_type", "game_year", "home_team", "away_team", "inning",
                "inning_topbot", "at_bat_number", "pitch_number", "pitcher", "player_name", "p_throws",
                "batter", "stand", "pitch_type", "release_speed", "events",
                "estimated_woba_using_speedangle", "woba_value", "woba_denom", "n_thruorder_pitcher"]
        out[season] = pd.DataFrame(rows, columns=cols)
    return out


def run(stable: bool, seed: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "data").mkdir()
        for season, df in simulate(stable, seed).items():
            df.to_csv(root / "data" / f"statcast_{season}.csv.gz", index=False, compression="gzip")
        res = run_analysis.main(root, verbose=False)
        st = pd.read_csv(root / "outputs" / "starts_with_exclusions.csv")
        res["_reliever_leak"] = bool((st["pitcher"] == 9999).any())
        return res


def main() -> None:
    ok = True
    s = run(stable=True, seed=1)
    r = run(stable=False, seed=2)
    checks = [
        ("STABLE: H1 velo stability confirmed", s["H1_velo_drop"]["confirmed"]),
        ("STABLE: H3 shrunk beats league for 2026 velo", s["H3_decision"]["velo_drop_2026"]["confirmed"]["shrunk"]),
        ("STABLE: H4 live velo drop -> higher xwOBA", s["H4_live_velo"]["confirmed"]),
        ("STABLE: velo drop recovered ~ -0.9 mph", abs(s["H1_velo_drop"]["mean_2026"] + 0.9) < 0.25),
        ("RANDOM: H1 not confirmed", not r["H1_velo_drop"]["confirmed"]),
        ("RANDOM: H3 shrunk not confirmed", not r["H3_decision"]["velo_drop_2026"]["confirmed"]["shrunk"]),
        ("RANDOM: H4 not confirmed", not r["H4_live_velo"]["confirmed"]),
        ("BOTH: reliever pitches dropped (no starter id 9999)", not s["_reliever_leak"] and not r["_reliever_leak"]),
    ]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
        ok &= bool(passed)
    print(f"\nSTABLE H1 r={s['H1_velo_drop']['r_2025_2026']:.2f} | RANDOM H1 r={r['H1_velo_drop']['r_2025_2026']:.2f}")
    print(f"STABLE H4 xwOBA/mph={s['H4_live_velo']['xwoba_per_mph_lost']:.3f} | "
          f"RANDOM H4 xwOBA/mph={r['H4_live_velo']['xwoba_per_mph_lost']:.3f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
