"""Sensitivity of H1-H4 to the analysis settings. / Sensibilidad a los parámetros.

Writes outputs/sensitivity.csv. Run after src/pull_data.py.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hook_timing as ht  # noqa: E402
import run_analysis as ra  # noqa: E402

ROOT = ht.ROOT

SETTINGS = [
    ("base", {}, False),
    ("min_late_starts=5", {"MIN_LATE_STARTS": 5}, False),
    ("min_late_starts=12", {"MIN_LATE_STARTS": 12}, False),
    ("early=10-30", {"EARLY": (10, 30)}, False),
    ("late=81-105", {"LATE": (81, 105)}, False),
    ("min_tto3_pa=60", {"MIN_TTO3_PA": 60}, False),
    ("pre_gap_window=30d", {"PRE_GAP_DAYS": 30}, False),
    ("exclude annotated injury seasons", {}, True),
]


def run_setting(overrides: dict, exclude_annotated: bool) -> dict:
    saved = {k: getattr(ht, k) for k in overrides}
    for k, v in overrides.items():
        setattr(ht, k, v)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "data").mkdir()
            for y in (ra.PREV, ra.NEXT):
                (tmp / "data" / f"statcast_{y}.csv.gz").symlink_to(
                    (ROOT / "data" / f"statcast_{y}.csv.gz").resolve())
            m = ht.load_manual_exclusions(ROOT)
            if exclude_annotated:
                m["action"] = "exclude"
            m.to_csv(tmp / "data" / "injury_exclusions.csv", index=False, encoding="utf-8")
            return ra.main(tmp, verbose=False)
    finally:
        for k, v in saved.items():
            setattr(ht, k, v)


def main() -> None:
    rows = []
    for label, kw, excl in SETTINGS:
        r = run_setting(kw, excl)
        h3 = r["H3_decision"]["velo_drop_2026"]
        rows.append({
            "setting": label,
            "H1_n": r["H1_velo_drop"]["n"], "H1_r": round(r["H1_velo_drop"]["r_2025_2026"], 2),
            "H1_conf": r["H1_velo_drop"]["confirmed"],
            "H2_n": r["H2_tto3_penalty"]["n"], "H2_r": round(r["H2_tto3_penalty"]["r_2025_2026"], 2),
            "H3_gain_shrunk": round(h3["gain_vs_league"]["shrunk"], 2),
            "H3_conf": h3["confirmed"]["shrunk"],
            "H4_xwoba_mph": round(r["H4_live_velo"]["xwoba_per_mph_lost"], 4),
            "H4_conf": r["H4_live_velo"]["confirmed"],
        })
        print(rows[-1])
    pd.DataFrame(rows).to_csv(ROOT / "outputs" / "sensitivity.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
