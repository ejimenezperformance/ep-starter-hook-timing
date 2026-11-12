"""Core pipeline for ep-starter-hook-timing.

Decision question / Pregunta de decisión:
    Should a club use each starter's own 2025 fatigue profile to decide when to
    pull him in 2026, or the league-average rule?
    ¿Debe un club usar el perfil de fatiga 2025 de cada abridor para decidir
    cuándo sacarlo en 2026, o la regla promedio de la liga?

All functions take a Statcast-format pitch DataFrame (see src/pull_data.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# ---- Pre-registered settings / Parámetros definidos de antemano ----------
EARLY = (1, 25)          # baseline window (pitch count) / ventana base
LATE = (76, 100)         # late window / ventana tardía
MIN_FB_PER_WINDOW = 5    # fastballs needed in each window
MIN_LATE_STARTS = 8      # starts reaching LATE needed for a pitcher-season
MIN_TTO3_PA = 40         # PAs third time through needed for a pitcher-season
MIN_TTO1_PA = 80
GAP_DAYS = 14            # absence gap that flags a possible injury
PRE_GAP_DAYS = 0         # also drop starts this many days before a gap (sensitivity: 30)
OPENER_MEDIAN_PITCHES = 60
CONFIRM_R = 0.30         # stability threshold (practical)
CONFIRM_MAE_GAIN = 0.05  # decision threshold: >=5% lower error than league rule
PITCH_BINS = [0, 25, 50, 75, 100, 200]
PITCH_BIN_LABELS = ["1-25", "26-50", "51-75", "76-100", "101+"]


# ---- Starts --------------------------------------------------------------
def starter_pitches(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the pitches thrown by each team's starting pitcher, with the
    pitcher's running pitch count in the game."""
    d = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).copy()
    # Top of inning -> home team is pitching
    d["fld_team"] = np.where(d["inning_topbot"] == "Top", d["home_team"], d["away_team"])
    first = d.groupby(["game_pk", "fld_team"], sort=False)["pitcher"].first()
    first = first.rename("starter").reset_index()
    d = d.merge(first, on=["game_pk", "fld_team"], how="left")
    d = d[d["pitcher"] == d["starter"]].drop(columns="starter")
    d["pitch_count"] = d.groupby(["game_pk", "pitcher"]).cumcount() + 1
    d["game_date"] = pd.to_datetime(d["game_date"])
    return d


def start_table(sp: pd.DataFrame) -> pd.DataFrame:
    s = (sp.groupby(["game_pk", "pitcher"])
           .agg(game_date=("game_date", "first"), season=("game_year", "first"),
                player_name=("player_name", "first"), pitches=("pitch_count", "max"))
           .reset_index().sort_values(["pitcher", "game_date"]))
    return s


def flag_absences(starts: pd.DataFrame, season_end: dict[int, pd.Timestamp] | None = None,
                  manual: pd.DataFrame | None = None) -> pd.DataFrame:
    """Automatic + manual injury/absence filter (applied before any analysis).

    exclude when:
      - next start is GAP_DAYS+ later (start may have ended with an injury)
      - previous start was GAP_DAYS+ earlier (return / ramp-up start)
      - last start of the season is GAP_DAYS+ before season end
      - pitcher is an opener/bulk type (median pitches < OPENER_MEDIAN_PITCHES)
      - listed in data/injury_exclusions.csv with action == exclude
    """
    s = starts.copy()
    g = s.groupby(["pitcher", "season"])["game_date"]
    s["days_to_next"] = (g.shift(-1) - s["game_date"]).dt.days
    s["days_from_prev"] = (s["game_date"] - g.shift(1)).dt.days
    s["exclude_reason"] = ""
    s.loc[s["days_to_next"] >= GAP_DAYS, "exclude_reason"] = "gap_after"
    if PRE_GAP_DAYS > 0:
        gap_dates = s.loc[s["days_to_next"] >= GAP_DAYS, ["pitcher", "season", "game_date"]]
        for (p, y), g in gap_dates.groupby(["pitcher", "season"]):
            for gd in g["game_date"]:
                m = ((s["pitcher"] == p) & (s["season"] == y) & (s["game_date"] < gd)
                     & ((gd - s["game_date"]).dt.days <= PRE_GAP_DAYS))
                s.loc[m & (s["exclude_reason"] == ""), "exclude_reason"] = "pre_gap_window"
    s.loc[s["days_from_prev"] >= GAP_DAYS, "exclude_reason"] = "return_after_gap"
    if season_end is None:
        season_end = s.groupby("season")["game_date"].max().to_dict()
    last = s["days_to_next"].isna()
    end = s["season"].map(season_end)
    s.loc[last & ((end - s["game_date"]).dt.days >= GAP_DAYS), "exclude_reason"] = "season_ended_early"
    med = s.groupby(["pitcher", "season"])["pitches"].transform("median")
    s.loc[med < OPENER_MEDIAN_PITCHES, "exclude_reason"] = "opener_bulk"
    if manual is not None and len(manual):
        ex = manual[manual["action"] == "exclude"]
        key = set(zip(ex["pitcher_name"], ex["season"].astype(int)))
        hit = [(n, y) in key for n, y in zip(s["player_name"], s["season"])]
        s.loc[hit, "exclude_reason"] = "manual_injury"
    s["excluded"] = s["exclude_reason"] != ""
    return s


def load_manual_exclusions(root: Path = ROOT) -> pd.DataFrame:
    p = root / "data" / "injury_exclusions.csv"
    if p.exists():
        return pd.read_csv(p, encoding="utf-8")
    return pd.DataFrame(columns=["pitcher_name", "season", "action", "injury_type", "reason"])


# ---- Fatigue: velocity ---------------------------------------------------
def primary_fastball(sp: pd.DataFrame) -> pd.Series:
    """Primary fastball type per pitcher-season (FF or SI; FC if neither >=15%)."""
    out = {}
    for (p, y), g in sp.groupby(["pitcher", "game_year"]):
        share = g["pitch_type"].value_counts(normalize=True)
        ff, si, fc = share.get("FF", 0), share.get("SI", 0), share.get("FC", 0)
        if max(ff, si) >= 0.15:
            out[(p, y)] = "FF" if ff >= si else "SI"
        elif fc >= 0.15:
            out[(p, y)] = "FC"
    return pd.Series(out, name="fb_type")


def start_velo_drop(sp: pd.DataFrame, fb: pd.Series) -> pd.DataFrame:
    """Per start: late-window fastball velo minus same-start early-window velo."""
    d = sp.merge(fb.rename_axis(["pitcher", "game_year"]).reset_index(),
                 on=["pitcher", "game_year"], how="inner")
    d = d[(d["pitch_type"] == d["fb_type"]) & d["release_speed"].notna()]
    early = d[d["pitch_count"].between(*EARLY)]
    late = d[d["pitch_count"].between(*LATE)]
    e = early.groupby(["game_pk", "pitcher"])["release_speed"].agg(["mean", "size"])
    l = late.groupby(["game_pk", "pitcher"])["release_speed"].agg(["mean", "size"])
    m = e.join(l, lsuffix="_early", rsuffix="_late", how="inner")
    m = m[(m["size_early"] >= MIN_FB_PER_WINDOW) & (m["size_late"] >= MIN_FB_PER_WINDOW)]
    m["velo_drop"] = m["mean_late"] - m["mean_early"]
    return m.reset_index()[["game_pk", "pitcher", "mean_early", "mean_late", "velo_drop"]]


def league_velo_curve(sp: pd.DataFrame, fb: pd.Series, min_pitches: int = 101) -> pd.DataFrame:
    """Fixed-cohort league curve: only starts that reached `min_pitches`, so the
    curve is not distorted by who gets pulled early (survivorship)."""
    d = sp.merge(fb.rename_axis(["pitcher", "game_year"]).reset_index(),
                 on=["pitcher", "game_year"], how="inner")
    d = d[(d["pitch_type"] == d["fb_type"]) & d["release_speed"].notna()]
    tot = d.groupby(["game_pk", "pitcher"])["pitch_count"].transform("max")
    reach = sp.groupby(["game_pk", "pitcher"])["pitch_count"].transform("max")
    reached = sp.loc[reach >= min_pitches, ["game_pk", "pitcher"]].drop_duplicates()
    d = d.merge(reached, on=["game_pk", "pitcher"])
    base = d[d["pitch_count"].between(*EARLY)].groupby(["game_pk", "pitcher"])["release_speed"].mean()
    d = d.join(base.rename("base"), on=["game_pk", "pitcher"])
    d["rel"] = d["release_speed"] - d["base"]
    d["bin"] = pd.cut(d["pitch_count"], PITCH_BINS, labels=PITCH_BIN_LABELS)
    per_start = d.groupby(["game_pk", "pitcher", "bin"], observed=True)["rel"].mean().reset_index()
    c = per_start.groupby("bin", observed=True)["rel"].agg(["mean", "sem", "size"]).reset_index()
    c.columns = ["pitch_bin", "velo_vs_first25", "sem", "n_starts"]
    return c


# ---- Fatigue: outcomes (times through the order) --------------------------
def pa_table(sp: pd.DataFrame) -> pd.DataFrame:
    """One row per completed PA vs the starter, with expected wOBA (xwOBA)."""
    pa = sp[sp["events"].notna() & (sp["woba_denom"] == 1)].copy()
    pa["xwoba"] = pa["estimated_woba_using_speedangle"].fillna(pa["woba_value"])
    return pa[["game_pk", "pitcher", "game_year", "batter", "stand", "p_throws",
               "pitch_count", "n_thruorder_pitcher", "xwoba"]]


def tto_penalty(pa: pd.DataFrame) -> pd.DataFrame:
    g = pa.groupby(["pitcher", "game_year", "n_thruorder_pitcher"])["xwoba"].agg(["mean", "size"])
    w = g.unstack("n_thruorder_pitcher")
    out = pd.DataFrame({
        "xwoba_tto1": w[("mean", 1)], "pa_tto1": w[("size", 1)],
        "xwoba_tto3": w.get(("mean", 3)), "pa_tto3": w.get(("size", 3)),
    })
    out["tto3_penalty"] = out["xwoba_tto3"] - out["xwoba_tto1"]
    return out.reset_index()


def league_tto_curve(pa: pd.DataFrame) -> pd.DataFrame:
    c = (pa[pa["n_thruorder_pitcher"].between(1, 4)]
         .groupby("n_thruorder_pitcher")["xwoba"].agg(["mean", "sem", "size"]).reset_index())
    c.columns = ["times_through", "xwoba", "sem", "pa"]
    return c


# ---- Pitcher-season profiles ---------------------------------------------
def pitcher_profiles(sp: pd.DataFrame, starts: pd.DataFrame) -> pd.DataFrame:
    keep = starts.loc[~starts["excluded"], ["game_pk", "pitcher"]]
    spk = sp.merge(keep, on=["game_pk", "pitcher"])
    fb = primary_fastball(spk)
    v = start_velo_drop(spk, fb)
    v = v.merge(starts[["game_pk", "pitcher", "season", "game_date"]], on=["game_pk", "pitcher"])
    v = v.sort_values(["pitcher", "game_date"])
    v["half"] = v.groupby(["pitcher", "season"]).cumcount() % 2
    vp = v.groupby(["pitcher", "season"]).agg(velo_drop=("velo_drop", "mean"),
                                              velo_sd=("velo_drop", "std"),
                                              late_starts=("velo_drop", "size"))
    halves = v.groupby(["pitcher", "season", "half"])["velo_drop"].mean().unstack("half")
    vp["velo_drop_h0"], vp["velo_drop_h1"] = halves[0], halves[1]

    pa = pa_table(spk)
    pa = pa.merge(starts[["game_pk", "pitcher", "game_date"]], on=["game_pk", "pitcher"])
    # split-half by start (odd/even) for reliability
    order = pa[["pitcher", "game_year", "game_date"]].drop_duplicates().sort_values(["pitcher", "game_date"])
    order["half"] = order.groupby(["pitcher", "game_year"]).cumcount() % 2
    pa = pa.merge(order, on=["pitcher", "game_year", "game_date"])
    t = tto_penalty(pa).rename(columns={"game_year": "season"}).set_index(["pitcher", "season"])
    th = {h: tto_penalty(pa[pa["half"] == h]).rename(columns={"game_year": "season"})
          .set_index(["pitcher", "season"])["tto3_penalty"] for h in (0, 1)}
    t["tto3_penalty_h0"], t["tto3_penalty_h1"] = th[0], th[1]

    names = starts.groupby(["pitcher", "season"])["player_name"].first()
    n_starts = starts[~starts["excluded"]].groupby(["pitcher", "season"]).size().rename("starts_used")
    prof = vp.join(t, how="outer").join(names).join(n_starts)
    prof["velo_ok"] = prof["late_starts"] >= MIN_LATE_STARTS
    prof["tto_ok"] = (prof["pa_tto3"] >= MIN_TTO3_PA) & (prof["pa_tto1"] >= MIN_TTO1_PA)
    return prof.reset_index()


# ---- Statistics ------------------------------------------------------------
def spearman_brown(r_half: float) -> float:
    return 2 * r_half / (1 + r_half)


def _cluster_index(groups: np.ndarray | None, n: int, n_boot: int, rng) -> list[np.ndarray]:
    """Bootstrap row indices; resamples whole pitchers when groups are given
    (a pitcher can appear in several season pairs)."""
    if groups is None:
        return list(rng.integers(0, n, (n_boot, n)))
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        out.append(np.concatenate([rows[g] for g in pick]))
    return out


def boot_r(x, y, n: int = 5000, seed: int = 7, groups=None):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = np.corrcoef(x, y)[0, 1]
    bs = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in _cluster_index(groups, len(x), n, rng)])
    return r, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


@dataclass
class DecisionResult:
    target: str
    n: int
    mae: dict
    gain_vs_league: dict      # proportional MAE reduction
    gain_ci: dict             # 95% paired bootstrap CI of the reduction
    p_value: dict             # one-sided bootstrap p (gain <= 0)


def loo_decision_test(prev_feature: dict[str, np.ndarray], prev_target: np.ndarray,
                      reliability_halves: tuple[np.ndarray, np.ndarray],
                      next_target: np.ndarray, target: str,
                      n_boot: int = 5000, seed: int = 11, groups=None) -> DecisionResult:
    """Leave-one-pitcher-out prediction of next-season target.

    Models:
      league      : mean of the other pitchers' previous-season values (league rule)
      individual  : the pitcher's own previous-season value (raw)
      shrunk      : league + reliability * (own - league), reliability from the
                    other pitchers' split-half correlation (Spearman-Brown)
      <feature>   : OLS on other pitchers, previous-season feature -> next target
    """
    n = len(next_target)
    h0, h1 = reliability_halves
    preds = {"league": np.empty(n), "individual": np.empty(n), "shrunk": np.empty(n)}
    for f in prev_feature:
        preds[f"ols_{f}"] = np.empty(n)
    gr = np.arange(n) if groups is None else np.asarray(groups)
    for i in range(n):
        o = gr != gr[i]   # leave out every row of this pitcher
        lg = prev_target[o].mean()
        rel = max(0.0, spearman_brown(np.corrcoef(h0[o], h1[o])[0, 1]))
        preds["league"][i] = lg
        preds["individual"][i] = prev_target[i]
        preds["shrunk"][i] = lg + rel * (prev_target[i] - lg)
        for f, x in prev_feature.items():
            b1, b0 = np.polyfit(x[o], next_target[o], 1)
            preds[f"ols_{f}"][i] = b0 + b1 * x[i]
    err = {k: np.abs(v - next_target) for k, v in preds.items()}
    mae = {k: float(e.mean()) for k, e in err.items()}
    rng = np.random.default_rng(seed)
    idx = _cluster_index(groups, n, n_boot, rng)
    gain, ci, p = {}, {}, {}
    base = err["league"]
    for k, e in err.items():
        if k == "league":
            continue
        bs = np.array([1 - e[i].mean() / base[i].mean() for i in idx])
        gain[k] = 1 - mae[k] / mae["league"]
        ci[k] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        p[k] = float((bs <= 0).mean())
    return DecisionResult(target, n, mae, gain, ci, p)


def benjamini_hochberg(p: dict[str, float]) -> dict[str, float]:
    keys = list(p)
    vals = np.array([p[k] for k in keys])
    order = np.argsort(vals)
    m = len(vals)
    adj = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = m - rank
        prev = min(prev, vals[i] * m / k)
        adj[i] = prev
    return dict(zip(keys, adj))
