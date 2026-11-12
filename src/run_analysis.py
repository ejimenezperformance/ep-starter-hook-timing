"""Run the full ep-starter-hook-timing analysis.

Corre el análisis completo con todas las temporadas disponibles en data/
(statcast_<year>.csv.gz). Outputs in outputs/.

v2 (pre-specified before running it): the primary tests pool every consecutive
season pair available (2023->24, 2024->25, 2025->26). The v1 pair 2025->2026 is
kept and reported on its own. H5 is new in v2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hook_timing as ht  # noqa: E402

ROOT = ht.ROOT
V1_PAIR = (2025, 2026)
LIVE_BINS = [50, 75, 100, 200]
LIVE_BIN_LABELS = ["51-75", "76-100", "101+"]


def available_seasons(root: Path) -> list[int]:
    return sorted(int(p.name[9:13]) for p in (root / "data").glob("statcast_*.csv.gz"))


def load(season: int, root: Path = ROOT) -> pd.DataFrame:
    return pd.read_csv(root / "data" / f"statcast_{season}.csv.gz", low_memory=False)


def ok(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna(False).astype(bool)


# ---- fixed-effects OLS (within transform), SEs clustered by game -----------
def fe_ols(df: pd.DataFrame, y: str, xs: list[str], fe: str, cluster: str):
    d = df[[y, *xs, fe, cluster]].dropna().copy()
    cols = [y, *xs]
    d[cols] = d[cols] - d.groupby(fe)[cols].transform("mean")
    m = sm.OLS(d[y], d[xs]).fit(cov_type="cluster", cov_kwds={"groups": d[cluster]})
    return m, len(d)


# ---- PA table with the live velo signal ------------------------------------
def live_pa_table(sp: pd.DataFrame, starts: pd.DataFrame) -> pd.DataFrame:
    """One row per PA vs the starter, with live_drop = mean of the last 5
    fastballs thrown BEFORE the PA started minus the start's first-25 baseline."""
    keep = starts.loc[~starts["excluded"], ["game_pk", "pitcher"]]
    d = sp.merge(keep, on=["game_pk", "pitcher"])
    fb = ht.primary_fastball(d).rename_axis(["pitcher", "game_year"]).reset_index()
    d = d.merge(fb, on=["pitcher", "game_year"], how="inner").sort_values(
        ["game_pk", "pitcher", "pitch_count"])
    isfb = (d["pitch_type"] == d["fb_type"]) & d["release_speed"].notna()
    d["fb_velo"] = d["release_speed"].where(isfb)
    base = d[d["pitch_count"].between(*ht.EARLY)].groupby(["game_pk", "pitcher"])["fb_velo"].mean()
    d = d.join(base.rename("base"), on=["game_pk", "pitcher"])
    fbd = d[isfb].copy()
    fbd["roll5"] = fbd.groupby(["game_pk", "pitcher"])["fb_velo"].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    d = d.merge(fbd[["game_pk", "pitcher", "pitch_count", "roll5"]],
                on=["game_pk", "pitcher", "pitch_count"], how="left")
    # known BEFORE each pitch (current pitch excluded) ...
    d["roll5_prior"] = d.groupby(["game_pk", "pitcher"])["roll5"].transform(lambda s: s.ffill().shift(1))
    d["live_drop"] = d["roll5_prior"] - d["base"]
    # ... and taken at the first pitch of the PA
    g = d.groupby(["game_pk", "pitcher", "at_bat_number"])
    d["live_drop"] = g["live_drop"].transform("first")
    d["pc_start"] = g["pitch_count"].transform("first")
    pa = d[d["events"].notna() & (d["woba_denom"] == 1) & (d["pc_start"] > 50)].copy()
    pa["xwoba"] = pa["estimated_woba_using_speedangle"].fillna(pa["woba_value"])
    pa = pa.dropna(subset=["live_drop", "xwoba"])
    pa["tto2"] = (pa["n_thruorder_pitcher"] == 2).astype(int)
    pa["tto3"] = (pa["n_thruorder_pitcher"] >= 3).astype(int)
    pa["same_hand"] = (pa["stand"] == pa["p_throws"]).astype(int)
    pa["pitcher_season"] = pa["pitcher"].astype(str) + "_" + pa["game_year"].astype(str)
    pa["bin"] = pd.cut(pa["pc_start"], LIVE_BINS, labels=LIVE_BIN_LABELS).astype(str)
    return pa


def expected_curves(sp: pd.DataFrame, starts: pd.DataFrame, rel1: float) -> pd.DataFrame:
    """Each pitcher-season's velo vs first 25 pitches by pitch-count bin, shrunk
    toward that season's league bin mean (k = n*r1 / (1+(n-1)*r1), n = starts)."""
    keep = starts.loc[~starts["excluded"], ["game_pk", "pitcher"]]
    d = sp.merge(keep, on=["game_pk", "pitcher"])
    fb = ht.primary_fastball(d).rename_axis(["pitcher", "game_year"]).reset_index()
    d = d.merge(fb, on=["pitcher", "game_year"])
    d = d[(d["pitch_type"] == d["fb_type"]) & d["release_speed"].notna()]
    base = d[d["pitch_count"].between(*ht.EARLY)].groupby(["game_pk", "pitcher"])["release_speed"].mean()
    d = d.join(base.rename("base"), on=["game_pk", "pitcher"]).dropna(subset=["base"])
    d["rel"] = d["release_speed"] - d["base"]
    d = d[d["pitch_count"] > 50]
    d["bin"] = pd.cut(d["pitch_count"], LIVE_BINS, labels=LIVE_BIN_LABELS).astype(str)
    ps = d.groupby(["game_year", "pitcher", "bin", "game_pk"])["rel"].mean().reset_index()
    c = ps.groupby(["game_year", "pitcher", "bin"])["rel"].agg(["mean", "size"]).reset_index()
    lg = ps.groupby(["game_year", "bin"])["rel"].mean().rename("league").reset_index()
    c = c.merge(lg, on=["game_year", "bin"])
    k = c["size"] * rel1 / (1 + (c["size"] - 1) * rel1)
    c["expected"] = c["league"] + k * (c["mean"] - c["league"])
    return c.rename(columns={"size": "n_starts"})


# ---- H4 / H5 ---------------------------------------------------------------
def h4_model(pa: pd.DataFrame) -> dict:
    xs = ["live_drop", "pc_start", "tto2", "tto3", "same_hand"]
    m, n = fe_ols(pa, "xwoba", xs, "pitcher_season", "game_pk")
    b = m.params["live_drop"]
    lo, hi = m.conf_int().loc["live_drop"]
    return {"n_pa": int(n), "n_pitcher_seasons": int(pa["pitcher_season"].nunique()),
            "seasons": sorted(int(s) for s in pa["game_year"].unique()),
            "xwoba_per_mph_lost": float(-b), "ci95": [float(-hi), float(-lo)],
            "p_two_sided": float(m.pvalues["live_drop"])}


def h5_model(pa: pd.DataFrame, curves: pd.DataFrame) -> dict:
    """H5: is an UNUSUAL velo drop costlier than the drop the pitcher USUALLY has?

    live_drop = expected + unusual
      expected : the pitcher's own PREVIOUS-season fade at this pitch-count bin
                 (shrunk to league; out of sample by construction)
      unusual  : live_drop - expected
    xwOBA ~ expected + unusual + pitch count + times through + platoon,
    pitcher-season fixed effects, SEs clustered by game.

    Raw world      : both parts cost the same per mph.
    Own-curve world: the expected part costs ~0, the unusual part costs > 0.
    H5 CONFIRMED if cost(unusual) - cost(expected) > 0 (one-sided, BH-adjusted).
    Measurement noise in the live signal sits in 'unusual' and biases its cost
    toward 0, so the test is conservative.
    """
    prev = curves[["game_year", "pitcher", "bin", "expected"]].copy()
    prev["game_year"] = prev["game_year"] + 1          # last season's curve
    d = pa.merge(prev, on=["game_year", "pitcher", "bin"], how="inner").dropna(subset=["expected"])
    d["unusual"] = d["live_drop"] - d["expected"]
    ctrl = ["pc_start", "tto2", "tto3", "same_hand"]
    m, n = fe_ols(d, "xwoba", ["expected", "unusual", *ctrl], "pitcher_season", "game_pk")
    cost_exp, cost_unu = -m.params["expected"], -m.params["unusual"]   # xwOBA per mph LOST
    t = m.t_test("expected - unusual = 0")      # = cost(unusual) - cost(expected)
    diff = float(t.effect[0])
    se = float(t.sd[0][0])
    z = diff / se
    ci_e = m.conf_int().loc["expected"]
    ci_u = m.conf_int().loc["unusual"]
    return {
        "n_pa": int(n), "n_pitcher_seasons": int(d["pitcher_season"].nunique()),
        "seasons": sorted(int(s) for s in d["game_year"].unique()),
        "sd_expected": float(d["expected"].std()), "sd_unusual": float(d["unusual"].std()),
        "cost_expected_per_mph": float(cost_exp), "cost_expected_ci95": [float(-ci_e[1]), float(-ci_e[0])],
        "cost_unusual_per_mph": float(cost_unu), "cost_unusual_ci95": [float(-ci_u[1]), float(-ci_u[0])],
        "diff_unusual_minus_expected": diff, "diff_ci95": [diff - 1.96 * se, diff + 1.96 * se],
        "p_one_sided": float(stats.norm.sf(z)),
    }


# ---- main --------------------------------------------------------------------
def main(root: Path = ROOT, verbose: bool = True) -> dict:
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    seasons = available_seasons(root)
    pairs = [(a, b) for a, b in zip(seasons, seasons[1:]) if b == a + 1]
    manual = ht.load_manual_exclusions(root)
    sps, sts = [], []
    for y in seasons:
        sp_y = ht.starter_pitches(load(y, root))
        sps.append(sp_y)
        sts.append(ht.flag_absences(ht.start_table(sp_y), manual=manual))
    sp = pd.concat(sps, ignore_index=True)
    starts = pd.concat(sts, ignore_index=True)
    starts.to_csv(out / "starts_with_exclusions.csv", index=False, encoding="utf-8")

    prof = ht.pitcher_profiles(sp, starts)
    prof.to_csv(out / "pitcher_season_profiles.csv", index=False, encoding="utf-8")

    keep = starts.loc[~starts["excluded"], ["game_pk", "pitcher"]]
    sp_keep = sp.merge(keep, on=["game_pk", "pitcher"])
    vcs, tcs = [], []
    for y in seasons:
        spy = sp_keep[sp_keep["game_year"] == y]
        vc = ht.league_velo_curve(spy, ht.primary_fastball(spy))
        vc["season"] = y
        tc = ht.league_tto_curve(ht.pa_table(spy))
        tc["season"] = y
        vcs.append(vc)
        tcs.append(tc)
    pd.concat(vcs).to_csv(out / "league_velo_curve.csv", index=False)
    pd.concat(tcs).to_csv(out / "league_tto_curve.csv", index=False)

    # ---- season pairs -------------------------------------------------------
    rows = []
    for a, b in pairs:
        pa_ = prof[prof["season"] == a].set_index("pitcher")
        pb_ = prof[prof["season"] == b].set_index("pitcher")
        j = pa_.join(pb_, lsuffix="_prev", rsuffix="_next", how="inner").reset_index()
        j["pair"] = f"{a}->{b}"
        rows.append(j)
    pairs_df = pd.concat(rows, ignore_index=True)
    pairs_df["player_name"] = pairs_df["player_name_next"].fillna(pairs_df["player_name_prev"])

    res: dict = {"version": "v2", "seasons": seasons, "pairs": [f"{a}->{b}" for a, b in pairs],
                 "filters": {
                     "starts_total": int(len(starts)),
                     "starts_excluded": int(starts["excluded"].sum()),
                     "excluded_by_reason": starts.loc[starts["excluded"], "exclude_reason"].value_counts().to_dict(),
                     "pitches_by_season": {int(y): int((sp_keep["game_year"] == y).sum()) for y in seasons}}}
    tests_p = {}

    # H1 / H2: stability, pooled over pairs (bootstrap resamples pitchers)
    latest = seasons[-1]
    for key, col, okc in (("H1_velo_drop", "velo_drop", "velo_ok"),
                          ("H2_tto3_penalty", "tto3_penalty", "tto_ok")):
        q = pairs_df[ok(pairs_df, f"{okc}_prev") & ok(pairs_df, f"{okc}_next")]
        q = q.dropna(subset=[f"{col}_prev", f"{col}_next"])
        r, lo, hi = ht.boot_r(q[f"{col}_prev"], q[f"{col}_next"], groups=q["pitcher"].values)
        p = stats.pearsonr(q[f"{col}_prev"], q[f"{col}_next"]).pvalue
        tests_p[key] = p
        rels = {}
        for y in seasons:
            w = prof[(prof["season"] == y) & ok(prof, okc)].dropna(subset=[f"{col}_h0", f"{col}_h1"])
            if len(w) > 5:
                rels[int(y)] = float(ht.spearman_brown(np.corrcoef(w[f"{col}_h0"], w[f"{col}_h1"])[0, 1]))
        by_pair = {}
        for pr, g in q.groupby("pair"):
            rr, l2, h2 = ht.boot_r(g[f"{col}_prev"], g[f"{col}_next"])
            by_pair[pr] = {"n": int(len(g)), "r": float(rr), "ci95": [float(l2), float(h2)]}
        res[key] = {"n_pairs": int(len(q)), "n_pitchers": int(q["pitcher"].nunique()),
                    "r_pooled": float(r), "ci95": [float(lo), float(hi)], "p": float(p),
                    "split_half_reliability_by_season": rels, "by_pair": by_pair,
                    "mean_prev": float(q[f"{col}_prev"].mean()), "mean_next": float(q[f"{col}_next"].mean()),
                    "confirmed": bool(r >= ht.CONFIRM_R and lo > 0)}
        q[["pitcher", "player_name", "pair", f"{col}_prev", f"{col}_next"]].to_csv(
            out / f"{key}_pairs.csv", index=False, encoding="utf-8")

    # H3: decision test, leave-one-PITCHER-out over pooled pairs
    need = ["velo_drop_prev", "velo_drop_next", "tto3_penalty_prev", "tto3_penalty_next",
            "velo_drop_h0_prev", "velo_drop_h1_prev", "tto3_penalty_h0_prev", "tto3_penalty_h1_prev"]
    both = pairs_df[ok(pairs_df, "velo_ok_prev") & ok(pairs_df, "velo_ok_next")
                    & ok(pairs_df, "tto_ok_prev") & ok(pairs_df, "tto_ok_next")].dropna(subset=need)
    grp = both["pitcher"].values
    d1 = ht.loo_decision_test({}, both["velo_drop_prev"].values,
                              (both["velo_drop_h0_prev"].values, both["velo_drop_h1_prev"].values),
                              both["velo_drop_next"].values, "velo_drop_next", groups=grp)
    d2 = ht.loo_decision_test({"velo_drop": both["velo_drop_prev"].values}, both["tto3_penalty_prev"].values,
                              (both["tto3_penalty_h0_prev"].values, both["tto3_penalty_h1_prev"].values),
                              both["tto3_penalty_next"].values, "tto3_penalty_next", groups=grp)
    dec = {}
    for d in (d1, d2):
        dec[d.target] = {"n_pairs": d.n, "n_pitchers": int(len(np.unique(grp))), "mae": d.mae,
                         "gain_vs_league": d.gain_vs_league, "gain_ci95": d.gain_ci,
                         "p_one_sided": d.p_value}
        for k, pv in d.p_value.items():
            tests_p[f"H3_{d.target}_{k}"] = pv
    res["H3_decision"] = dec

    # v1 replication: the 2025->2026 pair alone
    v1 = both[both["pair"] == f"{V1_PAIR[0]}->{V1_PAIR[1]}"]
    if len(v1) > 10:
        dv = ht.loo_decision_test({}, v1["velo_drop_prev"].values,
                                  (v1["velo_drop_h0_prev"].values, v1["velo_drop_h1_prev"].values),
                                  v1["velo_drop_next"].values, "v1", groups=v1["pitcher"].values)
        res["H3_v1_2025_2026_shrunk"] = {"n": dv.n, "gain": dv.gain_vs_league["shrunk"],
                                         "ci95": dv.gain_ci["shrunk"]}

    # per-start reliability for shrinkage (latest season, 76-100 window)
    rel = max(0.0, res["H1_velo_drop"]["split_half_reliability_by_season"].get(latest, 0.0))
    allv = prof[ok(prof, "velo_ok")]
    n_half = allv.loc[allv["season"] == latest, "late_starts"].median() / 2
    r_half = rel / (2 - rel)
    rel1 = float(r_half / (n_half - (n_half - 1) * r_half)) if rel > 0 else 0.0

    # H4 + H5 (all seasons pooled; H5 needs the previous season's curve)
    pa = live_pa_table(sp, starts)
    res["H4_live_velo"] = h4_model(pa)
    tests_p["H4_live_velo"] = res["H4_live_velo"]["p_two_sided"]
    curves = expected_curves(sp, starts, rel1)
    curves.to_csv(out / "expected_curves_by_pitcher_season.csv", index=False, encoding="utf-8")
    res["H5_own_curve"] = h5_model(pa, curves)
    tests_p["H5_own_curve"] = res["H5_own_curve"]["p_one_sided"]

    adj = ht.benjamini_hochberg(tests_p)
    res["bh_adjusted_p"] = adj
    for t, d in dec.items():
        d["confirmed"] = {k: bool(d["gain_ci95"][k][0] > 0 and d["gain_vs_league"][k] >= ht.CONFIRM_MAE_GAIN
                                  and adj[f"H3_{t}_{k}"] < 0.05) for k in d["gain_vs_league"]}
    h4 = res["H4_live_velo"]
    h4["confirmed"] = bool(adj["H4_live_velo"] < 0.05 and h4["ci95"][0] > 0)
    h5 = res["H5_own_curve"]
    h5["confirmed"] = bool(adj["H5_own_curve"] < 0.05 and h5["diff_ci95"][0] > 0)
    for k in ("H1_velo_drop", "H2_tto3_penalty"):
        res[k]["confirmed"] = bool(res[k]["confirmed"] and adj[k] < 0.05)

    # ---- application table: all seasons combined, shrunk to league ----------
    comb = (allv.assign(w=allv["late_starts"], wv=allv["velo_drop"] * allv["late_starts"])
            .groupby("pitcher").agg(wv=("wv", "sum"), w=("w", "sum"),
                                     player_name=("player_name", "last"),
                                     seasons=("season", "nunique"), last_season=("season", "max")))
    comb["velo_drop_raw"] = comb["wv"] / comb["w"]
    lg = comb["velo_drop_raw"].mean()
    comb["reliability"] = comb["w"] * rel1 / (1 + (comb["w"] - 1) * rel1)
    comb["velo_drop_shrunk"] = lg + comb["reliability"] * (comb["velo_drop_raw"] - lg)
    comb = comb.drop(columns=["wv"]).rename(columns={"w": "late_starts"}).sort_values("velo_drop_shrunk")
    q20, q80 = comb["velo_drop_shrunk"].quantile([0.2, 0.8])
    comb["profile"] = np.select([comb["velo_drop_shrunk"] <= q20, comb["velo_drop_shrunk"] >= q80],
                                ["early_fade", "holds_velo"], "league_typical")
    ann = manual[manual["action"] == "annotate"].copy()
    comb["injury_notes"] = ""
    if len(ann):
        ann["note"] = ann["season"].astype(str) + ": " + ann["injury_type"]
        notes = ann.groupby("pitcher_name")["note"].agg("; ".join)
        comb["injury_notes"] = comb["player_name"].map(notes).fillna("")
    comb.to_csv(out / "starter_fatigue_profiles.csv", encoding="utf-8")
    res["application"] = {"n_starters": int(len(comb)), "league_mean_drop": float(lg),
                          "early_fade_cut": float(q20), "holds_velo_cut": float(q80),
                          "per_start_reliability": rel1,
                          "n_early_fade": int((comb["profile"] == "early_fade").sum())}

    # ---- out-of-sample profile curves: groups set on season N-1, shown in N --
    prev_y, next_y = seasons[-2], seasons[-1]
    p_prev = prof[(prof["season"] == prev_y) & ok(prof, "velo_ok")].copy()
    a20, b80 = p_prev["velo_drop"].quantile([0.2, 0.8])
    p_prev["group_prev"] = np.select([p_prev["velo_drop"] <= a20, p_prev["velo_drop"] >= b80],
                                     ["early_fade", "holds_velo"], "league_typical")
    s_next = sp_keep[sp_keep["game_year"] == next_y]
    fbn = ht.primary_fastball(s_next).rename_axis(["pitcher", "game_year"]).reset_index()
    d = s_next.merge(fbn, on=["pitcher", "game_year"])
    d = d[(d["pitch_type"] == d["fb_type"]) & d["release_speed"].notna()]
    reach = d.groupby(["game_pk", "pitcher"])["pitch_count"].transform("max")
    d = d[reach >= ht.LATE[0]]
    base = d[d["pitch_count"].between(*ht.EARLY)].groupby(["game_pk", "pitcher"])["release_speed"].mean()
    d = d.join(base.rename("base"), on=["game_pk", "pitcher"]).dropna(subset=["base"])
    d["rel"] = d["release_speed"] - d["base"]
    d["bin"] = pd.cut(d["pitch_count"], ht.PITCH_BINS[:-1], labels=ht.PITCH_BIN_LABELS[:-1])
    d = d.merge(p_prev[["pitcher", "group_prev"]], on="pitcher")
    ps = d.groupby(["group_prev", "pitcher", "bin"], observed=True)["rel"].mean().reset_index()
    gc = ps.groupby(["group_prev", "bin"], observed=True)["rel"].agg(["mean", "sem", "size"]).reset_index()
    gc.columns = ["group_prev", "pitch_bin", "velo_vs_first25_next", "sem", "n_pitchers"]
    gc.to_csv(out / "profile_curves_out_of_sample.csv", index=False)
    late = gc[gc["pitch_bin"] == "76-100"].set_index("group_prev")["velo_vs_first25_next"]
    res["profile_groups_out_of_sample"] = {
        "groups_defined_on": f"{prev_y} velo_drop (bottom/top 20%)", "shown_in": next_y,
        "late_velo_next": late.to_dict(),
        "gap_fade_vs_hold_mph": float(late["holds_velo"] - late["early_fade"]),
        "n_pitchers": gc[gc["pitch_bin"] == "76-100"].set_index("group_prev")["n_pitchers"].to_dict()}

    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    if verbose:
        print(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    main()
