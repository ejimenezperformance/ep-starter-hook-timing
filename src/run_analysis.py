"""Run the full ep-starter-hook-timing analysis on data/statcast_2025/2026.

Corre el análisis completo. Outputs in outputs/ and figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hook_timing as ht  # noqa: E402

ROOT = ht.ROOT
PREV, NEXT = 2025, 2026


def load(season: int, root: Path = ROOT) -> pd.DataFrame:
    return pd.read_csv(root / "data" / f"statcast_{season}.csv.gz", low_memory=False)


def live_velo_model(sp: pd.DataFrame, starts: pd.DataFrame) -> dict:
    """H4: inside a start, does the live fastball-velo drop come with worse
    contact quality, within the same pitcher? (monitoring metric check)"""
    keep = starts.loc[~starts["excluded"], ["game_pk", "pitcher"]]
    d = sp.merge(keep, on=["game_pk", "pitcher"])
    fb = ht.primary_fastball(d).rename_axis(["pitcher", "game_year"]).reset_index()
    d = d.merge(fb, on=["pitcher", "game_year"], how="inner").sort_values(
        ["game_pk", "pitcher", "pitch_count"])
    isfb = (d["pitch_type"] == d["fb_type"]) & d["release_speed"].notna()
    d["fb_velo"] = d["release_speed"].where(isfb)
    base = d[d["pitch_count"].between(*ht.EARLY)].groupby(["game_pk", "pitcher"])["fb_velo"].mean()
    d = d.join(base.rename("base"), on=["game_pk", "pitcher"])
    # rolling mean of the last 5 fastballs thrown so far (fastballs only)
    fbd = d[isfb].copy()
    fbd["roll5"] = fbd.groupby(["game_pk", "pitcher"])["fb_velo"].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    d = d.merge(fbd[["game_pk", "pitcher", "pitch_count", "roll5"]],
                on=["game_pk", "pitcher", "pitch_count"], how="left")
    # value known BEFORE each pitch (excludes the current pitch), then the value
    # at the first pitch of the PA = what the dugout knew when the PA started
    d["roll5_prior"] = d.groupby(["game_pk", "pitcher"])["roll5"].transform(lambda s: s.ffill().shift(1))
    d["live_drop"] = d["roll5_prior"] - d["base"]
    first = d.groupby(["game_pk", "pitcher", "at_bat_number"])["live_drop"].transform("first")
    d["live_drop"] = first
    pa = d[d["events"].notna() & (d["woba_denom"] == 1) & (d["pitch_count"] > 50)].copy()
    pa["xwoba"] = pa["estimated_woba_using_speedangle"].fillna(pa["woba_value"])
    pa = pa.dropna(subset=["live_drop", "xwoba"])
    pa["tto"] = pa["n_thruorder_pitcher"].clip(upper=3).astype(int)
    pa["same_hand"] = (pa["stand"] == pa["p_throws"]).astype(int)
    m = smf.ols("xwoba ~ live_drop + pitch_count + C(tto) + same_hand + C(pitcher)", data=pa).fit(
        cov_type="cluster", cov_kwds={"groups": pa["game_pk"]})
    b = m.params["live_drop"]
    lo, hi = m.conf_int().loc["live_drop"]
    # coefficient is per +1 mph; report per 1 mph LOST (sign flipped)
    return {"n_pa": int(len(pa)), "n_pitchers": int(pa["pitcher"].nunique()),
            "xwoba_per_mph_lost": float(-b), "ci95": [float(-hi), float(-lo)],
            "p_two_sided": float(m.pvalues["live_drop"]),
            "share_pa_live_drop_le_minus1": float((pa["live_drop"] <= -1).mean())}


def main(root: Path = ROOT, verbose: bool = True) -> dict:
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    manual = ht.load_manual_exclusions(root)
    sps, sts = [], []
    for y in (PREV, NEXT):
        sp = ht.starter_pitches(load(y, root))
        st = ht.flag_absences(ht.start_table(sp), manual=manual)
        sps.append(sp)
        sts.append(st)
    sp = pd.concat(sps, ignore_index=True)
    starts = pd.concat(sts, ignore_index=True)
    starts.to_csv(out / "starts_with_exclusions.csv", index=False, encoding="utf-8")

    prof = ht.pitcher_profiles(sp, starts)
    prof.to_csv(out / "pitcher_season_profiles.csv", index=False, encoding="utf-8")

    # ---- comparison line: league curves (2026) ----------------------------
    keep = starts.loc[~starts["excluded"], ["game_pk", "pitcher"]]
    sp_keep = sp.merge(keep, on=["game_pk", "pitcher"])
    curves = {}
    for y in (PREV, NEXT):
        spy = sp_keep[sp_keep["game_year"] == y]
        vc = ht.league_velo_curve(spy, ht.primary_fastball(spy))
        vc["season"] = y
        tc = ht.league_tto_curve(ht.pa_table(spy))
        tc["season"] = y
        curves[y] = (vc, tc)
    pd.concat([c[0] for c in curves.values()]).to_csv(out / "league_velo_curve.csv", index=False)
    pd.concat([c[1] for c in curves.values()]).to_csv(out / "league_tto_curve.csv", index=False)

    # ---- pair seasons -----------------------------------------------------
    a = prof[prof["season"] == PREV].set_index("pitcher")
    b = prof[prof["season"] == NEXT].set_index("pitcher")
    pair = a.join(b, lsuffix="_25", rsuffix="_26", how="inner")
    pair["player_name"] = pair["player_name_26"].fillna(pair["player_name_25"])

    res: dict = {"filters": {
        "starts_total": int(len(starts)),
        "starts_excluded": int(starts["excluded"].sum()),
        "excluded_by_reason": starts.loc[starts["excluded"], "exclude_reason"].value_counts().to_dict()}}

    # H1 velo stability / H2 TTO stability
    tests_p = {}
    for key, col, ok in (("H1_velo_drop", "velo_drop", "velo_ok"),
                         ("H2_tto3_penalty", "tto3_penalty", "tto_ok")):
        q = pair[pair[f"{ok}_25"].fillna(False).astype(bool) & pair[f"{ok}_26"].fillna(False).astype(bool)]
        q = q.dropna(subset=[f"{col}_25", f"{col}_26"])
        r, lo, hi = ht.boot_r(q[f"{col}_25"], q[f"{col}_26"])
        # within-season split-half reliability (2026), Spearman-Brown corrected
        w = prof[(prof["season"] == NEXT) & prof[ok].fillna(False).astype(bool)].dropna(
            subset=[f"{col}_h0", f"{col}_h1"])
        rel = ht.spearman_brown(np.corrcoef(w[f"{col}_h0"], w[f"{col}_h1"])[0, 1])
        from scipy import stats
        p = stats.pearsonr(q[f"{col}_25"], q[f"{col}_26"]).pvalue
        tests_p[key] = p
        res[key] = {"n": int(len(q)), "r_2025_2026": float(r), "ci95": [float(lo), float(hi)],
                    "p": float(p), "split_half_reliability_2026": float(rel),
                    "mean_2025": float(q[f"{col}_25"].mean()), "mean_2026": float(q[f"{col}_26"].mean()),
                    "sd_2025": float(q[f"{col}_25"].std()),
                    "confirmed": bool(r >= ht.CONFIRM_R and lo > 0)}
        q[["player_name", f"{col}_25", f"{col}_26"]].to_csv(out / f"{key}_pairs.csv", encoding="utf-8")

    # H3 decision test: predict each starter's 2026 values (LOO)
    both = pair[pair["velo_ok_25"].fillna(False).astype(bool) & pair["velo_ok_26"].fillna(False).astype(bool)
                & pair["tto_ok_25"].fillna(False).astype(bool) & pair["tto_ok_26"].fillna(False).astype(bool)]
    both = both.dropna(subset=["velo_drop_25", "velo_drop_26", "tto3_penalty_25", "tto3_penalty_26",
                               "velo_drop_h0_25", "velo_drop_h1_25", "tto3_penalty_h0_25", "tto3_penalty_h1_25"])
    dec = {}
    d1 = ht.loo_decision_test({}, both["velo_drop_25"].values,
                              (both["velo_drop_h0_25"].values, both["velo_drop_h1_25"].values),
                              both["velo_drop_26"].values, "velo_drop_2026")
    d2 = ht.loo_decision_test({"velo_drop": both["velo_drop_25"].values}, both["tto3_penalty_25"].values,
                              (both["tto3_penalty_h0_25"].values, both["tto3_penalty_h1_25"].values),
                              both["tto3_penalty_26"].values, "tto3_penalty_2026")
    for d in (d1, d2):
        dec[d.target] = {"n": d.n, "mae": d.mae, "gain_vs_league": d.gain_vs_league,
                         "gain_ci95": d.gain_ci, "p_one_sided": d.p_value}
        for k, pv in d.p_value.items():
            tests_p[f"H3_{d.target}_{k}"] = pv
    res["H3_decision"] = dec

    # H4 live monitoring metric (2026)
    res["H4_live_velo"] = live_velo_model(sp[sp["game_year"] == NEXT],
                                          starts[starts["season"] == NEXT])
    tests_p["H4_live_velo"] = res["H4_live_velo"]["p_two_sided"]

    adj = ht.benjamini_hochberg(tests_p)
    res["bh_adjusted_p"] = adj
    for t, d in dec.items():
        d["confirmed"] = {k: bool(d["gain_ci95"][k][0] > 0 and d["gain_vs_league"][k] >= ht.CONFIRM_MAE_GAIN
                                  and adj[f"H3_{t}_{k}"] < 0.05) for k in d["gain_vs_league"]}
    h4 = res["H4_live_velo"]
    h4["confirmed"] = bool(adj["H4_live_velo"] < 0.05 and h4["ci95"][0] > 0)
    for k in ("H1_velo_drop", "H2_tto3_penalty"):
        res[k]["confirmed"] = bool(res[k]["confirmed"] and adj[k] < 0.05)

    # ---- application table: 2025+2026 combined, shrunk to league ---------
    rel = max(0.0, res["H1_velo_drop"]["split_half_reliability_2026"])
    allv = prof[prof["velo_ok"].fillna(False).astype(bool)]
    comb = (allv.assign(w=allv["late_starts"], wv=allv["velo_drop"] * allv["late_starts"])
            .groupby("pitcher").agg(wv=("wv", "sum"), w=("w", "sum"),
                                     player_name=("player_name", "last"),
                                     seasons=("season", "nunique")))
    comb["velo_drop_raw"] = comb["wv"] / comb["w"]
    lg = comb["velo_drop_raw"].mean()
    # reliability grows with starts: k = n*rel1 / (1 + (n-1)*rel1) using per-start reliability
    # invert Spearman-Brown: full-season -> half-season -> one start
    n_half = allv.loc[allv["season"] == NEXT, "late_starts"].median() / 2
    r_half = rel / (2 - rel)
    rel1 = r_half / (n_half - (n_half - 1) * r_half) if rel > 0 else 0
    comb["reliability"] = comb["w"] * rel1 / (1 + (comb["w"] - 1) * rel1)
    comb["velo_drop_shrunk"] = lg + comb["reliability"] * (comb["velo_drop_raw"] - lg)
    comb = comb.drop(columns=["wv"]).rename(columns={"w": "late_starts"}).sort_values("velo_drop_shrunk")
    q20, q80 = comb["velo_drop_shrunk"].quantile([0.2, 0.8])
    comb["profile"] = np.select([comb["velo_drop_shrunk"] <= q20, comb["velo_drop_shrunk"] >= q80],
                                ["early_fade", "holds_velo"], "league_typical")
    ann = manual[manual["action"] == "annotate"].copy()
    if len(ann):
        ann["note"] = ann["season"].astype(str) + ": " + ann["injury_type"]
        notes = ann.groupby("pitcher_name")["note"].agg("; ".join)
        comb["injury_notes"] = comb["player_name"].map(notes).fillna("")
    comb.to_csv(out / "starter_fatigue_profiles_2025_2026.csv", encoding="utf-8")
    res["application"] = {"n_starters": int(len(comb)), "league_mean_drop": float(lg),
                          "early_fade_cut": float(q20), "holds_velo_cut": float(q80),
                          "per_start_reliability": float(rel1)}

    # profile curves in 2026 (out of sample for groups defined on 2025 only)
    prof25 = prof[(prof["season"] == PREV) & prof["velo_ok"].fillna(False).astype(bool)].copy()
    a25, b25 = prof25["velo_drop"].quantile([0.2, 0.8])
    prof25["group_2025"] = np.select([prof25["velo_drop"] <= a25, prof25["velo_drop"] >= b25],
                                     ["early_fade", "holds_velo"], "league_typical")
    s26 = sp_keep[sp_keep["game_year"] == NEXT]
    fb26 = ht.primary_fastball(s26).rename_axis(["pitcher", "game_year"]).reset_index()
    d = s26.merge(fb26, on=["pitcher", "game_year"])
    d = d[(d["pitch_type"] == d["fb_type"]) & d["release_speed"].notna()]
    reach = d.groupby(["game_pk", "pitcher"])["pitch_count"].transform("max")
    d = d[reach >= ht.LATE[0]]
    base = d[d["pitch_count"].between(*ht.EARLY)].groupby(["game_pk", "pitcher"])["release_speed"].mean()
    d = d.join(base.rename("base"), on=["game_pk", "pitcher"]).dropna(subset=["base"])
    d["rel"] = d["release_speed"] - d["base"]
    d["bin"] = pd.cut(d["pitch_count"], ht.PITCH_BINS[:-1], labels=ht.PITCH_BIN_LABELS[:-1])
    d = d.merge(prof25[["pitcher", "group_2025"]], on="pitcher")
    ps = d.groupby(["group_2025", "pitcher", "bin"], observed=True)["rel"].mean().reset_index()
    gc = ps.groupby(["group_2025", "bin"], observed=True)["rel"].agg(["mean", "sem", "size"]).reset_index()
    gc.columns = ["group_2025", "pitch_bin", "velo_vs_first25_2026", "sem", "n_pitchers"]
    gc.to_csv(out / "profile_curves_2026_by_2025_group.csv", index=False)
    late = gc[gc["pitch_bin"] == "76-100"].set_index("group_2025")["velo_vs_first25_2026"]
    res["profile_groups_out_of_sample"] = {
        "groups_defined_on": "2025 velo_drop (bottom/top 20%)",
        "late_velo_2026": late.to_dict(),
        "gap_fade_vs_hold_mph": float(late["holds_velo"] - late["early_fade"]),
        "n_pitchers": gc[gc["pitch_bin"] == "76-100"].set_index("group_2025")["n_pitchers"].to_dict()}

    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    if verbose:
        print(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    main()
