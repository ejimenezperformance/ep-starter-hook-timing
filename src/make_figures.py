"""EP-branded figures for the README. / Gráficos con la marca EP."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"

NAVY_TEXT = "#0B1B33"   # EP navy (text, titles)
NAVY = "#2A4A7F"        # mark navy (lighter step for data marks)
GOLD = "#B07F1A"        # mark gold (darker step: >=3:1 on off-white)
BG = "#F5F3EC"          # EP off-white
GRID = "#D9D5C9"
MUTED = "#5B6472"

plt.rcParams.update({
    "font.family": ["Poppins", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": NAVY_TEXT, "text.color": NAVY_TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": BG,
    "figure.facecolor": BG, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})


def stability_figure(res: dict) -> None:
    v = pd.read_csv(OUT / "H1_velo_drop_pairs.csv")
    t = pd.read_csv(OUT / "H2_tto3_penalty_pairs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    panels = [
        (axes[0], v, "velo_drop", NAVY, res["H1_velo_drop"],
         "Physical signal: late fastball velo drop (mph)\nSeñal física: caída de velocidad tardía"),
        (axes[1], t, "tto3_penalty", GOLD, res["H2_tto3_penalty"],
         "Results signal: 3rd-time-through xwOBA penalty\nSeñal de resultados: penalización 3ª vuelta"),
    ]
    for ax, d, col, color, r, title in panels:
        x, y = d[f"{col}_prev"], d[f"{col}_next"]
        ax.scatter(x, y, s=28, color=color, alpha=0.7, edgecolor=BG, linewidth=1.0)
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color=NAVY_TEXT, lw=2)
        ax.set_xlabel("Season N / Temporada N")
        ax.set_ylabel("Season N+1 / Temporada N+1")
        ax.set_title(title, fontsize=10.5, loc="left", fontweight="bold")
        verdict = "CONFIRMED: repeats" if r["confirmed"] else "Not confirmed: does not repeat"
        ax.text(0.03, 0.97,
                f"r = {r['r_pooled']:.2f}  (95% CI {r['ci95'][0]:.2f} to {r['ci95'][1]:.2f})\n"
                f"{r['n_pairs']} season pairs · {r['n_pitchers']} starters\n{verdict}",
                transform=ax.transAxes, va="top", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))
    fig.suptitle("Which fatigue signal repeats year to year? / ¿Qué señal de fatiga se repite de un año a otro?",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.005, "Emerson Performance · Statcast 2023–2026 regular season · pairs 2023→24, 2024→25, "
             "2025→26 · each dot = one starter in one pair", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(FIG / "fig1_stability.png", dpi=180)
    plt.close(fig)


def curve_figure(res: dict) -> None:
    gc = pd.read_csv(OUT / "profile_curves_out_of_sample.csv")
    info = res["profile_groups_out_of_sample"]
    py, ny = info["shown_in"] - 1, info["shown_in"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    styles = {"early_fade": (GOLD, "-", f"Early fade ({py} bottom 20%)"),
              "league_typical": (MUTED, "--", "League typical"),
              "holds_velo": (NAVY, "-", f"Holds velo ({py} top 20%)")}
    labels = ["1-25", "26-50", "51-75", "76-100"]
    for g, (color, ls, name) in styles.items():
        d = gc[gc["group_prev"] == g].set_index("pitch_bin").reindex(labels)
        x = np.arange(len(labels))
        ax.plot(x, d["velo_vs_first25_next"], color=color, lw=2, ls=ls, marker="o", ms=8,
                markeredgecolor=BG, markeredgewidth=2)
        n = int(d["n_pitchers"].iloc[-1])
        ax.text(x[-1] + 0.1, d["velo_vs_first25_next"].iloc[-1],
                f"{name}\n{d['velo_vs_first25_next'].iloc[-1]:+.2f} mph · n={n}",
                color=NAVY_TEXT, va="center", fontsize=9)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.3, len(labels) + 0.9)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("Pitch count in the start / Conteo de lanzamientos")
    ax.set_ylabel(f"{ny} fastball velo vs. first 25 pitches (mph)")
    gap = res["profile_groups_out_of_sample"]["gap_fade_vs_hold_mph"]
    ax.set_title(f"Groups set on {py} still separate in {ny}: {gap:.1f} mph gap late in the start\n"
                 f"Grupos de {py} se siguen separando en {ny}: {gap:.1f} mph de diferencia",
                 loc="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.01, f"Emerson Performance · Statcast {ny}, starts reaching 76+ pitches · "
             f"out of sample: groups defined only with {py} data", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig2_profile_curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    stability_figure(res)
    curve_figure(res)
    print("figures written")
