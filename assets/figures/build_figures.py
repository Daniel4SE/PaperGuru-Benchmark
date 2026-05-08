"""
Build all data-driven figures for the PaperGuru-Benchmark README.

Outputs (written next to this file):
  - paperbench_bar.png         -- per-paper PaperBench reproduction scores
  - paperbench_topline.png     -- aggregate comparison vs human + baselines
  - surveybench_radar.png      -- 5-axis content radar
  - surveybench_richness.png   -- composite richness comparison
  - lift_distribution.png      -- histogram of per-paper lift over baselines
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#1a1a1d",
        "axes.linewidth": 1.0,
        "axes.labelcolor": "#1a1a1d",
        "xtick.color": "#1a1a1d",
        "ytick.color": "#1a1a1d",
        "text.color": "#1a1a1d",
    }
)

OUT = Path(__file__).parent
INK = "#0a0a0b"
BLUE = "#2563eb"
TEAL = "#0d9488"
RED = "#dc2626"
MUTED = "#9ca0a8"
GOLD = "#c9a14a"

# ---------- DATA (lifted verbatim from sections/experiments.tex) ----------

PAPERBENCH = [
    # (slug, paperguru, best_baseline)  None = no public baseline
    ("adaptive-pruning", 50.59, 33.26),
    ("all-in-one", 53.96, 49.47),
    ("bam", 84.72, 61.11),
    ("bbox", 40.34, 33.79),
    ("bridging-data-gaps", 57.14, 26.46),
    ("fre", 61.51, 35.21),
    ("ftrl", 62.66, 10.11),
    ("lbcs", 85.74, 30.10),
    ("lca-on-the-line", 63.16, 30.23),
    ("mechanistic-understanding", 70.19, 40.55),
    ("pinn", 54.29, 58.76),
    ("rice", 57.65, 10.87),
    ("robust-clip", 52.55, 28.66),
    ("sample-specific-masks", 86.52, 44.13),
    ("sapg", 46.49, 31.69),
    ("sequential-neural-score-estimation", 89.32, 64.94),
    ("stay-on-topic-with-classifier-free-guidance", 88.16, 20.13),
    ("stochastic-interpolants", 82.99, 42.10),
    ("test-time-model-adaptation", 70.06, 32.45),
    ("what-will-my-model-forget", 60.98, 30.82),
    ("semantic-self-consistency", 95.45, None),
    ("self-composing-policies", 65.03, None),
    ("self-expansion", 39.77, None),
]
HUMAN_REF = 41.00

PAPERBENCH_BASELINES = {
    "AiScientist + Gemini-3-Flash": 30.52,
    "PaperBench original baseline": 21.30,
    "IterativeAgent (best ext.)": 35.74,
    "Human Expert (48h ML-PhD)": 41.00,
    "PaperGuru (ours)": 66.05,
}

SURVEYBENCH = {
    "Coverage": {
        "PaperGuru": 94.0,
        "AutoSurvey": 61.0,
        "LLMxMR-v2": 70.0,
        "SurveyForge": 60.0,
        "ASur": 59.0,
    },
    "Coherence": {
        "PaperGuru": 87.4,
        "AutoSurvey": 80.0,
        "LLMxMR-v2": 78.0,
        "SurveyForge": 79.0,
        "ASur": 76.0,
    },
    "Depth": {
        "PaperGuru": 92.0,
        "AutoSurvey": 75.0,
        "LLMxMR-v2": 75.0,
        "SurveyForge": 72.0,
        "ASur": 60.0,
    },
    "Focus": {
        "PaperGuru": 100.0,
        "AutoSurvey": 99.0,
        "LLMxMR-v2": 94.0,
        "SurveyForge": 86.0,
        "ASur": 76.0,
    },
    "Fluency": {
        "PaperGuru": 100.0,
        "AutoSurvey": 88.0,
        "LLMxMR-v2": 80.0,
        "SurveyForge": 80.0,
        "ASur": 80.0,
    },
}

RICHNESS = {
    "ASur": 0.00,
    "SurveyForge": 0.00,
    "AutoSurvey": 6.24,
    "LLMxMR-v2": 20.36,
    "PaperGuru": 43.76,
}


# =================== FIGURE 1: PaperBench per-paper bar ===================
def fig_paperbench_bar() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=150)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    labels = [p[0] for p in PAPERBENCH]
    pg = np.array([p[1] for p in PAPERBENCH])
    base = np.array([p[2] if p[2] is not None else np.nan for p in PAPERBENCH])
    x = np.arange(len(labels))
    w = 0.42

    bars_pg = ax.bar(
        x - w / 2, pg, w, label="PaperGuru (ours)", color=INK, edgecolor=INK, zorder=3
    )
    bars_b = ax.bar(
        x + w / 2,
        np.nan_to_num(base),
        w,
        label="Best published baseline",
        color=MUTED,
        edgecolor=MUTED,
        alpha=0.85,
        zorder=3,
    )
    # mark missing
    for i, b in enumerate(base):
        if np.isnan(b):
            ax.text(
                x[i] + w / 2,
                2,
                "no\nbaseline",
                ha="center",
                va="bottom",
                fontsize=7,
                color=MUTED,
                style="italic",
            )

    # human-expert reference line
    ax.axhline(HUMAN_REF, color=RED, ls="--", lw=1.3, alpha=0.85, zorder=2)
    ax.text(
        len(labels) - 0.3,
        HUMAN_REF + 1.8,
        f"  Human ML-PhD (48h) = {HUMAN_REF:.0f}%",
        color=RED,
        fontsize=10,
        ha="right",
        va="bottom",
        style="italic",
        weight="600",
    )

    # PaperGuru mean
    mean_pg = float(np.mean(pg))
    ax.axhline(mean_pg, color=INK, ls=":", lw=1.0, alpha=0.55, zorder=2)
    ax.text(
        0.2,
        mean_pg + 1.5,
        f"PaperGuru mean = {mean_pg:.2f}%",
        color=INK,
        fontsize=9,
        ha="left",
        va="bottom",
        weight="600",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Reproduction score (%)", fontsize=11, weight="600")
    ax.set_ylim(0, 105)
    ax.set_title(
        "PaperBench: per-paper reproduction score across all 23 papers",
        fontsize=13,
        weight="700",
        color=INK,
        pad=14,
        loc="left",
    )
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#eaeaee", lw=0.8, zorder=1)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(
        OUT / "paperbench_bar.png", dpi=170, bbox_inches="tight", facecolor="white"
    )
    plt.close()
    print("  + paperbench_bar.png")


# =================== FIGURE 2: aggregate top-line ===================
def fig_paperbench_topline() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    items = list(PAPERBENCH_BASELINES.items())
    items.sort(key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [MUTED if "PaperGuru" not in k else INK for k in labels]
    colors = [RED if "Human" in k else c for c, k in zip(colors, labels)]

    bars = ax.barh(labels, vals, color=colors, edgecolor=colors, height=0.6, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(
            v + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}%",
            va="center",
            fontsize=10,
            weight="600",
            color=INK,
        )

    # arrow showing lift
    pg_y = labels.index("PaperGuru (ours)")
    best_baseline_y = labels.index("IterativeAgent (best ext.)")
    ax.annotate(
        "",
        xy=(66.05, pg_y - 0.25),
        xytext=(35.74, pg_y - 0.25),
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=2),
    )
    ax.text(
        50.9,
        pg_y - 0.5,
        "+30.21% absolute lift",
        color=TEAL,
        fontsize=11,
        ha="center",
        weight="700",
    )

    ax.set_xlim(0, 80)
    ax.set_xlabel("PaperBench score (%)", fontsize=11, weight="600")
    ax.set_title(
        "PaperBench: PaperGuru beats every published baseline — and the human bar",
        fontsize=13,
        weight="700",
        color=INK,
        pad=14,
        loc="left",
    )
    ax.grid(axis="x", color="#eaeaee", lw=0.8, zorder=1)
    plt.tight_layout()
    plt.savefig(
        OUT / "paperbench_topline.png", dpi=170, bbox_inches="tight", facecolor="white"
    )
    plt.close()
    print("  + paperbench_topline.png")


# =================== FIGURE 3: SurveyBench radar ===================
def fig_surveybench_radar() -> None:
    metrics = list(SURVEYBENCH.keys())
    systems = list(SURVEYBENCH[metrics[0]].keys())
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(8.5, 7.5), dpi=150, subplot_kw=dict(projection="polar")
    )
    fig.patch.set_facecolor("white")

    palette = {
        "PaperGuru": INK,
        "AutoSurvey": "#94a3b8",
        "LLMxMR-v2": "#cbd5e1",
        "SurveyForge": "#e2e8f0",
        "ASur": "#f1f5f9",
    }

    # Plot baselines first (background), then PaperGuru on top
    order = ["ASur", "SurveyForge", "LLMxMR-v2", "AutoSurvey", "PaperGuru"]
    for sys in order:
        vals = [SURVEYBENCH[m][sys] for m in metrics]
        vals += vals[:1]
        is_pg = sys == "PaperGuru"
        ax.plot(
            angles,
            vals,
            color=palette[sys],
            lw=2.6 if is_pg else 1.4,
            label=sys,
            zorder=4 if is_pg else 2,
        )
        ax.fill(
            angles,
            vals,
            color=palette[sys],
            alpha=0.25 if is_pg else 0.10,
            zorder=3 if is_pg else 1,
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, weight="600")
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color=MUTED, fontsize=8)
    ax.grid(color="#eaeaee", lw=0.8)
    ax.spines["polar"].set_color("#eaeaee")

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.32, 1.10), frameon=False, fontsize=10
    )
    ax.set_title(
        "SurveyBench: content quality across five dimensions",
        fontsize=13,
        weight="700",
        color=INK,
        pad=24,
        loc="left",
    )
    plt.tight_layout()
    plt.savefig(
        OUT / "surveybench_radar.png", dpi=170, bbox_inches="tight", facecolor="white"
    )
    plt.close()
    print("  + surveybench_radar.png")


# =================== FIGURE 4: composite richness ===================
def fig_surveybench_richness() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    items = list(RICHNESS.items())
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [INK if "PaperGuru" in k else MUTED for k in labels]
    bars = ax.barh(labels, vals, color=colors, edgecolor=colors, height=0.55, zorder=3)
    for bar, v in zip(bars, vals):
        txt = "0.00 — no figures generated" if v == 0 else f"{v:.2f}%"
        col = MUTED if v == 0 else INK
        ax.text(
            v + 0.7 if v > 0 else 0.7,
            bar.get_y() + bar.get_height() / 2,
            txt,
            va="center",
            fontsize=10,
            weight="600",
            color=col,
        )

    ax.set_xlim(0, 55)
    ax.set_xlabel("Composite richness score (%)", fontsize=11, weight="600")
    ax.set_title(
        "SurveyBench: only PaperGuru produces evidence-rich figures, tables, and code",
        fontsize=12,
        weight="700",
        color=INK,
        pad=12,
        loc="left",
    )
    ax.grid(axis="x", color="#eaeaee", lw=0.8, zorder=1)
    plt.tight_layout()
    plt.savefig(
        OUT / "surveybench_richness.png",
        dpi=170,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("  + surveybench_richness.png")


# =================== FIGURE 5: per-paper lift histogram ===================
def fig_lift_distribution() -> None:
    lifts = [(p[1] - p[2]) for p in PAPERBENCH if p[2] is not None]
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bins = np.arange(-10, 75, 5)
    ax.hist(lifts, bins=bins, color=INK, edgecolor="white", lw=1.5, zorder=3)
    ax.axvline(0, color=RED, ls="--", lw=1.2, zorder=2)
    ax.text(
        0.5,
        ax.get_ylim()[1] * 0.92,
        "← worse",
        color=RED,
        fontsize=9,
        ha="left",
        weight="600",
    )
    ax.text(
        -0.5,
        ax.get_ylim()[1] * 0.92,
        "better →",
        color=TEAL,
        fontsize=9,
        ha="right",
        weight="600",
    )
    ax.set_xlabel(
        "Lift over best published baseline (% points)", fontsize=11, weight="600"
    )
    ax.set_ylabel("Number of papers", fontsize=11, weight="600")
    mean_lift = float(np.mean(lifts))
    ax.axvline(mean_lift, color=TEAL, ls=":", lw=1.5, zorder=2)
    ax.text(
        mean_lift + 1,
        ax.get_ylim()[1] * 0.6,
        f"mean lift = +{mean_lift:.2f}%",
        color=TEAL,
        fontsize=10,
        weight="700",
    )
    ax.set_title(
        f"PaperBench: 19 of 20 papers improve — mean lift +{mean_lift:.2f}%",
        fontsize=12,
        weight="700",
        color=INK,
        pad=12,
        loc="left",
    )
    ax.grid(axis="y", color="#eaeaee", lw=0.8, zorder=1)
    plt.tight_layout()
    plt.savefig(
        OUT / "lift_distribution.png", dpi=170, bbox_inches="tight", facecolor="white"
    )
    plt.close()
    print("  + lift_distribution.png")


def main() -> None:
    print(f"Building data figures into {OUT}/\n")
    fig_paperbench_bar()
    fig_paperbench_topline()
    fig_surveybench_radar()
    fig_surveybench_richness()
    fig_lift_distribution()
    # also dump the raw data as JSON so others can rebuild
    with open(OUT / "data.json", "w") as f:
        json.dump(
            {
                "paperbench_per_paper": [
                    {"slug": p[0], "paperguru": p[1], "best_baseline": p[2]}
                    for p in PAPERBENCH
                ],
                "paperbench_baselines": PAPERBENCH_BASELINES,
                "surveybench": SURVEYBENCH,
                "surveybench_richness": RICHNESS,
                "human_reference": HUMAN_REF,
            },
            f,
            indent=2,
        )
    print("  + data.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
