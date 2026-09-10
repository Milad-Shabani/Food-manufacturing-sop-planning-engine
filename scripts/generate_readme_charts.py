#!/usr/bin/env python3
"""Regenerate the static chart PNGs used in README.md (docs/charts/).

These are plain matplotlib renders of the same data the live dashboard
uses, sized for GitHub's README preview (which can't run JavaScript, so
dashboard/index.html itself won't render there). Run after a fresh
pipeline run to refresh them:

    python -m sop_planning.cli run
    python scripts/generate_readme_charts.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

WAREHOUSE = Path("data/processed/planning_warehouse.db")
OUT_DIR = Path("docs/charts")

# Palette matches dashboard/index.html
BG, PANEL, LINE = "#26100a", "#2f150c", "#5c2e18"
ACCENT, ACCENT2, GREEN, BROWN, RISK = "#EB5B2A", "#F2994A", "#8FA377", "#B08872", "#D62839"
TEXT, TEXT_DIM = "#FBEFE6", "#D9B9A8"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "axes.edgecolor": LINE,
            "axes.labelcolor": TEXT_DIM,
            "text.color": TEXT,
            "xtick.color": TEXT_DIM,
            "ytick.color": TEXT_DIM,
            "grid.color": LINE,
            "font.size": 11,
        }
    )


def plot_fill_rate(weekly: pd.DataFrame, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    x = weekly["week_start"]
    colors = [RISK if fr < 1 else ACCENT for fr in weekly["fill_rate"]]
    ax1.bar(
        x - pd.Timedelta(days=1.5), weekly["forecast_units"], width=3, color=LINE, label="Forecast"
    )
    ax1.bar(
        x + pd.Timedelta(days=1.5),
        weekly["allocated_units"],
        width=3,
        color=colors,
        label="Planned",
    )
    ax1.set_ylabel("Units")
    ax1.grid(axis="y", alpha=0.4)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        weekly["fill_rate"] * 100,
        color=TEXT,
        marker="o",
        markersize=4,
        linewidth=1.8,
        label="Fill rate",
    )
    ax2.set_ylabel("Fill rate (%)")
    ax2.set_ylim(80, 105)
    ax2.spines[["top"]].set_visible(False)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate(rotation=30)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower left",
        frameon=False,
        labelcolor=TEXT_DIM,
        fontsize=9,
    )
    ax1.set_title(
        "Forecast vs. planned output — the Nowruz capacity crunch",
        color=TEXT,
        fontsize=13,
        loc="left",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ccc_trend(ccc: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(ccc["week_start"], ccc["dso"], color=ACCENT2, linewidth=1.4, label="DSO")
    ax.plot(ccc["week_start"], ccc["dpo"], color=GREEN, linewidth=1.4, label="DPO")
    ax.plot(ccc["week_start"], ccc["dio"], color=BROWN, linewidth=1.4, label="DIO")
    ax.plot(ccc["week_start"], ccc["ccc"], color=ACCENT, linewidth=2.8, label="CCC")
    ax.set_ylabel("Days")
    ax.grid(axis="y", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)
    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT_DIM, fontsize=9, ncol=4)
    avg = ccc["ccc"].tail(12).mean()
    ax.set_title(
        f"Cash conversion cycle — steady around ~{avg:.0f} days",
        color=TEXT,
        fontsize=13,
        loc="left",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not WAREHOUSE.exists():
        raise SystemExit(f"{WAREHOUSE} not found — run `python -m sop_planning.cli run` first.")

    _style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(WAREHOUSE)

    weekly = pd.read_sql("select * from fact_sop_weekly_summary order by week_start", conn)
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    plot_fill_rate(weekly, OUT_DIR / "fill_rate.png")

    ccc = pd.read_sql("select * from fact_working_capital order by week_start", conn)
    ccc["week_start"] = pd.to_datetime(ccc["week_start"])
    plot_ccc_trend(ccc, OUT_DIR / "ccc_trend.png")

    print(f"Wrote {OUT_DIR}/fill_rate.png and {OUT_DIR}/ccc_trend.png")


if __name__ == "__main__":
    main()
