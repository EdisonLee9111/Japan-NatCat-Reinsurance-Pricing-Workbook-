"""
Module 6: Visualization
=======================
Produces publication-quality charts for the reinsurance analysis.

Outputs saved to outputs/figures/ and committed to the repo so they
render directly in the GitHub README.

Charts:
1. Loss Exceedance Curve (OEP/AEP)
2. Layer Loss Allocation (stacked bar)
3. Structure Comparison (grouped bar)
4. Vulnerability Curves
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from .reinsurance_structures import ReinsuranceProgram, apply_program

logger = logging.getLogger(__name__)

# --- Global style ---
COLORS = {
    "cedent": "#E74C3C",
    "reinsurer": "#3498DB",
    "gross": "#2C3E50",
    "net": "#27AE60",
    "eq": "#E67E22",
    "ty": "#8E44AD",
}

PROGRAM_PALETTE = sns.color_palette("Set2", 8)


def _setup_style():
    """Apply consistent chart styling."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


# =============================================================================
# 1. Loss Exceedance Curve
# =============================================================================

def plot_loss_exceedance_curve(
    annual_losses: pd.DataFrame,
    loss_col: str = "total_annual_loss",
    programs: dict[str, ReinsuranceProgram] | None = None,
    output_dir: str = "outputs/figures",
    filename: str = "loss_exceedance_curve.png",
    dpi: int = 150,
) -> str:
    """
    Plot OEP (Occurrence Exceedance Probability) curve.

    Shows the probability of exceeding a given loss level both gross and
    net of each reinsurance program.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Gross curve
    gross = np.sort(annual_losses[loss_col].values)
    n = len(gross)
    exceedance_prob = 1.0 - np.arange(1, n + 1) / (n + 1)
    return_period = 1.0 / exceedance_prob

    ax.plot(
        return_period, gross,
        color=COLORS["gross"], linewidth=2.5, label="Gross Loss",
        zorder=5
    )

    # Net curves for each program
    if programs:
        for i, (key, program) in enumerate(programs.items()):
            result = apply_program(annual_losses, program, loss_col)
            net = np.sort(result["net_loss"].values)
            ax.plot(
                return_period, net,
                color=PROGRAM_PALETTE[i % len(PROGRAM_PALETTE)],
                linewidth=1.8, linestyle="--",
                label=f"Net: {program.name}",
                alpha=0.85,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel("Annual Aggregate Loss (¥ Billion)")
    ax.set_title("Loss Exceedance Curve — Gross vs Net of Reinsurance")

    # Add reference lines
    for rp, label in [(100, "1-in-100"), (200, "1-in-200"), (250, "1-in-250")]:
        if rp <= return_period.max():
            ax.axvline(rp, color="gray", linestyle=":", alpha=0.5)
            ax.text(rp * 1.05, ax.get_ylim()[1] * 0.95, label,
                    fontsize=8, color="gray", rotation=90, va="top")

    ax.legend(loc="upper left", framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}B"))

    plt.tight_layout()
    out_path = _save_figure(fig, output_dir, filename, dpi)
    plt.close(fig)
    return out_path


# =============================================================================
# 2. Layer Loss Allocation
# =============================================================================

def plot_layer_loss_allocation(
    annual_losses: pd.DataFrame,
    programs: dict[str, ReinsuranceProgram],
    loss_col: str = "total_annual_loss",
    output_dir: str = "outputs/figures",
    filename: str = "layer_loss_allocation.png",
    dpi: int = 150,
) -> str:
    """
    Stacked bar chart showing mean loss split between cedent and reinsurer
    for each program.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    names = []
    cedent_means = []
    reinsurer_means = []

    for key, program in programs.items():
        result = apply_program(annual_losses, program, loss_col)
        names.append(program.name)
        cedent_means.append(result["net_loss"].mean())
        reinsurer_means.append(result["ceded_loss"].mean())

    x = np.arange(len(names))
    width = 0.5

    bars_cedent = ax.bar(
        x, cedent_means, width,
        label="Cedent Retained", color=COLORS["cedent"], alpha=0.85
    )
    bars_reinsurer = ax.bar(
        x, reinsurer_means, width, bottom=cedent_means,
        label="Reinsurer Ceded", color=COLORS["reinsurer"], alpha=0.85
    )

    # Add value labels
    for bar_c, bar_r, c_val, r_val in zip(bars_cedent, bars_reinsurer, cedent_means, reinsurer_means):
        ax.text(
            bar_c.get_x() + bar_c.get_width() / 2, c_val / 2,
            f"¥{c_val:.1f}B", ha="center", va="center", fontsize=9,
            fontweight="bold", color="white"
        )
        ax.text(
            bar_r.get_x() + bar_r.get_width() / 2, c_val + r_val / 2,
            f"¥{r_val:.1f}B", ha="center", va="center", fontsize=9,
            fontweight="bold", color="white"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Mean Annual Loss (¥ Billion)")
    ax.set_title("Loss Allocation: Cedent vs Reinsurer by Program")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"¥{v:,.0f}B"))

    plt.tight_layout()
    out_path = _save_figure(fig, output_dir, filename, dpi)
    plt.close(fig)
    return out_path


# =============================================================================
# 3. Structure Comparison
# =============================================================================

def plot_structure_comparison(
    comparison_df: pd.DataFrame,
    output_dir: str = "outputs/figures",
    filename: str = "structure_comparison.png",
    dpi: int = 150,
) -> str:
    """
    Grouped bar chart comparing key metrics across programs:
    - Technical Premium
    - VaR Reduction (99th)
    - Cost-Efficiency Ratio
    """
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    programs = comparison_df["program_name"].values
    x = np.arange(len(programs))
    width = 0.5

    # Panel 1: Technical Premium
    ax = axes[0]
    bars = ax.bar(x, comparison_df["technical_premium_bn"], width,
                  color=PROGRAM_PALETTE[:len(programs)], alpha=0.85)
    ax.set_title("Technical Premium")
    ax.set_ylabel("¥ Billion")
    ax.set_xticks(x)
    ax.set_xticklabels(programs, rotation=25, ha="right", fontsize=8)
    for bar, val in zip(bars, comparison_df["technical_premium_bn"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                f"¥{val:.1f}B", ha="center", va="bottom", fontsize=8)

    # Panel 2: VaR Reduction at 99th percentile
    ax = axes[1]
    bars = ax.bar(x, comparison_df["var_reduction_99_bn"], width,
                  color=PROGRAM_PALETTE[:len(programs)], alpha=0.85)
    ax.set_title("99th %ile VaR Reduction")
    ax.set_ylabel("¥ Billion")
    ax.set_xticks(x)
    ax.set_xticklabels(programs, rotation=25, ha="right", fontsize=8)
    for bar, val in zip(bars, comparison_df["var_reduction_99_bn"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                f"¥{val:.1f}B", ha="center", va="bottom", fontsize=8)

    # Panel 3: Cost-Efficiency Ratio
    ax = axes[2]
    ce = comparison_df["cost_efficiency_ratio"].values
    ce_clipped = np.clip(ce, 0, np.percentile(ce[np.isfinite(ce)], 95) * 1.5
                         if np.any(np.isfinite(ce)) else 10)
    bars = ax.bar(x, ce_clipped, width,
                  color=PROGRAM_PALETTE[:len(programs)], alpha=0.85)
    ax.set_title("Cost-Efficiency Ratio\n(lower = better)")
    ax.set_ylabel("Premium / VaR Reduction")
    ax.set_xticks(x)
    ax.set_xticklabels(programs, rotation=25, ha="right", fontsize=8)
    for bar, val in zip(bars, ce):
        label = f"{val:.2f}" if np.isfinite(val) else "∞"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                label, ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Reinsurance Structure Comparison",
        fontsize=15, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    out_path = _save_figure(fig, output_dir, filename, dpi)
    plt.close(fig)
    return out_path


# =============================================================================
# 4. Vulnerability Curves (bonus visualization)
# =============================================================================

def plot_vulnerability_curves(
    config: dict,
    output_dir: str = "outputs/figures",
    filename: str = "vulnerability_curves.png",
    dpi: int = 150,
) -> str:
    """Plot the earthquake and typhoon vulnerability curves side by side."""
    from .loss_model import sigmoid_vulnerability

    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    loss_cfg = config.get("loss", {})

    # Earthquake
    eq_params = loss_cfg.get("earthquake_vulnerability", {"k": 1.5, "x0": 6.0, "cap": 0.35})
    intensities = np.linspace(4, 7.5, 200)
    dr = [sigmoid_vulnerability(i, **eq_params) for i in intensities]
    ax1.plot(intensities, dr, color=COLORS["eq"], linewidth=2.5)
    ax1.fill_between(intensities, dr, alpha=0.15, color=COLORS["eq"])
    ax1.set_xlabel("JMA Seismic Intensity")
    ax1.set_ylabel("Damage Ratio")
    ax1.set_title("Earthquake Vulnerability Curve")
    ax1.set_ylim(0, 0.4)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Typhoon
    ty_params = loss_cfg.get("typhoon_vulnerability", {"k": 0.08, "x0": 75.0, "cap": 0.25})
    winds = np.linspace(30, 140, 200)
    dr = [sigmoid_vulnerability(w, **ty_params) for w in winds]
    ax2.plot(winds, dr, color=COLORS["ty"], linewidth=2.5)
    ax2.fill_between(winds, dr, alpha=0.15, color=COLORS["ty"])
    ax2.set_xlabel("Max Sustained Wind (kt)")
    ax2.set_ylabel("Damage Ratio")
    ax2.set_title("Typhoon Vulnerability Curve")
    ax2.set_ylim(0, 0.3)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()
    out_path = _save_figure(fig, output_dir, filename, dpi)
    plt.close(fig)
    return out_path


# =============================================================================
# Helpers
# =============================================================================

def _save_figure(fig, output_dir: str, filename: str, dpi: int) -> str:
    """Save figure and return the output path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    logger.info("Saved figure: %s", path)
    return path
