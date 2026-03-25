"""
Module 5: Pricing Engine
========================
Computes reinsurance pricing metrics and produces a comparative analysis
across different program structures.

Key metrics:
- Expected Loss (EL): mean annual ceded loss
- Standard Deviation of ceded loss
- Rate on Line (ROL): EL / limit
- Technical Premium: EL × loading factor
- Tail-risk compression: VaR reduction at 99th and 99.5th percentiles
- Cost-efficiency ratio: premium / VaR reduction
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .reinsurance_structures import (
    ReinsuranceProgram,
    apply_program,
    ExcessOfLoss,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Core Pricing Functions
# =============================================================================

def compute_expected_loss(ceded_losses: np.ndarray) -> float:
    """Mean annual ceded (reinsurer) loss."""
    return float(np.mean(ceded_losses))


def compute_loss_std(ceded_losses: np.ndarray) -> float:
    """Standard deviation of annual ceded loss."""
    return float(np.std(ceded_losses))


def compute_rol(expected_loss: float, limit: float) -> float:
    """
    Rate on Line = Expected Loss / Limit.

    For Quota Share programs where 'limit' is conceptually the QS share
    of total portfolio, this is adjusted accordingly.
    """
    if limit <= 0:
        return 0.0
    return expected_loss / limit


def compute_technical_premium(
    expected_loss: float,
    loading_factor: float = 1.35,
    expense_ratio: float = 0.05,
) -> float:
    """
    Technical Premium = EL × loading_factor × (1 + expense_ratio).

    Parameters
    ----------
    expected_loss : float
        Expected annual ceded loss.
    loading_factor : float
        Risk loading multiplier (e.g. 1.35 = 35% loading).
    expense_ratio : float
        Additional expense loading (e.g. 0.05 = 5%).
    """
    return expected_loss * loading_factor * (1 + expense_ratio)


def compute_var(losses: np.ndarray, percentile: float = 99.0) -> float:
    """Value at Risk at the given percentile."""
    return float(np.percentile(losses, percentile))


def compute_tvar(losses: np.ndarray, percentile: float = 99.0) -> float:
    """Tail Value at Risk (expected shortfall) above the percentile."""
    threshold = np.percentile(losses, percentile)
    tail_losses = losses[losses >= threshold]
    if len(tail_losses) == 0:
        return float(threshold)
    return float(np.mean(tail_losses))


# =============================================================================
# Program Limit Calculation
# =============================================================================

def get_program_total_limit(program: ReinsuranceProgram, total_tsi: float = 0) -> float:
    """
    Calculate the effective total limit of a reinsurance program.

    For XoL layers, this is the sum of individual layer limits.
    For QS, this is the cession_pct × total_tsi (if provided).
    """
    total_limit = 0.0
    for layer in program.layers:
        if isinstance(layer, ExcessOfLoss):
            total_limit += layer.limit
        elif hasattr(layer, "cession_pct"):
            # For QS, use the proportion of total exposure as the 'limit'
            total_limit += layer.cession_pct * total_tsi if total_tsi > 0 else 0
    return total_limit


# =============================================================================
# Structure Comparison
# =============================================================================

def compare_structures(
    programs: dict[str, ReinsuranceProgram],
    annual_losses: pd.DataFrame,
    config: dict,
    loss_col: str = "total_annual_loss",
    total_tsi: float = 0,
) -> pd.DataFrame:
    """
    Compare multiple reinsurance programs on key pricing metrics.

    Parameters
    ----------
    programs : dict[str, ReinsuranceProgram]
        Map of program key → ReinsuranceProgram.
    annual_losses : DataFrame
        Combined annual loss catalog.
    config : dict
        Configuration dict (for pricing parameters).
    loss_col : str
        Column name for annual losses.
    total_tsi : float
        Total Sum Insured (for QS ROL calculation).

    Returns
    -------
    DataFrame with one row per program, columns:
        program_name, expected_loss, std_dev, var_99, var_995, tvar_99,
        program_limit, rol, technical_premium,
        gross_var_99, net_var_99, var_reduction_99, var_reduction_995,
        cost_efficiency_ratio
    """
    pricing_cfg = config.get("pricing", {})
    loading = pricing_cfg.get("loading_factor", 1.35)
    expense = pricing_cfg.get("expense_ratio", 0.05)

    gross_losses = annual_losses[loss_col].values
    gross_var_99 = compute_var(gross_losses, 99.0)
    gross_var_995 = compute_var(gross_losses, 99.5)
    gross_tvar_99 = compute_tvar(gross_losses, 99.0)

    rows = []
    for key, program in programs.items():
        result_df = apply_program(annual_losses, program, loss_col)

        ceded = result_df["ceded_loss"].values
        net = result_df["net_loss"].values

        el = compute_expected_loss(ceded)
        std = compute_loss_std(ceded)
        net_var_99 = compute_var(net, 99.0)
        net_var_995 = compute_var(net, 99.5)
        net_tvar_99 = compute_tvar(net, 99.0)

        limit = get_program_total_limit(program, total_tsi)
        rol = compute_rol(el, limit) if limit > 0 else None
        premium = compute_technical_premium(el, loading, expense)

        # Tail-risk compression: how much VaR is reduced
        var_reduction_99 = gross_var_99 - net_var_99
        var_reduction_995 = gross_var_995 - net_var_995

        # Cost-efficiency: premium per unit of VaR reduction
        cost_eff = premium / var_reduction_99 if var_reduction_99 > 0 else float("inf")

        rows.append({
            "program_key": key,
            "program_name": program.name,
            "expected_loss_bn": round(el, 2),
            "std_dev_bn": round(std, 2),
            "gross_var_99_bn": round(gross_var_99, 2),
            "net_var_99_bn": round(net_var_99, 2),
            "var_reduction_99_bn": round(var_reduction_99, 2),
            "gross_var_995_bn": round(gross_var_995, 2),
            "net_var_995_bn": round(net_var_995, 2),
            "var_reduction_995_bn": round(var_reduction_995, 2),
            "gross_tvar_99_bn": round(gross_tvar_99, 2),
            "net_tvar_99_bn": round(net_tvar_99, 2),
            "tvar_reduction_99_bn": round(gross_tvar_99 - net_tvar_99, 2),
            "program_limit_bn": round(limit, 2),
            "rol_pct": round(rol * 100, 2) if rol else None,
            "technical_premium_bn": round(premium, 2),
            "cost_efficiency_ratio": round(cost_eff, 3),
        })

    comparison = pd.DataFrame(rows)
    logger.info("Structure comparison complete: %d programs analysed", len(rows))
    return comparison


# =============================================================================
# Broker Advisory Memo Generator
# =============================================================================

def generate_advisory_memo(
    comparison_df: pd.DataFrame,
    annual_losses: pd.DataFrame,
    loss_col: str = "total_annual_loss",
) -> str:
    """
    Generate a broker-style advisory memo analysing reinsurance options.

    This produces the kind of narrative a reinsurance broker would include
    in a placement recommendation to a cedent — focusing on tail-risk
    compression efficiency, cost-benefit trade-offs, and a concrete
    recommendation.
    """
    gross = annual_losses[loss_col].values
    mean_gross = float(np.mean(gross))
    gross_99 = compute_var(gross, 99.0)
    gross_995 = compute_var(gross, 99.5)

    memo = []
    memo.append("# Reinsurance Placement Advisory Memo")
    memo.append("")
    memo.append("## Portfolio Loss Profile")
    memo.append("")
    memo.append(
        f"Based on {len(gross):,} simulated years of combined earthquake and typhoon "
        f"losses against the hypothetical Japanese property portfolio:"
    )
    memo.append("")
    memo.append(f"- **Mean Annual Loss**: ¥{mean_gross:.2f}B")
    memo.append(f"- **99th Percentile (1-in-100 year)**: ¥{gross_99:.2f}B")
    memo.append(f"- **99.5th Percentile (1-in-200 year)**: ¥{gross_995:.2f}B")
    memo.append("")
    memo.append("---")
    memo.append("")
    memo.append("## Structure-by-Structure Analysis")
    memo.append("")

    # Sort by cost efficiency (best = lowest ratio)
    sorted_df = comparison_df.sort_values("cost_efficiency_ratio")

    for _, row in sorted_df.iterrows():
        memo.append(f"### {row['program_name']}")
        memo.append("")
        memo.append(f"| Metric | Value |")
        memo.append(f"|--------|-------|")
        memo.append(f"| Expected Ceded Loss | ¥{row['expected_loss_bn']:.2f}B |")
        memo.append(f"| Technical Premium | ¥{row['technical_premium_bn']:.2f}B |")
        if row.get("rol_pct"):
            memo.append(f"| Rate on Line | {row['rol_pct']:.2f}% |")
        memo.append(f"| 99th %ile VaR Reduction | ¥{row['var_reduction_99_bn']:.2f}B |")
        memo.append(f"| 99.5th %ile VaR Reduction | ¥{row['var_reduction_995_bn']:.2f}B |")
        memo.append(f"| Cost-Efficiency Ratio | {row['cost_efficiency_ratio']:.3f} |")
        memo.append("")

    # Recommendation
    memo.append("---")
    memo.append("")
    memo.append("## Recommendation")
    memo.append("")

    best = sorted_df.iloc[0]
    worst = sorted_df.iloc[-1]

    # Find best XoL and best QS for comparison
    xol_rows = sorted_df[sorted_df["program_name"].str.contains("XoL|Cat", case=False)]
    qs_rows = sorted_df[sorted_df["program_name"].str.contains("Quota", case=False)]

    if not xol_rows.empty and not qs_rows.empty:
        best_xol = xol_rows.iloc[0]
        best_qs = qs_rows.iloc[0]

        if best_xol["var_reduction_99_bn"] > 0 and best_qs["var_reduction_99_bn"] > 0:
            xol_efficiency = best_xol["cost_efficiency_ratio"]
            qs_efficiency = best_qs["cost_efficiency_ratio"]

            if xol_efficiency < qs_efficiency:
                ratio = qs_efficiency / xol_efficiency
                memo.append(
                    f"The **{best_xol['program_name']}** structure demonstrates "
                    f"**{ratio:.1f}x superior cost-efficiency** compared to "
                    f"**{best_qs['program_name']}** in terms of tail-risk "
                    f"compression per unit of premium."
                )
            else:
                ratio = xol_efficiency / qs_efficiency
                memo.append(
                    f"The **{best_qs['program_name']}** structure demonstrates "
                    f"**{ratio:.1f}x superior cost-efficiency** compared to "
                    f"**{best_xol['program_name']}** in terms of tail-risk "
                    f"compression per unit of premium."
                )
            memo.append("")

    # Main recommendation
    memo.append(
        f"Under the current loss distribution, **{best['program_name']}** "
        f"achieves the lowest cost-efficiency ratio ({best['cost_efficiency_ratio']:.3f}), "
        f"indicating it provides the most VaR reduction per premium dollar. "
        f"It reduces the 99th percentile loss from ¥{best['gross_var_99_bn']:.2f}B to "
        f"¥{best['net_var_99_bn']:.2f}B at a technical premium of "
        f"¥{best['technical_premium_bn']:.2f}B."
    )
    memo.append("")

    memo.append("### Strategic Considerations")
    memo.append("")
    memo.append(
        "- **If the cedent's priority is balance-sheet protection** against "
        "extreme scenarios (1-in-200 year events), an XoL-led structure with "
        "high attachment is recommended, as it provides targeted tail-risk "
        "transfer at a lower premium-to-benefit ratio."
    )
    memo.append(
        "- **If the cedent seeks earnings volatility smoothing** across all "
        "years (including attritional losses), a Quota Share or blended program "
        "provides broader but less capital-efficient protection."
    )
    memo.append(
        "- **Blended structures** (QS + XoL) offer a middle ground: the QS "
        "layer smooths frequency losses while the XoL layers protect against "
        "severity spikes. This is often the most pragmatic choice for Japanese "
        "cedents with significant earthquake exposure."
    )
    memo.append("")

    memo.append("---")
    memo.append("")
    memo.append(
        "*This analysis is based on Monte Carlo simulation using historical "
        "JMA data calibrations. Actual placement terms would depend on market "
        "conditions, cedent financials, and underwriting considerations.*"
    )

    return "\n".join(memo)
