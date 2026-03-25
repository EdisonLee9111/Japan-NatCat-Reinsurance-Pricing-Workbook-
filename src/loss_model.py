"""
Module 3: Loss Model
====================
Converts hazard intensities into financial losses using vulnerability curves
and an exposure portfolio.

Vulnerability curves map event intensity (magnitude, wind speed) to a damage
ratio (0-1), which is then multiplied by Total Sum Insured (TSI) to produce
ground-up losses.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Vulnerability Curves
# =============================================================================

def sigmoid_vulnerability(
    intensity: float,
    k: float = 1.5,
    x0: float = 6.0,
    cap: float = 0.35,
) -> float:
    """
    Sigmoid vulnerability curve: damage_ratio = cap / (1 + exp(-k*(x - x0)))

    Parameters
    ----------
    intensity : float
        Event intensity (JMA seismic intensity, wind speed in kt, etc.)
    k : float
        Steepness of the sigmoid curve.
    x0 : float
        Midpoint — intensity at which damage ratio = cap/2.
    cap : float
        Maximum damage ratio (asymptotic upper bound).

    Returns
    -------
    float
        Damage ratio in [0, cap].
    """
    return cap / (1.0 + np.exp(-k * (intensity - x0)))


def earthquake_damage_ratio(
    intensity: float,
    params: Optional[dict] = None,
) -> float:
    """
    Compute damage ratio for an earthquake event.

    Uses JMA seismic intensity as the input metric.
    Default calibration: significant damage begins around intensity 5.5-6.0,
    max damage ratio ~35% (typical for mixed commercial/residential portfolio).
    """
    if params is None:
        params = {"k": 1.5, "x0": 6.0, "cap": 0.35}
    return sigmoid_vulnerability(intensity, **params)


def typhoon_damage_ratio(
    wind_speed_kt: float,
    params: Optional[dict] = None,
) -> float:
    """
    Compute damage ratio for a typhoon event.

    Uses maximum sustained wind speed (knots) as the input metric.
    Default calibration: noticeable damage above ~60 kt, max ~25%.
    """
    if params is None:
        params = {"k": 0.08, "x0": 75.0, "cap": 0.25}
    return sigmoid_vulnerability(wind_speed_kt, **params)


# =============================================================================
# Exposure Portfolio
# =============================================================================

def load_exposure(path: str = "data/reference/japan_property_exposure.csv") -> pd.DataFrame:
    """Load the Japan property exposure portfolio CSV."""
    df = pd.read_csv(path, encoding="utf-8")
    logger.info(
        "Loaded exposure portfolio: %d regions, total TSI = ¥%.1fB",
        len(df), df["tsi_jpy_billion"].sum()
    )
    return df


# =============================================================================
# Loss Catalog Generation
# =============================================================================

def compute_event_loss(
    intensity: float,
    exposure_df: pd.DataFrame,
    peril: str,
    vuln_params: Optional[dict] = None,
) -> float:
    """
    Compute aggregate ground-up loss for a single event.

    Applies the vulnerability curve to the event intensity and multiplies
    by the total Sum Insured across all exposure regions.

    In a more sophisticated model, regional intensity attenuation would be
    applied. Here we use a simplified national-level approach with a
    randomised attenuation factor per region.
    """
    if peril == "earthquake":
        base_dr = earthquake_damage_ratio(intensity, vuln_params)
    elif peril == "typhoon":
        base_dr = typhoon_damage_ratio(intensity, vuln_params)
    else:
        raise ValueError(f"Unknown peril: {peril}")

    # Apply deductible and sum over regions
    deductible_col = "eq_deductible_pct" if peril == "earthquake" else "wind_deductible_pct"

    total_loss = 0.0
    for _, row in exposure_df.iterrows():
        # Gross damage ratio for this region
        dr = base_dr
        # Apply policy deductible
        deductible = row.get(deductible_col, 0)
        if pd.notna(deductible):
            dr = max(0, dr - deductible / 100.0)
        # Ground-up loss for this region (in ¥ billion)
        total_loss += dr * row["tsi_jpy_billion"]

    return total_loss


def generate_loss_catalog(
    hazard_simulations: list[list[float]],
    exposure_df: pd.DataFrame,
    peril: str,
    vuln_params: Optional[dict] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a year-by-year loss catalog from hazard simulations.

    Parameters
    ----------
    hazard_simulations : list of lists
        Output from HazardModel.simulate(). Each inner list has event
        intensities for one simulated year.
    exposure_df : DataFrame
        Exposure portfolio with tsi_jpy_billion column.
    peril : str
        "earthquake" or "typhoon".
    vuln_params : dict, optional
        Override vulnerability curve parameters.

    Returns
    -------
    DataFrame with columns: year, n_events, max_event_loss, annual_aggregate_loss
    """
    rng = np.random.default_rng(seed)
    records = []

    total_tsi = exposure_df["tsi_jpy_billion"].sum()

    for yr_idx, events in enumerate(hazard_simulations):
        if not events:
            records.append({
                "year": yr_idx,
                "n_events": 0,
                "max_event_loss": 0.0,
                "annual_aggregate_loss": 0.0,
            })
            continue

        event_losses = []
        for intensity in events:
            loss = compute_event_loss(intensity, exposure_df, peril, vuln_params)
            # Add stochastic variation (±30%) to account for event-level uncertainty
            loss *= max(0, 1.0 + rng.normal(0, 0.3))
            event_losses.append(loss)

        records.append({
            "year": yr_idx,
            "n_events": len(events),
            "max_event_loss": max(event_losses),
            "annual_aggregate_loss": sum(event_losses),
        })

    df = pd.DataFrame(records)
    logger.info(
        "Loss catalog for %s: %d years, mean annual loss = ¥%.2fB, "
        "max single-event = ¥%.2fB",
        peril, len(df), df["annual_aggregate_loss"].mean(),
        df["max_event_loss"].max()
    )
    return df


# =============================================================================
# Combined Loss Model
# =============================================================================

@dataclass
class LossModel:
    """
    Aggregates loss catalogs across perils into a combined annual loss view.
    """

    earthquake_losses: Optional[pd.DataFrame] = None
    typhoon_losses: Optional[pd.DataFrame] = None

    def combined_annual_losses(self) -> pd.DataFrame:
        """
        Merge earthquake and typhoon annual losses into a single DataFrame.

        Returns DataFrame with columns:
            year, eq_loss, ty_loss, total_annual_loss
        """
        frames = {}
        if self.earthquake_losses is not None:
            eq = self.earthquake_losses[["year", "annual_aggregate_loss"]].copy()
            eq = eq.rename(columns={"annual_aggregate_loss": "eq_loss"})
            frames["eq"] = eq
        if self.typhoon_losses is not None:
            ty = self.typhoon_losses[["year", "annual_aggregate_loss"]].copy()
            ty = ty.rename(columns={"annual_aggregate_loss": "ty_loss"})
            frames["ty"] = ty

        if not frames:
            return pd.DataFrame(columns=["year", "eq_loss", "ty_loss", "total_annual_loss"])

        if "eq" in frames and "ty" in frames:
            merged = frames["eq"].merge(frames["ty"], on="year", how="outer").fillna(0)
        elif "eq" in frames:
            merged = frames["eq"].copy()
            merged["ty_loss"] = 0.0
        else:
            merged = frames["ty"].copy()
            merged["eq_loss"] = 0.0

        merged["total_annual_loss"] = merged.get("eq_loss", 0) + merged.get("ty_loss", 0)
        merged = merged.sort_values("year").reset_index(drop=True)

        logger.info(
            "Combined loss model: %d years, mean total = ¥%.2fB, max = ¥%.2fB",
            len(merged), merged["total_annual_loss"].mean(),
            merged["total_annual_loss"].max()
        )
        return merged
