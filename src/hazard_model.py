"""
Module 2: Hazard Model
======================
Fits frequency-severity distributions to historical JMA disaster data.

Frequency: Poisson distribution (annual event count)
Severity:  Lognormal or Generalized Pareto Distribution (event intensity)

The fitted HazardModel object can simulate n years of events for the
downstream loss model.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# Distribution Fitting
# =============================================================================

def fit_frequency_model(annual_counts: np.ndarray) -> dict:
    """
    Fit a Poisson distribution to annual event counts.

    Parameters
    ----------
    annual_counts : array-like
        Number of events per year (e.g. [12, 8, 15, ...]).

    Returns
    -------
    dict with keys: 'distribution', 'lambda', 'n_years', 'mean', 'variance'.
    """
    counts = np.asarray(annual_counts, dtype=float)
    lam = counts.mean()

    result = {
        "distribution": "poisson",
        "lambda": float(lam),
        "n_years": len(counts),
        "mean": float(counts.mean()),
        "variance": float(counts.var()),
        "dispersion_ratio": float(counts.var() / lam) if lam > 0 else None,
    }

    logger.info(
        "Poisson fit: λ=%.2f (n=%d years, dispersion=%.2f)",
        lam, len(counts), result["dispersion_ratio"] or 0
    )
    return result


def fit_severity_model(
    values: np.ndarray,
    distribution: str = "lognormal",
    threshold: Optional[float] = None,
) -> dict:
    """
    Fit a severity distribution to event intensity values.

    Parameters
    ----------
    values : array-like
        Event intensities (magnitude, wind speed, etc.).
    distribution : str
        "lognormal" or "gpd" (Generalized Pareto Distribution).
    threshold : float, optional
        Threshold for GPD fitting. If None, uses the 90th percentile.

    Returns
    -------
    dict with fitted parameters, KS test results, and descriptive stats.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) < 10:
        logger.warning("Too few values (%d) for severity fitting", len(vals))
        return {"distribution": distribution, "error": "insufficient data"}

    result = {
        "distribution": distribution,
        "n_observations": len(vals),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "percentile_95": float(np.percentile(vals, 95)),
        "percentile_99": float(np.percentile(vals, 99)),
    }

    if distribution == "lognormal":
        # Fit lognormal: log(X) ~ Normal(mu, sigma)
        log_vals = np.log(vals[vals > 0])
        mu, sigma = log_vals.mean(), log_vals.std()

        result["params"] = {"mu": float(mu), "sigma": float(sigma)}

        # KS test
        ks_stat, ks_pval = stats.kstest(
            log_vals, "norm", args=(mu, sigma)
        )
        result["ks_statistic"] = float(ks_stat)
        result["ks_pvalue"] = float(ks_pval)

        logger.info(
            "Lognormal fit: μ=%.3f, σ=%.3f (KS p=%.4f)",
            mu, sigma, ks_pval
        )

    elif distribution == "gpd":
        # Fit Generalized Pareto to excesses above threshold
        if threshold is None:
            threshold = float(np.percentile(vals, 90))

        excesses = vals[vals > threshold] - threshold
        if len(excesses) < 10:
            result["error"] = "too few excesses above threshold"
            return result

        # scipy.stats.genpareto.fit returns (c, loc, scale)
        shape, loc, scale = stats.genpareto.fit(excesses, floc=0)
        result["params"] = {
            "shape": float(shape),
            "scale": float(scale),
            "threshold": float(threshold),
        }

        # KS test
        ks_stat, ks_pval = stats.kstest(
            excesses, "genpareto", args=(shape, 0, scale)
        )
        result["ks_statistic"] = float(ks_stat)
        result["ks_pvalue"] = float(ks_pval)

        logger.info(
            "GPD fit: ξ=%.3f, σ=%.3f, threshold=%.2f (KS p=%.4f)",
            shape, scale, threshold, ks_pval
        )

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return result


# =============================================================================
# Hazard Model Class
# =============================================================================

@dataclass
class HazardModel:
    """
    Encapsulates fitted frequency-severity distributions and provides
    Monte Carlo simulation of annual events.

    Attributes
    ----------
    peril : str
        "earthquake" or "typhoon".
    freq_params : dict
        Poisson fit result from fit_frequency_model().
    sev_params : dict
        Severity fit result from fit_severity_model().
    """

    peril: str
    freq_params: dict = field(default_factory=dict)
    sev_params: dict = field(default_factory=dict)

    def simulate(
        self,
        n_years: int = 10000,
        seed: int = 42,
    ) -> list[list[float]]:
        """
        Simulate n_years of events.

        Returns
        -------
        list of lists
            Each inner list contains event intensities for that simulated year.
            E.g. [[6.2, 5.5], [], [7.1], ...] for earthquake magnitudes.
        """
        rng = np.random.default_rng(seed)
        lam = self.freq_params.get("lambda", 5.0)
        dist = self.sev_params.get("distribution", "lognormal")

        years = []
        for _ in range(n_years):
            n_events = rng.poisson(lam)
            if n_events == 0:
                years.append([])
                continue

            if dist == "lognormal":
                params = self.sev_params.get("params", {"mu": 1.5, "sigma": 0.3})
                mu = params["mu"]
                sigma = params["sigma"]
                intensities = rng.lognormal(mu, sigma, size=n_events)
            elif dist == "gpd":
                params = self.sev_params.get("params", {"shape": 0.1, "scale": 1.0, "threshold": 5.0})
                shape = params["shape"]
                scale = params["scale"]
                threshold = params["threshold"]
                # Simulate from GPD and add back the threshold
                u = rng.uniform(size=n_events)
                if abs(shape) < 1e-10:
                    excesses = -scale * np.log(u)
                else:
                    excesses = (scale / shape) * (u ** (-shape) - 1)
                intensities = excesses + threshold
            else:
                raise ValueError(f"Unknown distribution: {dist}")

            years.append(intensities.tolist())

        logger.info(
            "Simulated %d years for %s: mean %.1f events/yr",
            n_years, self.peril, lam
        )
        return years


# =============================================================================
# High-level Fitting Pipeline
# =============================================================================

def build_hazard_model(
    events_df: pd.DataFrame,
    peril: str,
    intensity_col: str,
    severity_distribution: str = "lognormal",
    year_col: str = "year",
) -> HazardModel:
    """
    Convenience function: fit frequency + severity and return a HazardModel.

    Parameters
    ----------
    events_df : DataFrame
        Cleaned event catalog with year and intensity columns.
    peril : str
        "earthquake" or "typhoon".
    intensity_col : str
        Column name for severity measure (e.g. "magnitude", "max_wind_kt").
    severity_distribution : str
        "lognormal" or "gpd".
    year_col : str
        Column name for year.
    """
    # Annual counts for frequency fit
    year_range = range(events_df[year_col].min(), events_df[year_col].max() + 1)
    counts_by_year = events_df.groupby(year_col).size()
    annual_counts = np.array([
        counts_by_year.get(y, 0) for y in year_range
    ])

    freq_params = fit_frequency_model(annual_counts)
    sev_params = fit_severity_model(
        events_df[intensity_col].values,
        distribution=severity_distribution,
    )

    model = HazardModel(
        peril=peril,
        freq_params=freq_params,
        sev_params=sev_params,
    )

    logger.info("Built HazardModel for %s", peril)
    return model
