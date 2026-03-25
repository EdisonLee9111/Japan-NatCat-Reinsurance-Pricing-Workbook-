"""
Module 4: Reinsurance Structures
================================
Core module implementing reinsurance layer allocation.

Structures supported:
- Excess of Loss (XoL): cedent retains up to attachment; reinsurer pays
  within (attachment, attachment+limit]; cedent retains excess above limit.
- Quota Share (QS): proportional sharing of all losses.
- ReinsuranceProgram: stacks multiple layers, applying them in order.

All monetary values are in ¥ billion to match the exposure portfolio.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Base Layer
# =============================================================================

class ReinsuranceLayer(ABC):
    """Base class for all reinsurance structures."""

    @abstractmethod
    def allocate(self, gross_loss: float) -> tuple[float, float]:
        """
        Allocate a single loss between cedent and reinsurer.

        Parameters
        ----------
        gross_loss : float
            Gross loss amount (¥ billion).

        Returns
        -------
        (cedent_retained, reinsurer_pays) : tuple[float, float]
            Must sum to gross_loss.
        """
        ...

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of this layer."""
        ...


# =============================================================================
# Excess of Loss
# =============================================================================

@dataclass
class ExcessOfLoss(ReinsuranceLayer):
    """
    Excess of Loss (XoL) layer.

    The reinsurer covers losses between 'attachment' and 'attachment + limit'.
    - Loss ≤ attachment      → cedent retains 100%
    - attachment < loss ≤ attachment + limit → reinsurer pays the excess
    - Loss > attachment + limit → reinsurer pays 'limit', cedent retains rest

    Attributes
    ----------
    attachment : float
        Attachment point (¥ billion). Cedent retention below this.
    limit : float
        Layer limit (¥ billion). Maximum reinsurer payout.
    description : str
        Human-readable label (e.g. "¥50B xs ¥30B").
    """

    attachment: float
    limit: float
    description: str = ""

    def __post_init__(self):
        if not self.description:
            self.description = f"¥{self.limit:.0f}B xs ¥{self.attachment:.0f}B"

    def allocate(self, gross_loss: float) -> tuple[float, float]:
        """Split loss between cedent and reinsurer for this XoL layer."""
        if gross_loss <= 0:
            return (0.0, 0.0)

        if gross_loss <= self.attachment:
            # Loss entirely within cedent retention
            return (gross_loss, 0.0)

        excess = gross_loss - self.attachment
        reinsurer_pays = min(excess, self.limit)
        cedent_retained = gross_loss - reinsurer_pays

        return (cedent_retained, reinsurer_pays)

    def describe(self) -> str:
        return f"XoL: {self.description}"


# =============================================================================
# Quota Share
# =============================================================================

@dataclass
class QuotaShare(ReinsuranceLayer):
    """
    Quota Share (QS) layer.

    The reinsurer takes a fixed percentage of every loss.

    Attributes
    ----------
    cession_pct : float
        Cession percentage (0.0 to 1.0). E.g. 0.30 = 30% QS.
    description : str
        Human-readable label.
    """

    cession_pct: float
    description: str = ""

    def __post_init__(self):
        if not (0.0 <= self.cession_pct <= 1.0):
            raise ValueError(
                f"cession_pct must be between 0 and 1, got {self.cession_pct}"
            )
        if not self.description:
            self.description = f"{self.cession_pct:.0%} Quota Share"

    def allocate(self, gross_loss: float) -> tuple[float, float]:
        """Split loss proportionally between cedent and reinsurer."""
        if gross_loss <= 0:
            return (0.0, 0.0)

        reinsurer_pays = gross_loss * self.cession_pct
        cedent_retained = gross_loss - reinsurer_pays
        return (cedent_retained, reinsurer_pays)

    def describe(self) -> str:
        return f"QS: {self.description}"


# =============================================================================
# Reinsurance Program (layer stacking)
# =============================================================================

@dataclass
class ReinsuranceProgram:
    """
    A reinsurance program consisting of one or more ordered layers.

    Layers are applied sequentially. For mixed programs (QS + XoL),
    QS is applied first (reducing the loss base), then XoL layers act
    on the net-after-QS loss.

    Attributes
    ----------
    name : str
        Program name for reporting.
    layers : list[ReinsuranceLayer]
        Ordered list of reinsurance layers.
    """

    name: str
    layers: list[ReinsuranceLayer] = field(default_factory=list)

    def allocate(self, gross_loss: float) -> dict:
        """
        Run a gross loss through the entire program.

        Returns
        -------
        dict with keys:
            - gross_loss: original loss
            - net_loss: cedent's final retained loss
            - total_ceded: total reinsurer payment across all layers
            - layer_details: list of per-layer (description, cedent, reinsurer)
        """
        remaining = gross_loss
        total_ceded = 0.0
        layer_details = []

        for layer in self.layers:
            if isinstance(layer, QuotaShare):
                # QS applies to the current remaining loss
                cedent, reinsurer = layer.allocate(remaining)
                remaining = cedent
                total_ceded += reinsurer
                layer_details.append({
                    "layer": layer.describe(),
                    "input_loss": remaining + reinsurer,
                    "cedent_retained": cedent,
                    "reinsurer_pays": reinsurer,
                })
            elif isinstance(layer, ExcessOfLoss):
                # XoL applies to the current remaining (net-after-QS) loss
                cedent, reinsurer = layer.allocate(remaining)
                remaining = cedent
                total_ceded += reinsurer
                layer_details.append({
                    "layer": layer.describe(),
                    "input_loss": cedent + reinsurer,
                    "cedent_retained": cedent,
                    "reinsurer_pays": reinsurer,
                })

        return {
            "gross_loss": gross_loss,
            "net_loss": remaining,
            "total_ceded": total_ceded,
            "layer_details": layer_details,
        }

    def describe(self) -> str:
        parts = [f"Program: {self.name}"]
        for i, layer in enumerate(self.layers, 1):
            parts.append(f"  Layer {i}: {layer.describe()}")
        return "\n".join(parts)


# =============================================================================
# Apply Program to Loss Catalog
# =============================================================================

def apply_program(
    annual_losses: pd.DataFrame,
    program: ReinsuranceProgram,
    loss_col: str = "total_annual_loss",
) -> pd.DataFrame:
    """
    Apply a reinsurance program to an annual loss catalog.

    Parameters
    ----------
    annual_losses : DataFrame
        Must contain a column with annual aggregate losses.
    program : ReinsuranceProgram
        The reinsurance structure to apply.
    loss_col : str
        Column name for the annual aggregate loss.

    Returns
    -------
    DataFrame with added columns:
        - gross_loss: copy of the input loss column
        - net_loss: cedent's retained loss after reinsurance
        - ceded_loss: total paid by reinsurer(s)
    """
    results = []
    for _, row in annual_losses.iterrows():
        gross = row[loss_col]
        allocation = program.allocate(gross)
        results.append({
            "year": row.get("year", _),
            "gross_loss": allocation["gross_loss"],
            "net_loss": allocation["net_loss"],
            "ceded_loss": allocation["total_ceded"],
        })

    df = pd.DataFrame(results)
    logger.info(
        "Applied '%s': mean gross=¥%.2fB → net=¥%.2fB, ceded=¥%.2fB",
        program.name,
        df["gross_loss"].mean(),
        df["net_loss"].mean(),
        df["ceded_loss"].mean(),
    )
    return df


# =============================================================================
# Build Programs from Config
# =============================================================================

def build_programs_from_config(config: dict) -> dict[str, ReinsuranceProgram]:
    """
    Parse reinsurance program definitions from config.yaml.

    Returns a dict mapping program key → ReinsuranceProgram object.
    """
    programs = {}
    ri_config = config.get("reinsurance", {}).get("programs", {})

    for key, prog_def in ri_config.items():
        layers = []
        for layer_def in prog_def.get("layers", []):
            ltype = layer_def["type"].lower()
            desc = layer_def.get("description", "")

            if ltype == "xol":
                layers.append(ExcessOfLoss(
                    attachment=layer_def["attachment"],
                    limit=layer_def["limit"],
                    description=desc,
                ))
            elif ltype == "qs":
                layers.append(QuotaShare(
                    cession_pct=layer_def["cession_pct"],
                    description=desc,
                ))
            else:
                logger.warning("Unknown layer type '%s' in %s", ltype, key)

        programs[key] = ReinsuranceProgram(
            name=prog_def.get("name", key),
            layers=layers,
        )
        logger.info("Built program '%s' with %d layers", programs[key].name, len(layers))

    return programs
