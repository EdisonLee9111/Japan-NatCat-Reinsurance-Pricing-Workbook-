#!/usr/bin/env python3
"""
Japan NatCat Reinsurance Simulator — End-to-End Demo
====================================================
Runs the full pipeline:
  1. Data Ingestion  → load & clean JMA earthquake/typhoon data
  2. Hazard Modeling  → fit Poisson frequency + lognormal severity
  3. Loss Simulation  → generate 10,000-year loss catalog
  4. Reinsurance      → apply 4 program structures
  5. Pricing          → compute EL, ROL, VaR, cost-efficiency
  6. Visualization    → generate charts for README
  7. Advisory Report  → broker-style placement memo

Usage:
    python run_demo.py
    python run_demo.py --config config.yaml --seed 42
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.data_ingestion import load_config, run_ingestion
from src.hazard_model import build_hazard_model
from src.loss_model import (
    load_exposure,
    generate_loss_catalog,
    LossModel,
)
from src.reinsurance_structures import build_programs_from_config, apply_program
from src.pricing_engine import compare_structures, generate_advisory_memo
from src.visualization import (
    plot_loss_exceedance_curve,
    plot_layer_loss_allocation,
    plot_structure_comparison,
    plot_vulnerability_curves,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_demo")


def main(config_path: str = "config.yaml"):
    """Run the full pipeline."""

    logger.info("=" * 70)
    logger.info("  Japan NatCat Reinsurance Simulator")
    logger.info("=" * 70)

    # ─────────────────────────────────────────────────────────────────────
    # 0. Load configuration
    # ─────────────────────────────────────────────────────────────────────
    config = load_config(config_path)
    sim_cfg = config.get("simulation", {})
    n_years = sim_cfg.get("n_years", 10000)
    seed = sim_cfg.get("random_seed", 42)

    logger.info("Config loaded: %d simulation years, seed=%d", n_years, seed)

    # ─────────────────────────────────────────────────────────────────────
    # 1. Data Ingestion
    # ─────────────────────────────────────────────────────────────────────
    logger.info("\n[STEP 1/7] Data Ingestion")
    eq_df, ty_df = run_ingestion(config)
    logger.info("  Earthquakes: %d events", len(eq_df))
    logger.info("  Typhoons:    %d events", len(ty_df))

    # ─────────────────────────────────────────────────────────────────────
    # 2. Hazard Modeling
    # ─────────────────────────────────────────────────────────────────────
    logger.info("\n[STEP 2/7] Hazard Modeling")
    sev_dist = config.get("hazard", {}).get("severity_distribution", "lognormal")

    eq_model = build_hazard_model(
        eq_df, peril="earthquake",
        intensity_col="magnitude",
        severity_distribution=sev_dist,
    )
    logger.info("  EQ model: λ=%.2f events/yr", eq_model.freq_params["lambda"])

    # For typhoons, use max_wind_kt as intensity metric
    ty_model = build_hazard_model(
        ty_df, peril="typhoon",
        intensity_col="max_wind_kt",
        severity_distribution=sev_dist,
    )
    logger.info("  TY model: λ=%.2f events/yr", ty_model.freq_params["lambda"])

    # ─────────────────────────────────────────────────────────────────────
    # 3. Loss Simulation
    # ─────────────────────────────────────────────────────────────────────
    logger.info("\n[STEP 3/7] Loss Simulation (%d years)", n_years)
    exposure = load_exposure(
        os.path.join(
            config.get("data", {}).get("reference_dir", "data/reference"),
            "japan_property_exposure.csv"
        )
    )
    total_tsi = exposure["tsi_jpy_billion"].sum()
    logger.info("  Total portfolio TSI: ¥%.1fB", total_tsi)

    # Simulate hazard events
    eq_sims = eq_model.simulate(n_years, seed=seed)
    ty_sims = ty_model.simulate(n_years, seed=seed + 1)

    # Generate loss catalogs
    loss_cfg = config.get("loss", {})
    eq_losses = generate_loss_catalog(
        eq_sims, exposure, "earthquake",
        vuln_params=loss_cfg.get("earthquake_vulnerability"),
        seed=seed,
    )
    ty_losses = generate_loss_catalog(
        ty_sims, exposure, "typhoon",
        vuln_params=loss_cfg.get("typhoon_vulnerability"),
        seed=seed + 1,
    )

    # Combine into LossModel
    loss_model = LossModel(earthquake_losses=eq_losses, typhoon_losses=ty_losses)
    combined = loss_model.combined_annual_losses()

    logger.info("  Mean annual loss:     ¥%.2fB", combined["total_annual_loss"].mean())
    logger.info("  99th percentile:      ¥%.2fB", np.percentile(combined["total_annual_loss"], 99))
    logger.info("  Max simulated loss:   ¥%.2fB", combined["total_annual_loss"].max())

    # ─────────────────────────────────────────────────────────────────────
    # 4. Reinsurance Programs
    # ─────────────────────────────────────────────────────────────────────
    logger.info("\n[STEP 4/7] Building Reinsurance Programs")
    programs = build_programs_from_config(config)
    for key, prog in programs.items():
        logger.info("  %s: %s (%d layers)", key, prog.name, len(prog.layers))

    # ─────────────────────────────────────────────────────────────────────
    # 5. Pricing & Comparison
    # ─────────────────────────────────────────────────────────────────────
    logger.info("\n[STEP 5/7] Pricing & Structure Comparison")
    comparison = compare_structures(
        programs, combined, config,
        total_tsi=total_tsi,
    )

    # Print comparison table
    print("\n" + "=" * 100)
    print("REINSURANCE STRUCTURE COMPARISON")
    print("=" * 100)
    display_cols = [
        "program_name", "expected_loss_bn", "technical_premium_bn",
        "rol_pct", "var_reduction_99_bn", "var_reduction_995_bn",
        "cost_efficiency_ratio",
    ]
    print(comparison[display_cols].to_string(index=False))
    print("=" * 100)

    # ─────────────────────────────────────────────────────────────────────
    # 6. Visualization
    # ─────────────────────────────────────────────────────────────────────
    logger.info("\n[STEP 6/7] Generating Visualizations")
    out_cfg = config.get("output", {})
    fig_dir = out_cfg.get("figures_dir", "outputs/figures")
    dpi = out_cfg.get("dpi", 150)

    plot_loss_exceedance_curve(
        combined, programs=programs,
        output_dir=fig_dir, dpi=dpi,
    )
    plot_layer_loss_allocation(
        combined, programs,
        output_dir=fig_dir, dpi=dpi,
    )
    plot_structure_comparison(
        comparison,
        output_dir=fig_dir, dpi=dpi,
    )
    plot_vulnerability_curves(
        config,
        output_dir=fig_dir, dpi=dpi,
    )
    logger.info("  Figures saved to %s/", fig_dir)

    # ─────────────────────────────────────────────────────────────────────
    # 7. Advisory Report
    # ─────────────────────────────────────────────────────────────────────
    logger.info("\n[STEP 7/7] Generating Advisory Report")
    report_dir = out_cfg.get("reports_dir", "outputs/reports")
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    memo = generate_advisory_memo(comparison, combined)

    # Add structure comparison table to the report
    full_report = []
    full_report.append(memo)
    full_report.append("")
    full_report.append("---")
    full_report.append("")
    full_report.append("## Detailed Metrics")
    full_report.append("")
    full_report.append(comparison.to_markdown(index=False))
    full_report.append("")
    full_report.append("---")
    full_report.append("")
    full_report.append("## Key Charts")
    full_report.append("")
    full_report.append("### Loss Exceedance Curve")
    full_report.append(f"![Loss Exceedance Curve](../figures/loss_exceedance_curve.png)")
    full_report.append("")
    full_report.append("### Loss Allocation by Program")
    full_report.append(f"![Layer Loss Allocation](../figures/layer_loss_allocation.png)")
    full_report.append("")
    full_report.append("### Structure Comparison")
    full_report.append(f"![Structure Comparison](../figures/structure_comparison.png)")
    full_report.append("")

    report_path = os.path.join(report_dir, "sample_placement_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_report))

    logger.info("  Report saved to %s", report_path)

    # ─────────────────────────────────────────────────────────────────────
    # Done
    # ─────────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  Pipeline complete!")
    logger.info("  - Figures:  %s/", fig_dir)
    logger.info("  - Report:   %s", report_path)
    logger.info("=" * 70)

    return comparison, combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Japan NatCat Reinsurance Simulator"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML file"
    )
    args = parser.parse_args()
    main(args.config)
