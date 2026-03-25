"""
Tests for Reinsurance Structures Module
=======================================
Covers:
- ExcessOfLoss allocation (below, within, above layer)
- QuotaShare allocation (proportional split)
- ReinsuranceProgram stacking (multi-layer, ordering)
- Zero-loss edge case (all structures return zero)
- Negative/edge inputs
"""

import pytest
import numpy as np
import pandas as pd

from src.reinsurance_structures import (
    ExcessOfLoss,
    QuotaShare,
    ReinsuranceProgram,
    apply_program,
)


# =============================================================================
# ExcessOfLoss Tests
# =============================================================================

class TestExcessOfLoss:
    """Test suite for XoL layer allocation logic."""

    def setup_method(self):
        # ¥50B xs ¥30B layer
        self.layer = ExcessOfLoss(attachment=30.0, limit=50.0)

    def test_loss_below_attachment(self):
        """Loss entirely below attachment → cedent retains all."""
        cedent, reinsurer = self.layer.allocate(20.0)
        assert cedent == 20.0
        assert reinsurer == 0.0

    def test_loss_at_attachment(self):
        """Loss exactly at attachment → cedent retains all."""
        cedent, reinsurer = self.layer.allocate(30.0)
        assert cedent == 30.0
        assert reinsurer == 0.0

    def test_loss_within_layer(self):
        """Loss within the layer → reinsurer pays excess above attachment."""
        cedent, reinsurer = self.layer.allocate(50.0)
        # 50 - 30 = 20 to reinsurer, 30 retained by cedent
        assert reinsurer == 20.0
        assert cedent == 30.0
        # Sum must equal gross
        assert cedent + reinsurer == 50.0

    def test_loss_exhausts_layer(self):
        """Loss exactly exhausts the layer → reinsurer pays full limit."""
        cedent, reinsurer = self.layer.allocate(80.0)
        # 80 - 30 = 50, but limit is 50 → reinsurer pays 50
        assert reinsurer == 50.0
        assert cedent == 30.0

    def test_loss_above_layer(self):
        """Loss exceeds attachment + limit → reinsurer capped at limit."""
        cedent, reinsurer = self.layer.allocate(120.0)
        assert reinsurer == 50.0
        assert cedent == 70.0  # 30 (retention) + 40 (above layer)
        assert cedent + reinsurer == 120.0

    def test_zero_loss(self):
        """Zero loss → both parties get zero."""
        cedent, reinsurer = self.layer.allocate(0.0)
        assert cedent == 0.0
        assert reinsurer == 0.0

    def test_negative_loss(self):
        """Negative loss (should not occur) → treated as zero."""
        cedent, reinsurer = self.layer.allocate(-10.0)
        assert cedent == 0.0
        assert reinsurer == 0.0

    def test_small_excess(self):
        """Very small excess above attachment."""
        cedent, reinsurer = self.layer.allocate(30.01)
        assert reinsurer == pytest.approx(0.01)
        assert cedent == pytest.approx(30.0)


# =============================================================================
# QuotaShare Tests
# =============================================================================

class TestQuotaShare:
    """Test suite for Quota Share allocation logic."""

    def setup_method(self):
        self.layer = QuotaShare(cession_pct=0.30)

    def test_proportional_split(self):
        """30% QS → reinsurer takes 30%, cedent retains 70%."""
        cedent, reinsurer = self.layer.allocate(100.0)
        assert reinsurer == pytest.approx(30.0)
        assert cedent == pytest.approx(70.0)

    def test_zero_loss(self):
        """Zero loss → zero for both."""
        cedent, reinsurer = self.layer.allocate(0.0)
        assert cedent == 0.0
        assert reinsurer == 0.0

    def test_sum_equals_gross(self):
        """Cedent + reinsurer must always equal gross loss."""
        for loss in [0, 1, 50, 100, 999.99]:
            cedent, reinsurer = self.layer.allocate(loss)
            assert cedent + reinsurer == pytest.approx(loss)

    def test_full_cession(self):
        """100% QS → reinsurer takes everything."""
        full_qs = QuotaShare(cession_pct=1.0)
        cedent, reinsurer = full_qs.allocate(100.0)
        assert reinsurer == pytest.approx(100.0)
        assert cedent == pytest.approx(0.0)

    def test_zero_cession(self):
        """0% QS → cedent retains everything."""
        no_qs = QuotaShare(cession_pct=0.0)
        cedent, reinsurer = no_qs.allocate(100.0)
        assert reinsurer == pytest.approx(0.0)
        assert cedent == pytest.approx(100.0)

    def test_invalid_cession_pct(self):
        """Cession percentage out of range should raise."""
        with pytest.raises(ValueError):
            QuotaShare(cession_pct=1.5)
        with pytest.raises(ValueError):
            QuotaShare(cession_pct=-0.1)


# =============================================================================
# ReinsuranceProgram Tests
# =============================================================================

class TestReinsuranceProgram:
    """Test suite for multi-layer program stacking."""

    def test_single_xol(self):
        """Program with single XoL layer."""
        program = ReinsuranceProgram(
            name="Single XoL",
            layers=[ExcessOfLoss(attachment=10, limit=20)]
        )
        result = program.allocate(25.0)
        assert result["gross_loss"] == 25.0
        assert result["total_ceded"] == 15.0  # 25 - 10 = 15, within limit
        assert result["net_loss"] == 10.0

    def test_multi_layer_xol(self):
        """Program with multiple XoL layers (tower)."""
        program = ReinsuranceProgram(
            name="XoL Tower",
            layers=[
                ExcessOfLoss(attachment=10, limit=20),   # covers 10-30
                ExcessOfLoss(attachment=30, limit=50),   # covers 30-80
            ]
        )
        # Loss of 60: layer 1 pays 20 (full), remaining 40
        # layer 2 sees remaining 10 (which is below its attachment of 30)
        # Wait, the logic is sequential — after layer 1, remaining = 10
        # Then layer 2 gets input=10, attachment=30 → below → 0
        result = program.allocate(60.0)
        assert result["gross_loss"] == 60.0
        # Layer 1: 60 > 10, excess = 50, capped at 20 → reinsurer pays 20, remaining = 40
        # Layer 2: 40 > 30, excess = 10, capped at 50 → reinsurer pays 10, remaining = 30
        assert result["total_ceded"] == 30.0
        assert result["net_loss"] == 30.0

    def test_qs_then_xol(self):
        """Blended: QS applied first, then XoL on net-after-QS."""
        program = ReinsuranceProgram(
            name="QS + XoL",
            layers=[
                QuotaShare(cession_pct=0.20),           # 20% ceded first
                ExcessOfLoss(attachment=15, limit=35),   # XoL on remaining 80%
            ]
        )
        # Loss = 100
        # QS: cede 20, remaining = 80
        # XoL: 80 > 15, excess = 65, capped at 35 → reinsurer pays 35, remaining = 45
        result = program.allocate(100.0)
        assert result["total_ceded"] == pytest.approx(55.0)  # 20 + 35
        assert result["net_loss"] == pytest.approx(45.0)

    def test_zero_loss_program(self):
        """Zero loss through any program → all zeros."""
        program = ReinsuranceProgram(
            name="Test",
            layers=[
                QuotaShare(cession_pct=0.30),
                ExcessOfLoss(attachment=10, limit=20),
            ]
        )
        result = program.allocate(0.0)
        assert result["gross_loss"] == 0.0
        assert result["total_ceded"] == 0.0
        assert result["net_loss"] == 0.0

    def test_conservation_of_loss(self):
        """Gross = net + ceded for various loss amounts."""
        program = ReinsuranceProgram(
            name="Conservation",
            layers=[
                QuotaShare(cession_pct=0.25),
                ExcessOfLoss(attachment=20, limit=40),
            ]
        )
        for loss in [0, 5, 20, 50, 100, 500]:
            result = program.allocate(loss)
            assert result["net_loss"] + result["total_ceded"] == pytest.approx(loss)


# =============================================================================
# Zero-Loss Catalog Edge Case
# =============================================================================

class TestZeroLossCatalog:
    """Test that all structures handle a catalog with only zero-loss years."""

    def setup_method(self):
        """Create a DataFrame of all-zero annual losses."""
        self.zero_catalog = pd.DataFrame({
            "year": list(range(100)),
            "total_annual_loss": [0.0] * 100,
        })

    def test_xol_zero_catalog(self):
        """XoL on zero catalog → all outputs are zero."""
        program = ReinsuranceProgram(
            name="XoL",
            layers=[ExcessOfLoss(attachment=10, limit=50)]
        )
        result = apply_program(self.zero_catalog, program)
        assert result["gross_loss"].sum() == 0.0
        assert result["net_loss"].sum() == 0.0
        assert result["ceded_loss"].sum() == 0.0

    def test_qs_zero_catalog(self):
        """QS on zero catalog → all outputs are zero."""
        program = ReinsuranceProgram(
            name="QS",
            layers=[QuotaShare(cession_pct=0.30)]
        )
        result = apply_program(self.zero_catalog, program)
        assert result["gross_loss"].sum() == 0.0
        assert result["net_loss"].sum() == 0.0
        assert result["ceded_loss"].sum() == 0.0

    def test_blended_zero_catalog(self):
        """Blended program on zero catalog → all outputs are zero."""
        program = ReinsuranceProgram(
            name="Blended",
            layers=[
                QuotaShare(cession_pct=0.20),
                ExcessOfLoss(attachment=15, limit=35),
            ]
        )
        result = apply_program(self.zero_catalog, program)
        assert result["gross_loss"].sum() == 0.0
        assert result["net_loss"].sum() == 0.0
        assert result["ceded_loss"].sum() == 0.0


# =============================================================================
# apply_program Integration Test
# =============================================================================

class TestApplyProgram:
    """Integration tests for applying programs to loss catalogs."""

    def test_apply_to_catalog(self):
        """Apply XoL to a small catalog and verify output shape."""
        catalog = pd.DataFrame({
            "year": [1, 2, 3, 4, 5],
            "total_annual_loss": [5.0, 15.0, 25.0, 50.0, 100.0],
        })
        program = ReinsuranceProgram(
            name="Test XoL",
            layers=[ExcessOfLoss(attachment=10, limit=30)]
        )
        result = apply_program(catalog, program)

        assert len(result) == 5
        assert "gross_loss" in result.columns
        assert "net_loss" in result.columns
        assert "ceded_loss" in result.columns

        # Year 1: loss=5, below attachment → net=5, ceded=0
        assert result.iloc[0]["net_loss"] == pytest.approx(5.0)
        assert result.iloc[0]["ceded_loss"] == pytest.approx(0.0)

        # Year 4: loss=50, excess=40, capped at 30 → ceded=30, net=20
        assert result.iloc[3]["ceded_loss"] == pytest.approx(30.0)
        assert result.iloc[3]["net_loss"] == pytest.approx(20.0)
