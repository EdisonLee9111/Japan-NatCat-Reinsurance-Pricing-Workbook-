# Reinsurance Placement Advisory Memo

## Portfolio Loss Profile

Based on 10,000 simulated years of combined earthquake and typhoon losses against the hypothetical Japanese property portfolio:

- **Mean Annual Loss**: ¥410.84B
- **99th Percentile (1-in-100 year)**: ¥1784.91B
- **99.5th Percentile (1-in-200 year)**: ¥1951.18B

---

## Structure-by-Structure Analysis

### Quota Share 30%

| Metric | Value |
|--------|-------|
| Expected Ceded Loss | ¥123.25B |
| Technical Premium | ¥174.71B |
| Rate on Line | 20.85% |
| 99th %ile VaR Reduction | ¥535.47B |
| 99.5th %ile VaR Reduction | ¥585.35B |
| Cost-Efficiency Ratio | 0.326 |

### QS 20% + XoL (2 layers)

| Metric | Value |
|--------|-------|
| Expected Ceded Loss | ¥166.49B |
| Technical Premium | ¥236.00B |
| Rate on Line | 31.47% |
| 99th %ile VaR Reduction | ¥491.98B |
| 99.5th %ile VaR Reduction | ¥525.24B |
| Cost-Efficiency Ratio | 0.480 |

### XoL Tower (3 layers)

| Metric | Value |
|--------|-------|
| Expected Ceded Loss | ¥116.83B |
| Technical Premium | ¥165.61B |
| Rate on Line | 61.49% |
| 99th %ile VaR Reduction | ¥190.00B |
| 99.5th %ile VaR Reduction | ¥190.00B |
| Cost-Efficiency Ratio | 0.872 |

### Cat XoL (single layer, high attach)

| Metric | Value |
|--------|-------|
| Expected Ceded Loss | ¥94.65B |
| Technical Premium | ¥134.17B |
| Rate on Line | 63.10% |
| 99th %ile VaR Reduction | ¥150.00B |
| 99.5th %ile VaR Reduction | ¥150.00B |
| Cost-Efficiency Ratio | 0.894 |

---

## Recommendation

The **Quota Share 30%** structure demonstrates **1.5x superior cost-efficiency** compared to **QS 20% + XoL (2 layers)** in terms of tail-risk compression per unit of premium.

Under the current loss distribution, **Quota Share 30%** achieves the lowest cost-efficiency ratio (0.326), indicating it provides the most VaR reduction per premium dollar. It reduces the 99th percentile loss from ¥1784.91B to ¥1249.44B at a technical premium of ¥174.71B.

### Strategic Considerations

- **If the cedent's priority is balance-sheet protection** against extreme scenarios (1-in-200 year events), an XoL-led structure with high attachment is recommended, as it provides targeted tail-risk transfer at a lower premium-to-benefit ratio.
- **If the cedent seeks earnings volatility smoothing** across all years (including attritional losses), a Quota Share or blended program provides broader but less capital-efficient protection.
- **Blended structures** (QS + XoL) offer a middle ground: the QS layer smooths frequency losses while the XoL layers protect against severity spikes. This is often the most pragmatic choice for Japanese cedents with significant earthquake exposure.

---

*This analysis is based on Monte Carlo simulation using historical JMA data calibrations. Actual placement terms would depend on market conditions, cedent financials, and underwriting considerations.*

---

## Detailed Metrics

| program_key   | program_name                        |   expected_loss_bn |   std_dev_bn |   gross_var_99_bn |   net_var_99_bn |   var_reduction_99_bn |   gross_var_995_bn |   net_var_995_bn |   var_reduction_995_bn |   gross_tvar_99_bn |   net_tvar_99_bn |   tvar_reduction_99_bn |   program_limit_bn |   rol_pct |   technical_premium_bn |   cost_efficiency_ratio |
|:--------------|:------------------------------------|-------------------:|-------------:|------------------:|----------------:|----------------------:|-------------------:|-----------------:|-----------------------:|-------------------:|-----------------:|-----------------------:|-------------------:|----------:|-----------------------:|------------------------:|
| program_a     | XoL Tower (3 layers)                |             116.83 |        89.42 |           1784.91 |         1594.91 |                190    |            1951.18 |          1761.18 |                 190    |            2094.08 |          1904.08 |                 190    |             190    |     61.49 |                 165.61 |                   0.872 |
| program_b     | Quota Share 30%                     |             123.25 |       130.76 |           1784.91 |         1249.44 |                535.47 |            1951.18 |          1365.82 |                 585.35 |            2094.08 |          1465.86 |                 628.22 |             591.15 |     20.85 |                 174.71 |                   0.326 |
| program_c     | QS 20% + XoL (2 layers)             |             166.49 |       140.9  |           1784.91 |         1292.93 |                491.98 |            1951.18 |          1425.94 |                 525.24 |            2094.08 |          1540.26 |                 553.82 |             529.1  |     31.47 |                 236    |                   0.48  |
| program_d     | Cat XoL (single layer, high attach) |              94.65 |        71.34 |           1784.91 |         1634.91 |                150    |            1951.18 |          1801.18 |                 150    |            2094.08 |          1944.08 |                 150    |             150    |     63.1  |                 134.17 |                   0.894 |

---

## Key Charts

### Loss Exceedance Curve
![Loss Exceedance Curve](../figures/loss_exceedance_curve.png)

### Loss Allocation by Program
![Layer Loss Allocation](../figures/layer_loss_allocation.png)

### Structure Comparison
![Structure Comparison](../figures/structure_comparison.png)
