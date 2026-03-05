# Japan NatCat Reinsurance Pricing Workbook

A fully dynamic Excel workbook for modeling catastrophe reinsurance placement on a hypothetical Japanese property insurance portfolio. Compares Excess of Loss and Quota Share treaty structures across historical and hypothetical disaster scenarios.

Built as self-directed research into reinsurance broking mechanics — how brokers structure, price, and advise on catastrophe risk transfer for Japanese cedents.

---

## What This Workbook Does

A Japanese property insurer (cedent) with ¥2,000B total sum insured faces earthquake and typhoon risk. This workbook answers the question every reinsurance broker must address at renewal:

**Which reinsurance structure gives this client the best balance of cost, tail risk protection, and earnings stability?**

The workbook models two structures side by side:

| Structure | How It Works | Best For |
|-----------|-------------|----------|
| **Excess of Loss (XoL)** | Reinsurer pays losses above an attachment point, up to a limit. 4-layer tower. | Protecting balance sheet against severe/peak events |
| **Quota Share** | Cedent cedes a fixed % of every loss. Proportional premium sharing. | Smoothing frequency losses, sharing volatility |

All inputs are adjustable — change an attachment point, cession percentage, or loss estimate and 289 linked formulas recalculate instantly.

---

## Workbook Structure

| Sheet | Contents |
|-------|----------|
| **Assumptions** | All tunable parameters (XoL layers, QS cession %, loading factor, brokerage). Blue = input cells. |
| **Exposure Portfolio** | Hypothetical portfolio across 16 prefectures, weighted by regional GDP and construction type |
| **Loss Scenarios** | 7 historical events (Tohoku 2011, Typhoon Jebi 2018, Kobe 1995, etc.) + 5 hypothetical scenarios (Nankai Trough M8.5+, 首都直下地震, Super Typhoon Tokyo) |
| **Reinsurance Structures** | Per-event loss allocation under XoL (cedent vs. reinsurer by layer) and QS. Formulas reference Assumptions sheet dynamically. |
| **Pricing & Comparison** | Rate on Line, technical premium, brokerage, and head-to-head XoL vs QS comparison with auto-generated recommendation |
| **Sensitivity Analysis** | QS cession % (10–50%) and XoL attachment point (¥5B–¥50B) impact on cost, ceded loss, and tail risk retention |
| **Broker Advisory** | Full advisory memo written from broker perspective, recommending a blended XoL + QS structure to a Japanese cedent ahead of April 1 renewal |
| **Dashboard** | Charts comparing premium cost, ceded loss, and maximum cedent retention across structures |

---

## Screenshots

> *Screenshots to be added after opening the workbook locally*

**Reinsurance Structures — XoL per-event allocation:**

`screenshots/xol_allocation.png`

**Pricing & Comparison — Head-to-head:**

`screenshots/pricing_comparison.png`

**Sensitivity Analysis — Attachment point vs ROL:**

`screenshots/sensitivity.png`

---

## Key Reinsurance Concepts

For context on terminology used in the workbook:

- **Cedent** — The insurance company buying reinsurance protection
- **Reinsurer** — The company assuming transferred risk (e.g., Munich Re, Swiss Re)
- **Broker** — The intermediary (e.g., Gallagher Re) who structures, negotiates, and places reinsurance on behalf of the cedent
- **Attachment Point** — The loss threshold where reinsurance coverage kicks in
- **Limit** — Maximum amount the reinsurer will pay per layer
- **Rate on Line (ROL)** — Premium ÷ Limit; the key pricing metric for XoL
- **Ceding Commission** — Commission the reinsurer pays back to the cedent under Quota Share
- **OEP / AEP** — Occurrence / Aggregate Exceedance Probability; industry-standard tail risk curves
- **April 1 Renewal** — Japan's fiscal year starts April 1; most Japanese property cat programs renew on this date

---

## Data Sources & Assumptions

| Item | Source | Notes |
|------|--------|-------|
| Historical events | Swiss Re Sigma, public industry reports | Loss figures are illustrative, calibrated to published insured loss ranges |
| Exposure portfolio | Hypothetical | Prefecture TSI weights based on Cabinet Office GDP data and GIROJ published rate relativities |
| Earthquake data | JMA (気象庁) Seismic Intensity Database | `src/data_ingestion.py` fetches and cleans historical records |
| Typhoon data | JMA Best Track Archive | Western Pacific typhoon history |
| Vulnerability functions | Simplified sigmoid/power-law | Not proprietary cat model output — intended as illustrative proxy |

> **Disclaimer:** This is a personal research project. Exposure data is fabricated. Loss estimates are approximate. This workbook is not intended for commercial underwriting or placement decisions.

---

## Project Files

```
japan-natcat-reinsurance-simulator/
│
├── README.md
├── Japan_NatCat_Reinsurance_Workbook.xlsx    ← Main deliverable
│
├── src/
│   ├── data_ingestion.py      # Fetch & clean JMA earthquake/typhoon data
│   └── build_workbook.py      # Python script that generates the Excel file
│
├── data/
│   ├── raw/                   # JMA source files
│   └── processed/             # Cleaned event catalogs (CSV)
│
├── screenshots/               # Workbook screenshots for README
│
└── requirements.txt           # numpy, pandas, openpyxl, scipy, requests, bs4
```

---


## Related Projects

- [Global LNG Arbitrage Monitor](https://github.com/EdisonLee9111/-Global-LNG-Arbitrage-Monitor) — Cross-market energy price spread analysis and risk factor modeling (Python)

---

## Author

**Zhengchao (Edison) Li**
M.S. Candidate, Institute of Science Tokyo
llee92063@gmail.com
