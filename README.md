# CUESG — AI-Powered ESG Intelligence Platform

> Christ University · Internship Cohort 1 · Sustainable AI · 2026  
> Team 2 · Domain: Enterprise Sustainability Tech / ESG Compliance

---

## What is CUESG?

CUESG is a deterministic ESG intelligence platform built for Indian commercial real estate portfolios. It automates SEBI BRSR Principle 6 compliance — computing Scope 1, 2, and 3 emissions, detecting energy anomalies, forecasting 12 months ahead, and generating audit-ready PDF dossiers — all in under 60 seconds on CPU.

---

## Key Features

- **Deterministic GHG Engine** — CEA v20 (0.727 kg CO₂/kWh), IPCC AR4, ICAP constants locked at the source
- **Isolation Forest Anomaly Detection** — 3 physics-based signatures: refrigerant leak, diesel theft, HVAC drift
- **XGBoost 12-Month Forecast** — MAPE <15% (SRD v3.0 UC-3), vectorised batch inference
- **SHAP Explainability** — every anomaly detection is explainable and audit-ready
- **SEBI BRSR PDF Dossier** — 8-page compliance report generated automatically
- **eco2AI Carbon Tracking** — every training run logs tCO₂e

---

## Regulatory Sources

| Constant | Value | Source |
|---|---|---|
| Grid Emission Factor | 0.727 kg CO₂/kWh | CEA CO₂ Baseline Ver 20.0 (Dec 2024) |
| HSD Diesel | 2.68 kg CO₂/litre | IPCC 2006 AR4 — BEE/MoEFCC |
| R-410A GWP | 2,088 | India Cooling Action Plan (ICAP) / IPCC AR4 |
| EUI Baseline | 140–180 kWh/m²/yr | BEE Star Rating Baseline (Commercial) |

---

## Quick Start
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cuesg.git
cd cuesg

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then upload `data/cuesg_demo_portfolio.csv` in the UI to activate the full dashboard.

---

## Project Structure
```
cuesg/
├── app.py                    # Streamlit dashboard (main entry point)
├── greenlens_ai_core.py      # ML engine: anomaly detection, forecasting, SHAP
├── greenlens_pdf_engine.py   # ReportLab PDF dossier generator
├── data/
│   └── cuesg_demo_portfolio.csv   # 900-row synthetic ESG portfolio
└── requirements.txt
```
