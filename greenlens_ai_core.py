from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import importlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest

# ── PDF engine (ReportLab-based, replaces CorporateDossierPDF entirely) ──────
try:
    from greenlens_pdf_engine import build_pdf_report_v2 as _build_pdf_report_v2
    _PDF_ENGINE_AVAILABLE = True
except ImportError:
    _build_pdf_report_v2 = None  # type: ignore[assignment]
    _PDF_ENGINE_AVAILABLE = False

try:
    import plotly.io as pio
except Exception:
    pio = None


def track(project_name: str = "", emission_level: str = ""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


_OPTIONAL_CACHE: dict[str, Any] = {}


def _optional_import(module_name: str, attr_name: str | None = None) -> Any:
    cache_key = f"{module_name}:{attr_name or ''}"
    if cache_key in _OPTIONAL_CACHE:
        return _OPTIONAL_CACHE[cache_key]
    try:
        module = importlib.import_module(module_name)
        result = getattr(module, attr_name) if attr_name else module
    except Exception:
        result = None
    _OPTIONAL_CACHE[cache_key] = result
    return result


GREENLENS_SYSTEM_METADATA: dict[str, Any] = {
    "Regulatory_Constants": {
        "Grid_Emission_Factor_kgCO2_kWh": 0.727,
        "Grid_Emission_Factor_Source": "CEA CO2 Baseline Database Ver 20.0 (Dec 2024)",
        "HSD_Emission_Factor_kgCO2_L": 2.68,
        "HSD_Emission_Factor_Source": "IPCC 2006 AR4 (Adopted by BEE/MoEFCC)",
        "PNG_Emission_Factor_kgCO2_m3": 2.02,
        "PNG_Emission_Factor_Source": "GAIL / IPCC 2006",
        "GWP_R410A": 2088,
        "GWP_R134a": 1430,
        "GWP_Source": "India Cooling Action Plan (ICAP) / IPCC AR4",
        "Reference_Year": "2023-2024",
        "Regulatory_Framework": "SEBI BRSR Principle 6 | GRI Standards 2021",
    },
    "Building_Physics_Baselines": {
        "Source": "BEE Energy Conservation & Sustainable Building Code (ECSBC 2024) | ISHRAE",
        "Climate_Zones": {
            "Temperate_Bengaluru": {
                "Design_Temp_C": 24, "Design_RH_pct": 50,
                "Water_Intensity_kL_m2_yr": [0.8, 1.0],
                "Water_Benchmark_Source": "BEE Performance Benchmarking Grade-A Offices",
            },
            "Warm_Humid_Mumbai": {
                "Design_Temp_C": [24, 26], "Design_RH_pct": [55, 60],
                "Water_Intensity_kL_m2_yr": [1.25, 1.50],
                "Water_Benchmark_Source": "BEE Performance Benchmarking Grade-A Offices",
            },
        },
        "HVAC_Efficiencies": {
            "Thermal_Conversion_kW_per_TR": 3.517,
            "Water_Cooled_kW_TR": [0.65, 0.75],
            "Air_Cooled_kW_TR": [1.0, 1.2],
            "HVAC_Source": "Industrial HVAC Audit Standard India | BEE Star Rating",
        },
        "Fenestration": {
            "Max_SHGC": 0.25, "Max_U_Factor_Wall_W_m2K": 0.40,
            "Indoor_Design_Temp_C": 24,
            "Source": "ECSBC 2024 | ISHRAE / BEE",
        },
        "Energy_Use_Intensity": {
            "EUI_Baseline_kWh_m2_yr": [140, 180],
            "Source": "BEE Star Rating Baseline Commercial",
        },
        "Occupancy_Base_Load_Ratio": 0.30,
        "Occupancy_Load_Source": "Indian IT Park Load Profile Research",
    },
    "AI_Performance_Benchmarks": {
        "Source": "GreenLens SRD v3.0 Acceptance Criteria",
        "Anomaly_Precision_Min": 0.85,
        "Anomaly_Recall_Min": 0.80,
        "Forecast_MAPE_Max_pct": 15.0,
        "Field_Extraction_Accuracy_Min_pct": 85.0,
        "Benchmark_Reference": "UC-1 (Extraction), UC-2 (Anomaly), UC-3 (Forecast)",
    },
    "Isolation_Forest_Signatures": {
        "Refrigerant_Leak": {
            "Signature": "Positive residual error in Energy vs CDD regression; increased kW/TR",
            "Threshold": "> 12% energy spike vs baseline",
            "Source": "Physics-based Isolation Forest Signature | GreenLens SRD v3.0",
        },
        "Sensor_Drift": {
            "Signature": "x-intercept shift in Energy vs Outdoor Temp correlation matrix",
            "Source": "HVAC Audit Best Practice | ASHRAE Guideline 14",
        },
        "Fuel_Theft": {
            "Signature": "Negative dV/dt threshold breach while P_gen == 0",
            "Threshold": "> 0.5 L/hr at idle",
            "Source": "DG Set Telemetry Outlier Analysis | GreenLens SRD v3.0",
        },
    },
    "BRSR_Mapping_Logic": {
        "Principle": 6,
        "Source": "SEBI BRSR (May 2021) mandatory disclosures",
        "Essential_Indicators": ["EI-1 to EI-10", "Grid_kWh", "DG_Fuel_L", "Water_Withdrawal_KL", "Scope_1_2_Emissions"],
        "Leadership_Indicators": ["Scope_3_Emissions", "Intensity_Reduction_YoY", "LCA_Score"],
        "Scope3_Weighting_pct": "15-20% of Environmental Score",
        "Scope3_Source": "SEBI Leadership Indicators / Big 4 Rating Matrix",
    },
    "Mitigation_Recommendations": [
        "Re-calibrate chiller sequencing logic to prioritize high-efficiency base-load chillers.",
        "Deploy VFDs on secondary chilled water pumps using differential pressure control.",
        "Integrate return-air CO2 sensors with fresh-air dampers to reduce over-ventilation.",
        "Conduct a thermographic audit to identify envelope leakage and latent heat ingress.",
        "Investigate harmonic distortion and degraded power factor at the main incomer panel.",
    ],
}

RAG_REFERENCE = [
    "SEBI BRSR Principle 6 requires disclosure of total energy consumed, Scope 1 and Scope 2 emissions, water withdrawal, water discharge, and solid waste generation.",
    "CEA Version 20 sets the grid emission factor at 0.727 kg CO2/kWh for FY 2023-24.",
    "ICAP and IPCC AR4 set refrigerant GWP references such as R-410A at 2088 and R-134a at 1430.",
    "ECSBC 2024 requires tighter envelope and HVAC performance baselines for Indian commercial buildings.",
]

NUMERIC_HINTS: dict[str, list[str]] = {
    "asset_id": ["asset", "building", "site", "facility", "property", "campus", "building code"],
    "month": ["month", "date", "period", "reporting month", "invoice date"],
    "floor_area_sqm": ["area", "sqm", "sq m", "gfa", "floor area"],
    "energy_kwh": ["energy", "kwh", "power", "electricity"],
    "diesel_litres": ["diesel", "fuel", "dg fuel"],
    "solar_kwh": ["solar", "renewable", "offset"],
    "water_withdrawal_kl": ["water", "withdraw", "kl"],
    "ewaste_kg": ["e-waste", "waste", "ewaste"],
    "hazardous_waste_kg": ["haz", "hazardous"],
    "procurement_spend_inr": ["procurement", "spend", "purchase", "vendor", "capex", "opex"],
    "utility_cost_inr": ["cost", "bill", "invoice"],
    "occupancy_percent": ["occupancy", "utilization", "headcount", "footfall"],
    "outdoor_temp_c": ["outdoor", "ambient", "external", "temp c"],
    "indoor_temp_c": ["indoor", "internal"],
    "humidity_percent": ["humidity"],
    "hvac_load_kw": ["hvac", "cooling", "chiller", "demand kw"],
    "lighting_kwh": ["lighting", "light load"],
    "generator_output_kw": ["generator output", "gen output", "dg output"],
}


def load_master_dataset() -> pd.DataFrame:
    for path in [Path("master_esg_data.xlsx"), Path("master_esg_data.csv")]:
        if not path.exists():
            continue
        if path.suffix.lower() == ".xlsx":
            return pd.read_excel(path)
        return pd.read_csv(path)
    return pd.DataFrame()


def generate_synthetic_demo_data() -> pd.DataFrame:
    """
    Generate a realistic synthetic ESG dataset for Indian commercial buildings.
    Works fully offline — no file needed. Uses seeded randomness for reproducibility.
    Covers 8 assets across Bengaluru and Mumbai, 24 months of monthly data.
    Embeds deliberate anomalies, solar variance, and seasonal patterns.
    """
    rng = np.random.default_rng(42)
    assets = [
        {"asset_id": "BLD-BLR-001", "floor_area_sqm": 18500, "city": "BLR", "solar_base": 22000, "energy_base": 185000},
        {"asset_id": "BLD-BLR-002", "floor_area_sqm": 12200, "city": "BLR", "solar_base": 8000,  "energy_base": 125000},
        {"asset_id": "BLD-BLR-003", "floor_area_sqm": 9800,  "city": "BLR", "solar_base": 0,     "energy_base": 98000},
        {"asset_id": "BLD-MUM-001", "floor_area_sqm": 22000, "city": "MUM", "solar_base": 15000, "energy_base": 240000},
        {"asset_id": "BLD-MUM-002", "floor_area_sqm": 14500, "city": "MUM", "solar_base": 5000,  "energy_base": 155000},
        {"asset_id": "BLD-MUM-003", "floor_area_sqm": 8200,  "city": "MUM", "solar_base": 0,     "energy_base": 88000},
        {"asset_id": "BLD-BLR-004", "floor_area_sqm": 31000, "city": "BLR", "solar_base": 45000, "energy_base": 310000},
        {"asset_id": "BLD-MUM-004", "floor_area_sqm": 27500, "city": "MUM", "solar_base": 28000, "energy_base": 275000},
    ]
    months = pd.date_range("2023-01-31", periods=24, freq="ME")
    rows = []
    for asset in assets:
        for i, month in enumerate(months):
            month_num = month.month
            seasonal = 1.0 + 0.18 * np.sin((month_num - 4) * np.pi / 6.0)
            noise = rng.normal(1.0, 0.06)
            energy = asset["energy_base"] * seasonal * noise
            solar = asset["solar_base"] * (0.9 + 0.2 * np.sin(month_num * np.pi / 6)) * rng.normal(1.0, 0.08)
            solar = max(solar, 0)
            is_blr = asset["city"] == "BLR"
            outdoor_temp = rng.normal(26.5 if is_blr else 31.0, 3.5)
            diesel = rng.exponential(220 if is_blr else 280)
            water = asset["floor_area_sqm"] * rng.normal(0.078, 0.015) * (1.0 + 0.12 * seasonal)
            hvac_load = (outdoor_temp - 24) * (asset["floor_area_sqm"] / 1350) * rng.normal(0.9, 0.1)
            # Inject deliberate anomalies on specific months/assets
            if asset["asset_id"] == "BLD-MUM-002" and month_num in [7, 8]:
                diesel *= 4.5  # Monsoon DG spike
            if asset["asset_id"] == "BLD-BLR-003" and i == 14:
                energy *= 2.1  # Faulty chiller event
            rows.append({
                "asset_id": asset["asset_id"],
                "month": month,
                "floor_area_sqm": asset["floor_area_sqm"],
                "energy_kwh": round(max(energy, 1000), 1),
                "solar_kwh": round(solar, 1),
                "diesel_litres": round(diesel, 1),
                "water_withdrawal_kl": round(max(water, 0), 1),
                "ewaste_kg": round(rng.exponential(18), 1),
                "hazardous_waste_kg": round(rng.exponential(4), 1),
                "procurement_spend_inr": round(rng.normal(850000, 120000), 0),
                "utility_cost_inr": round(energy * 9.2 * rng.normal(1.0, 0.05), 0),
                "occupancy_percent": round(rng.normal(78, 12), 1),
                "outdoor_temp_c": round(outdoor_temp, 1),
                "indoor_temp_c": 24.0,
                "humidity_percent": round(rng.normal(62 if not is_blr else 52, 8), 1),
                "hvac_load_kw": round(max(hvac_load, 10), 1),
                "lighting_kwh": round(energy * 0.18, 1),
                "generator_output_kw": round(rng.exponential(12), 1),
            })
    return pd.DataFrame(rows)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(r"(?i)inr", "", regex=True)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("approx", "", regex=False)
        .str.replace(r"[^0-9\.\-]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


@dataclass
class ProcessedPortfolio:
    df: pd.DataFrame
    summary: dict[str, Any]
    anomaly_records: pd.DataFrame
    latest_shap: dict[str, Any] | None
    asset_rankings: pd.DataFrame
    detected_columns: dict[str, str]
    source_name: str
    esg_score: float


class SessionSustainabilityTracker:
    def estimate_upload_footprint(self, rows: int, columns: int, file_name: str) -> float:
        return round(0.11 + (rows * columns * 0.000005) + (len(file_name) * 0.001), 4)

    def estimate_prompt_footprint(self, prompt: str, rows: int) -> float:
        return round(0.02 + (len(prompt) * 0.00022) + (min(rows, 120_000) * 0.000002), 4)


class MemoryManager:
    WINDOW_SIZE = 10

    @staticmethod
    def compress(messages: list[dict[str, Any]], prior_summary: str = "") -> tuple[str, list[dict[str, Any]]]:
        if len(messages) <= MemoryManager.WINDOW_SIZE:
            return prior_summary, messages
        older = messages[:-MemoryManager.WINDOW_SIZE]
        recent = messages[-MemoryManager.WINDOW_SIZE:]
        lines = []
        for message in older[-12:]:
            role = message.get("role", "assistant").upper()
            content = str(message.get("content", "")).strip().replace("\n", " ")
            if content:
                lines.append(f"[{role}] {content[:180]}")
        summary = (prior_summary + "\n" + "\n".join(lines)).strip()
        summary = "\n".join(summary.splitlines()[-24:])
        return summary, recent


class DeterministicMath:
    CEA_FACTOR = 0.727
    DIESEL_FACTOR = 2.68
    WATER_FACTOR = 0.000344
    EWASTE_FACTOR = 0.021
    HAZ_WASTE_FACTOR = 0.075
    OCCUPANCY_BASE_LOAD_RATIO = 0.30
    TARGET_KW_PER_TR = 0.70

    def __init__(self) -> None:
        self.detected_columns: dict[str, str] = {}

    def auto_clean_and_map(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
        cleaned = df.copy()
        cleaned.columns = [str(col).strip() for col in cleaned.columns]
        detected: dict[str, str] = {}
        lowered = {column: re.sub(r"[^a-z0-9]+", " ", str(column).strip().lower()).strip() for column in cleaned.columns}
        for canonical, hints in NUMERIC_HINTS.items():
            for hint in hints:
                found = next((column for column, norm in lowered.items() if hint in norm), None)
                if found:
                    detected[canonical] = found
                    break
        self.detected_columns = detected
        return cleaned, detected

    def _canonical_frame(self, df: pd.DataFrame, detected: dict[str, str]) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for canonical in NUMERIC_HINTS:
            source = detected.get(canonical)
            if source is None:
                out[canonical] = np.nan
            elif canonical == "asset_id":
                out[canonical] = df[source].astype(str).str.strip().replace({"": "UNKNOWN-ASSET"})
            elif canonical == "month":
                out[canonical] = pd.to_datetime(df[source], errors="coerce")
            else:
                out[canonical] = _coerce_numeric(df[source])
        return out

    def _vectorized_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.sort_values(["asset_id", "month"]).reset_index(drop=True)
        if out["month"].isna().all():
            out["month"] = pd.date_range("2020-01-31", periods=len(out), freq="D")
        out["month"] = out["month"].ffill().bfill()
        out["asset_id"] = out["asset_id"].fillna("UNKNOWN-ASSET").replace({"nan": "UNKNOWN-ASSET"})
        numeric_cols = [col for col in out.columns if col not in {"asset_id", "month"}]
        for column in numeric_cols:
            out[column] = pd.to_numeric(out[column], errors="coerce")
            out[column] = out.groupby("asset_id", sort=False)[column].transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
            out[column] = out[column].fillna(out[column].median())
            out[column] = out[column].fillna(0.0)
        return out

    @staticmethod
    def _infer_zone(asset_series: pd.Series) -> pd.Series:
        upper = asset_series.astype(str).str.upper()
        zone = np.where(upper.str.contains("BLR"), "Temperate_Bengaluru", np.where(upper.str.contains("MUM"), "Warm_Humid_Mumbai", "General_India"))
        return pd.Series(zone, index=asset_series.index)

    def _derive_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for column in ["floor_area_sqm", "energy_kwh", "diesel_litres", "solar_kwh", "water_withdrawal_kl", "ewaste_kg", "hazardous_waste_kg", "procurement_spend_inr", "utility_cost_inr"]:
            out[column] = out[column].clip(lower=0)
        out["floor_area_sqm"] = out["floor_area_sqm"].clip(lower=500)
        out["occupancy_percent"] = out["occupancy_percent"].clip(lower=0, upper=100)
        out["humidity_percent"] = out["humidity_percent"].clip(lower=0, upper=100)
        out["indoor_temp_c"] = out["indoor_temp_c"].replace(0, np.nan).fillna(24.0)
        out["outdoor_temp_c"] = out["outdoor_temp_c"].replace(0, np.nan).fillna(29.0)
        out["lighting_kwh"] = out["lighting_kwh"].replace(0, np.nan).fillna(out["energy_kwh"] * 0.20)
        out["generator_output_kw"] = out["generator_output_kw"].fillna(0.0)
        out["Cooling_Delta_C"] = np.maximum(out["outdoor_temp_c"] - out["indoor_temp_c"], 0.0)
        out["Occupancy_Ratio"] = out["occupancy_percent"] / 100.0
        out["Occupancy_Adjusted_Load_Ratio"] = self.OCCUPANCY_BASE_LOAD_RATIO + ((1 - self.OCCUPANCY_BASE_LOAD_RATIO) * out["Occupancy_Ratio"])
        out["Base_Load_Estimate_kWh"] = out["energy_kwh"] * self.OCCUPANCY_BASE_LOAD_RATIO
        out["hvac_load_kw"] = out["hvac_load_kw"].replace(0, np.nan).fillna(out["Cooling_Delta_C"] * (out["floor_area_sqm"] / 1300.0) * out["Occupancy_Adjusted_Load_Ratio"])
        out["Cooling_TR_Est"] = out["hvac_load_kw"] / max(self.TARGET_KW_PER_TR, 0.001)
        out["kW_per_TR"] = out["hvac_load_kw"] / np.maximum(out["Cooling_TR_Est"], 0.001)
        out["Energy_Intensity_kWh_per_sqm"] = out["energy_kwh"] / np.maximum(out["floor_area_sqm"], 1.0)
        out["Solar_Integration_Ratio"] = out["solar_kwh"] / np.maximum(out["energy_kwh"], 1.0)
        out["Water_Intensity_kL_per_sqm"] = out["water_withdrawal_kl"] / np.maximum(out["floor_area_sqm"], 1.0)
        out["Climate_Zone"] = self._infer_zone(out["asset_id"])
        out["Water_Benchmark_Max"] = np.where(out["Climate_Zone"] == "Temperate_Bengaluru", 1.0, np.where(out["Climate_Zone"] == "Warm_Humid_Mumbai", 1.5, 1.3))
        out["Annualized_Water_Intensity"] = out["Water_Intensity_kL_per_sqm"] * 12.0
        out["Water_Benchmark_Breach"] = out["Annualized_Water_Intensity"] > out["Water_Benchmark_Max"]
        out["Scope1_tCO2e"] = (out["diesel_litres"] * self.DIESEL_FACTOR) / 1000.0
        out["Scope2_tCO2e"] = ((out["energy_kwh"] - out["solar_kwh"]).clip(lower=0) * self.CEA_FACTOR) / 1000.0
        out["Scope3_tCO2e"] = (
            (out["water_withdrawal_kl"] * self.WATER_FACTOR)
            + (out["ewaste_kg"] * self.EWASTE_FACTOR / 1000.0)
            + (out["hazardous_waste_kg"] * self.HAZ_WASTE_FACTOR / 1000.0)
            + (out["procurement_spend_inr"] * 0.00000016)
        )
        out["Total_tCO2e"] = out["Scope1_tCO2e"] + out["Scope2_tCO2e"] + out["Scope3_tCO2e"]
        return out

    def auto_clean_outliers(self, df: pd.DataFrame, actions: list[dict[str, Any]]) -> pd.DataFrame:
        cleaned = df.copy()
        for action in actions:
            if action.get("kind") != "auto_clean_imputation":
                continue
            payload = action["payload"]
            indices = payload["row_indices"]
            column = payload["column"]
            median_value = cleaned.loc[~cleaned.index.isin(indices), column].median()
            cleaned.loc[cleaned.index.isin(indices), column] = median_value
        return cleaned

    def calculate_esg_score(self, df: pd.DataFrame, energy_col: str) -> float:
        intensity = df[energy_col].sum() / max(df["floor_area_sqm"].sum(), 1.0)
        anomaly_rate = df["Is_Anomaly"].mean() if "Is_Anomaly" in df.columns else 0.0
        solar_ratio = df["solar_kwh"].sum() / max(df[energy_col].sum(), 1.0)
        scope3_eff = 1.0 - min(df["Scope3_tCO2e"].sum() / max(df["Total_tCO2e"].sum(), 1.0), 1.0) if "Scope3_tCO2e" in df.columns else 0.5
        score = 100 - (intensity * 2.5) - (anomaly_rate * 35) + (solar_ratio * 18) + (scope3_eff * 8)
        return float(np.clip(round(score, 1), 0, 100))

    def calculate(self, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
        if raw_df is None or raw_df.empty:
            raise ValueError("The uploaded dataset is empty.")
        mapped, detected = self.auto_clean_and_map(raw_df)
        canonical = self._canonical_frame(mapped, detected)
        if canonical["asset_id"].isna().all():
            canonical["asset_id"] = [f"ASSET-{idx:05d}" for idx in range(1, len(canonical) + 1)]
        cleaned = self._vectorized_impute(canonical)
        cleaned = self._derive_columns(cleaned)
        return cleaned.reset_index(drop=True), detected


class AdvancedAI:
    FEATURE_COLUMNS = [
        "energy_kwh",
        "diesel_litres",
        "solar_kwh",
        "water_withdrawal_kl",
        "ewaste_kg",
        "hazardous_waste_kg",
        "procurement_spend_inr",
        "occupancy_percent",
        "outdoor_temp_c",
        "hvac_load_kw",
        "lighting_kwh",
        "Cooling_Delta_C",
        "Occupancy_Adjusted_Load_Ratio",
        "kW_per_TR",
    ]

    def __init__(self, math_engine: DeterministicMath | None = None) -> None:
        self.math_engine = math_engine or DeterministicMath()
        self.anomaly_model = IsolationForest(contamination=0.04, random_state=42, n_estimators=160)
        self.forecaster = GradientBoostingRegressor(random_state=42)
        xgb = _optional_import("xgboost")
        self.explainer_model = xgb.XGBRegressor(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        ) if xgb is not None else GradientBoostingRegressor(random_state=42)

    def train_models(self, trained_master: pd.DataFrame) -> None:
        if trained_master.empty:
            return
        train_df = trained_master.copy().reset_index(drop=True)
        train_df["month_num"] = pd.to_datetime(train_df["month"]).dt.month
        train_x = train_df[self.FEATURE_COLUMNS + ["month_num"]].fillna(0.0)
        self.forecaster.fit(train_x, train_df["energy_kwh"].fillna(0.0))
        self.explainer_model.fit(train_df[self.FEATURE_COLUMNS].fillna(0.0), train_df["Total_tCO2e"].fillna(0.0))

    def _rank_assets(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped = (
            df.groupby("asset_id", as_index=False)
            .agg(
                Energy_Intensity_kWh_per_sqm=("Energy_Intensity_kWh_per_sqm", "mean"),
                Scope2_tCO2e=("Scope2_tCO2e", "sum"),
                Scope3_tCO2e=("Scope3_tCO2e", "sum"),
                Solar_Ratio=("Solar_Integration_Ratio", "mean"),
                Water_Benchmark_Breaches=("Water_Benchmark_Breach", "sum"),
                Anomalies=("Is_Anomaly", "sum"),
            )
            .reset_index(drop=True)
        )
        grouped["ESG_Score"] = grouped.apply(
            lambda row: float(np.clip(100 - (row["Energy_Intensity_kWh_per_sqm"] * 2.5) - (row["Anomalies"] * 0.7) + (row["Solar_Ratio"] * 18) - (row["Scope3_tCO2e"] * 0.02), 0, 100)),
            axis=1,
        )
        grouped["Rank_Label"] = np.where(
            grouped["ESG_Score"] >= grouped["ESG_Score"].quantile(0.8),
            "Best Performing",
            np.where(grouped["ESG_Score"] <= grouped["ESG_Score"].quantile(0.2), "Worst Performing", "Core Performing"),
        )
        grouped["BEE_Rating"] = pd.cut(grouped["Energy_Intensity_kWh_per_sqm"], bins=[-np.inf, 0.45, 0.58, 0.68, 0.80, np.inf], labels=["5 Star", "4 Star", "3 Star", "2 Star", "1 Star"])
        return grouped.sort_values(["ESG_Score", "Solar_Ratio"], ascending=[False, False]).reset_index(drop=True)

    def _detect_extreme_outliers(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for column in ["energy_kwh", "diesel_litres", "water_withdrawal_kl"]:
            z = (df[column] - df[column].mean()) / max(df[column].std(ddof=0), 1e-9)
            flagged = df.index[np.abs(z) > 4.0].tolist()
            if flagged:
                actions.append({"kind": "auto_clean_imputation", "label": f"EXECUTE AUTO-CLEAN IMPUTATION: {column}", "payload": {"column": column, "row_indices": flagged[:20]}})
        return actions

    def _classify_signature(self, df: pd.DataFrame) -> pd.Series:
        refrigerant = df["kW_per_TR"] > 0.85
        sensor = (df["Cooling_Delta_C"] < 2.0) & (df["hvac_load_kw"] > df["hvac_load_kw"].median())
        theft = (df["diesel_litres"] > df["diesel_litres"].median() * 2.5) & (df["generator_output_kw"] <= 0.0)
        return np.select([refrigerant, sensor, theft], ["Refrigerant_Leak_Signature", "HVAC_Sensor_Drift_Signature", "Diesel_Theft_Leakage_Signature"], default="General_Operational_Anomaly")

    def _build_shap_payload(self, df: pd.DataFrame, row: pd.Series) -> dict[str, Any] | None:
        feature_row = row[self.FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_frame().T
        shap = _optional_import("shap")
        if shap is not None:
            try:
                explainer = shap.Explainer(self.explainer_model)
                shap_values = explainer(feature_row)
                values = {feature: float(shap_values.values[0][idx]) for idx, feature in enumerate(self.FEATURE_COLUMNS)}
            except Exception:
                values = {}
        else:
            values = {}
        if not values:
            medians = df[self.FEATURE_COLUMNS].median()
            values = {feature: float(feature_row.iloc[0][feature] - medians[feature]) for feature in self.FEATURE_COLUMNS}
        return {"asset_id": str(row["asset_id"]), "month": pd.to_datetime(row["month"]), "values": dict(sorted(values.items(), key=lambda item: abs(item[1]), reverse=True))}

    @track(project_name="greenlens", emission_level="medium")
    def process(self, calculated_df: pd.DataFrame, detected_columns: dict[str, str], source_name: str) -> ProcessedPortfolio:
        df = calculated_df.copy().reset_index(drop=True)
        features = df[self.FEATURE_COLUMNS].fillna(0.0)
        anomaly_pred = self.anomaly_model.fit_predict(features)
        df["Anomaly_Score"] = -self.anomaly_model.score_samples(features)
        df["Is_Anomaly"] = anomaly_pred == -1
        df["Anomaly_Signature"] = self._classify_signature(df)
        df["ESG_Compliance_Score"] = self.math_engine.calculate_esg_score(df, "energy_kwh")
        anomaly_records = df[df["Is_Anomaly"]].sort_values("Anomaly_Score", ascending=False).reset_index(drop=True)
        latest_shap = self._build_shap_payload(df, anomaly_records.iloc[0]) if not anomaly_records.empty else None
        rankings = self._rank_assets(df)
        summary = {
            "records": int(len(df)),
            "assets": int(df["asset_id"].nunique()),
            "reporting_start": pd.to_datetime(df["month"]).min(),
            "reporting_end": pd.to_datetime(df["month"]).max(),
            "scope1_total": float(df["Scope1_tCO2e"].sum()),
            "scope2_total": float(df["Scope2_tCO2e"].sum()),
            "scope3_total": float(df["Scope3_tCO2e"].sum()),
            "total_carbon": float(df["Total_tCO2e"].sum()),
            "energy_total": float(df["energy_kwh"].sum()),
            "solar_total": float(df["solar_kwh"].sum()),
            "water_total": float(df["water_withdrawal_kl"].sum()),
            "anomaly_count": int(df["Is_Anomaly"].sum()),
            "source_name": source_name,
            "esg_score": float(self.math_engine.calculate_esg_score(df, "energy_kwh")),
            "brsr_principle": 6,
            "outlier_actions": self._detect_extreme_outliers(df),
        }
        return ProcessedPortfolio(df=df, summary=summary, anomaly_records=anomaly_records, latest_shap=latest_shap, asset_rankings=rankings, detected_columns=detected_columns, source_name=source_name, esg_score=summary["esg_score"])

    def build_shap_figure(self, package: ProcessedPortfolio) -> go.Figure | None:
        if package.latest_shap is None:
            return None
        drivers = list(package.latest_shap["values"].items())[:6]
        fig = go.Figure(
            go.Waterfall(
                orientation="v",
                measure=["relative"] * len(drivers) + ["total"],
                x=[name for name, _ in drivers] + ["Net Impact"],
                y=[value for _, value in drivers] + [sum(value for _, value in drivers)],
                connector={"line": {"color": "#6e7681"}},
                increasing={"marker": {"color": "#f85149"}},
                decreasing={"marker": {"color": "#00ff7f"}},
                totals={"marker": {"color": "#58a6ff"}},
            )
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3", margin=dict(l=10, r=10, t=30, b=10), title="SHAP Root-Cause Driver Attribution  |  Source: GreenLens SRD v3.0 (Precision >0.85, Recall >0.80)")
        return fig

    def build_forecast_figure(self, forecast_df: pd.DataFrame) -> go.Figure:
        monthly = forecast_df.groupby("month", as_index=False)[["Forecast_kWh", "Forecast_Scope2_tCO2e"]].sum().reset_index(drop=True)
        fig = go.Figure()
        fig.add_scatter(x=monthly["month"], y=monthly["Forecast_kWh"], mode="lines+markers", line=dict(color="#00ff7f", width=3), name="Forecast kWh")
        fig.add_bar(x=monthly["month"], y=monthly["Forecast_Scope2_tCO2e"], marker_color="#58a6ff", opacity=0.4, name="Forecast Scope 2")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3", margin=dict(l=10, r=10, t=30, b=10), title="12-Month Forecast Outlook", barmode="overlay")
        return fig

    def build_energy_figure(self, package: ProcessedPortfolio) -> go.Figure:
        """Monthly energy consumption — grid + solar (separate bars)."""
        monthly = (
            package.df.assign(mp=package.df["month"].dt.to_period("M").astype(str))
            .groupby("mp", as_index=False)
            .agg(energy_kwh=("energy_kwh", "sum"), solar_kwh=("solar_kwh", "sum"))
            .reset_index(drop=True)
        )
        fig = go.Figure()
        fig.add_bar(x=monthly["mp"], y=monthly["energy_kwh"], name="Grid Energy (kWh)", marker_color="#58a6ff", opacity=0.85)
        fig.add_bar(x=monthly["mp"], y=monthly["solar_kwh"], name="Solar Captured (kWh)", marker_color="#00ff7f", opacity=0.85)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3", margin=dict(l=10, r=10, t=40, b=10),
            title="Monthly Energy Consumption  |  Source: CEA v20 grid factor 0.727 kg CO₂/kWh",
            barmode="group", legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        return fig

    def build_water_figure(self, package: ProcessedPortfolio) -> go.Figure:
        """Monthly water withdrawal vs BEE benchmark band."""
        monthly = (
            package.df.assign(mp=package.df["month"].dt.to_period("M").astype(str))
            .groupby("mp", as_index=False)
            .agg(water_kl=("water_withdrawal_kl", "sum"))
            .reset_index(drop=True)
        )
        fig = go.Figure()
        fig.add_bar(x=monthly["mp"], y=monthly["water_kl"], name="Water Withdrawal (kL)",
                    marker_color="#4dd0e1", opacity=0.85)
        # BEE benchmark band (Bengaluru: 0.8-1.0 kL/m2/yr annualised proxy per record)
        n = len(monthly)
        if n > 0:
            avg = monthly["water_kl"].mean()
            fig.add_scatter(x=monthly["mp"], y=[avg * 0.85] * n, mode="lines",
                name="BEE Lower (Bengaluru 0.8 kL/m²/yr)", line=dict(color="#00ff7f", dash="dot", width=1.5))
            fig.add_scatter(x=monthly["mp"], y=[avg * 1.15] * n, mode="lines",
                name="BEE Upper (Mumbai 1.5 kL/m²/yr)", line=dict(color="#f85149", dash="dot", width=1.5))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3", margin=dict(l=10, r=10, t=40, b=10),
            title="Monthly Water Withdrawal  |  Benchmark: BEE Grade-A Office 0.80–1.50 kL/m²/yr",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        return fig

    def build_trend_figure(self, package: ProcessedPortfolio) -> go.Figure:
        """Legacy alias — returns energy figure only. Use build_energy_figure/build_water_figure directly."""
        return self.build_energy_figure(package)

    def forecast_portfolio(self, df: pd.DataFrame, months: int = 12) -> pd.DataFrame:
        """Vectorised forecast — one batched predict() call total instead of N_assets x months calls."""
        all_rows: list[dict[str, Any]] = []
        meta: list[tuple[str, Any, float]] = []  # (asset_id, future_date, solar_kwh)

        for asset_id, asset_df in df.groupby("asset_id", sort=True):
            history = asset_df.sort_values("month").reset_index(drop=True)
            tail3 = history.tail(3)
            base = {col: float(tail3[col].mean()) for col in self.FEATURE_COLUMNS}
            base_solar = base["solar_kwh"]
            last_month = pd.to_datetime(history["month"]).max()
            future_dates = pd.date_range(last_month + pd.offsets.MonthEnd(1), periods=months, freq="ME")
            for future_date in future_dates:
                month_num = future_date.month
                row = dict(base)
                row["outdoor_temp_c"] = base["outdoor_temp_c"] + np.sin((2 * np.pi * month_num) / 12.0) * 2.0
                row["month_num"] = month_num
                all_rows.append(row)
                meta.append((asset_id, future_date, base_solar))

        if not all_rows:
            return pd.DataFrame(columns=["asset_id", "month", "Forecast_kWh", "Forecast_Scope2_tCO2e"])

        # Single batched predict call — vastly faster than one call per row
        feature_cols = self.FEATURE_COLUMNS + ["month_num"]
        X = pd.DataFrame(all_rows)[feature_cols].fillna(0.0)
        predictions = self.forecaster.predict(X)

        records = []
        for (asset_id, future_date, solar_kwh), pred in zip(meta, predictions):
            pred = max(float(pred), 0.0)
            scope2 = max(pred - solar_kwh, 0.0) * DeterministicMath.CEA_FACTOR / 1000.0
            records.append({"asset_id": asset_id, "month": future_date, "Forecast_kWh": pred, "Forecast_Scope2_tCO2e": scope2})

        return pd.DataFrame(records).sort_values(["month", "asset_id"]).reset_index(drop=True)

    def simulate_scenario(self, package: ProcessedPortfolio, solar_kw: float = 0.0, setpoint_delta_c: float = 0.0) -> tuple[pd.DataFrame, dict[str, float]]:
        simulated = package.df.copy().reset_index(drop=True)
        if solar_kw > 0:
            simulated["solar_kwh"] = simulated["solar_kwh"] + (solar_kw * 4.2 * 30.0)
        if setpoint_delta_c > 0:
            factor = max(0.0, 1 - (0.045 * setpoint_delta_c))
            simulated["hvac_load_kw"] = simulated["hvac_load_kw"] * factor
            simulated["energy_kwh"] = simulated["Base_Load_Estimate_kWh"] + ((simulated["energy_kwh"] - simulated["Base_Load_Estimate_kWh"]) * factor)
            simulated["indoor_temp_c"] = simulated["indoor_temp_c"] + setpoint_delta_c
        simulated = self.math_engine._derive_columns(simulated)
        forecast = self.forecast_portfolio(simulated, months=12)
        energy_delta = float(package.df["energy_kwh"].sum() - simulated["energy_kwh"].sum())
        carbon_delta = float(package.df["Scope2_tCO2e"].sum() - simulated["Scope2_tCO2e"].sum())
        savings = energy_delta * 9.0
        capex = solar_kw * 45_000.0 if solar_kw > 0 else 1.0
        roi = 0.0 if solar_kw <= 0 else (savings / capex) * 100.0
        return forecast, {"annual_energy_delta_kwh": energy_delta, "annual_carbon_delta_tco2e": carbon_delta, "annual_cost_saving_inr": savings, "estimated_roi_pct": roi}


# ── CorporateDossierPDF has been removed. PDF generation is now handled ───────
# ── entirely by greenlens_pdf_engine.py (ReportLab-based).             ───────


class RegulatoryIntelligence:
    def __init__(self, math_engine: DeterministicMath | None = None, ai_engine: AdvancedAI | None = None) -> None:
        self.math_engine = math_engine or DeterministicMath()
        self.ai_engine = ai_engine or AdvancedAI(self.math_engine)
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self._genai = None
        self._openai_cls = None
        self._gemini_ready = bool(self.gemini_key)
        self._openai_ready = bool(self.openai_key)
        if self._gemini_ready:
            try:
                self._genai = _optional_import("google.generativeai")
                if self._genai is None:
                    raise RuntimeError("google.generativeai unavailable")
                self._genai.configure(api_key=self.gemini_key)
            except Exception:
                self._gemini_ready = False
        if self._openai_ready:
            try:
                self._openai_cls = _optional_import("openai", "OpenAI")
                if self._openai_cls is None:
                    raise RuntimeError("OpenAI SDK unavailable")
            except Exception:
                self._openai_ready = False

    @staticmethod
    def _schema_summary(df: pd.DataFrame) -> str:
        return json.dumps({"rows": len(df), "columns": list(df.columns), "sample": df.head(6).to_dict(orient="records")}, default=str, indent=2)

    @staticmethod
    def _rag_lookup(prompt: str) -> str:
        prompt_lower = prompt.lower()
        matches = [doc for doc in RAG_REFERENCE if any(token in doc.lower() for token in prompt_lower.split())]
        return " ".join(matches[:2])

    @staticmethod
    def _cite_if_needed(text: str, prompt: str) -> str:
        if any(token in prompt.lower() for token in ["brsr", "principle 6", "sebi", "cea", "icap", "ipcc"]):
            return text + "\n\n[CITED-RESEARCH] SEBI BRSR Principle 6; CEA Version 20; ICAP/IPCC AR4."
        return text

    def _system_prompt(self, recent_messages: list[dict[str, Any]], rolling_context: str, package: ProcessedPortfolio, prompt: str) -> str:
        # Build a rich, structured system prompt — this is what gets sent to Gemini/OpenAI
        # when an API key is available. It trains the external model on CUESG context.
        top_asset = package.asset_rankings.iloc[0]["asset_id"] if not package.asset_rankings.empty else "N/A"
        worst_asset = package.asset_rankings.iloc[-1]["asset_id"] if not package.asset_rankings.empty else "N/A"
        brsr_alerts = []
        if package.df["Water_Benchmark_Breach"].sum() > 0:
            breach_assets = package.df.loc[package.df["Water_Benchmark_Breach"], "asset_id"].unique().tolist()
            brsr_alerts.append(f"BRSR P6 Warning: Water benchmark exceeded in {len(breach_assets)} assets: {', '.join(breach_assets[:3])}")
        if package.summary["scope1_total"] > package.summary["scope2_total"] * 0.3:
            brsr_alerts.append("BRSR P6 Advisory: Scope 1 (diesel) is unusually high — review DG set utilization.")
        if package.summary["anomaly_count"] > package.summary["records"] * 0.08:
            brsr_alerts.append(f"BRSR SRD Alert: Anomaly rate {package.summary['anomaly_count']/max(package.summary['records'],1)*100:.1f}% exceeds 4% contamination threshold.")

        return f"""You are CUESG — an autonomous Chief Sustainability Officer AI built for Indian commercial real estate ESG compliance.

## CORE MANDATE
Answer from the active dataset first. Every number you cite must come from the live data below.
When no data supports a claim, say so explicitly. Never hallucinate numbers.
Cite SEBI BRSR Principle 6 when giving regulatory guidance.
Cite CEA Version 20 (0.727 kg CO2/kWh) for every Scope 2 calculation you reference.

## ACTIVE PORTFOLIO STATE
- ESG Score: {package.esg_score:.1f}/100
- Assets: {package.summary['assets']} | Records: {package.summary['records']}
- Scope 1: {package.summary['scope1_total']:.2f} tCO2e (Diesel — IPCC AR4 2.68 kg/L)
- Scope 2: {package.summary['scope2_total']:.2f} tCO2e (Grid — CEA v20 0.727 kg/kWh)
- Scope 3: {package.summary['scope3_total']:.2f} tCO2e (SEBI BRSR proxy)
- Total Carbon: {package.summary['total_carbon']:.2f} tCO2e
- Anomalies: {package.summary['anomaly_count']} records flagged (IF contamination=4%, SRD v3.0)
- Best Asset: {top_asset} | Worst Asset: {worst_asset}
- Reporting: {package.summary['reporting_start'].strftime('%b %Y')} → {package.summary['reporting_end'].strftime('%b %Y')}

## LIVE BRSR COMPLIANCE ALERTS
{chr(10).join(brsr_alerts) if brsr_alerts else '- All BRSR P6 thresholds currently within bounds.'}

## REGULATORY CONSTANTS (cite these in every relevant answer)
- Grid: 0.727 kg CO2/kWh (CEA Baseline v20.0, Dec 2024)
- Diesel: 2.68 kg CO2/L (IPCC 2006 AR4, adopted by BEE/MoEFCC)
- Water benchmarks: Bengaluru 0.80-1.00 kL/m2/yr, Mumbai 1.25-1.50 kL/m2/yr (BEE Grade-A)
- EUI baseline: 140-180 kWh/m2/yr (BEE Star Rating commercial)
- Chiller efficiency: 0.65-0.75 kW/TR water-cooled (HVAC Audit Std India)
- R-410A GWP: 2,088 | R-134a GWP: 1,430 (ICAP / IPCC AR4)
- Anomaly precision: >0.85 | Recall: >0.80 | MAPE: <15% (GreenLens SRD v3.0)

## ANSWER CAPABILITIES
When asked about scenarios: run the simulation using the simulate_scenario tool data.
When asked about forecasts: cite the GBR/XGBoost 12-month output.
When asked about anomalies: reference Isolation Forest signatures (Refrigerant_Leak, HVAC_Sensor_Drift, Diesel_Theft).
When asked about compliance: map to SEBI BRSR P6 Essential Indicators EI-1 to EI-10.
When asked about SHAP: explain the top drivers by name and their direction of impact.

## CONVERSATION MEMORY
{rolling_context or '(No prior context)'}

## RAG SUPPORT
{self._rag_lookup(prompt)}

## INDIA ESG METADATA
{json.dumps(GREENLENS_SYSTEM_METADATA, indent=2)[:2000]}

## LIVE DATA SCHEMA
{self._schema_summary(package.df)}

## RECENT MESSAGES
{json.dumps(recent_messages[-6:], default=str, indent=2)}"""

    # ── OCR Confidence Thresholds (CUESG Omni-Parser Blueprint) ─────────────
    _OCR_AUTO_ACCEPT   = 0.90   # Green — auto-ingest
    _OCR_WARN_THRESHOLD = 0.75  # Yellow — flag for review
    # Below 0.75 → HARD STOP (returned with confidence warning)

    @staticmethod
    def _regex_ocr_text(text: str) -> dict[str, Any]:
        """
        Phase 5 deterministic extraction from raw OCR / PDF text.
        Implements the CUESG Omni-Parser confidence + cross-validation pipeline.
        No API key required.
        """
        result: dict[str, Any] = {}
        confidence_scores: dict[str, float] = {}

        # ── energy_kwh ────────────────────────────────────────────────────────
        for pattern in [
            r"(?:total\s+units?|energy\s+consumed?|net\s+kwh|units?\s+consumed?)[^\d]*([0-9][0-9,]*(?:\.[0-9]+)?)\s*kwh",
            r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*kwh",
            r"consumption[^\d]*([0-9][0-9,]*(?:\.[0-9]+)?)",
        ]:
            m = re.search(pattern, text, re.I)
            if m:
                val = float(m.group(1).replace(",", ""))
                # Cross-validation: reject physically impossible values
                if 0 < val < 10_000_000:
                    result["energy_kwh"] = val
                    confidence_scores["energy_kwh"] = 0.92 if "kwh" in pattern else 0.78
                    break

        # ── diesel_litres ─────────────────────────────────────────────────────
        for pattern in [
            r"(?:qty|quantity|litres?|liters?)[^\d]*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:l|ltr|litres?)?",
            r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:ltrs?|litres?|liters?)",
            r"(?:hsd|diesel|fuel)[^\d]*([0-9][0-9,]*(?:\.[0-9]+)?)",
        ]:
            m = re.search(pattern, text, re.I)
            if m:
                val = float(m.group(1).replace(",", ""))
                if 0 < val < 100_000:
                    result["diesel_litres"] = val
                    confidence_scores["diesel_litres"] = 0.88 if "litres" in pattern else 0.75
                    break

        # ── water_withdrawal_kl ───────────────────────────────────────────────
        for pattern in [
            r"(?:consumption|units?|kl|kilolitres?)[^\d]*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:kl|kilo)",
            r"(?:water\s+used?|volume)[^\d]*([0-9][0-9,]*(?:\.[0-9]+)?)",
            r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:kl|kilolitres?|kilo)",
        ]:
            m = re.search(pattern, text, re.I)
            if m:
                val = float(m.group(1).replace(",", ""))
                if 0 < val < 500_000:
                    result["water_withdrawal_kl"] = val
                    confidence_scores["water_withdrawal_kl"] = 0.85
                    break

        # ── utility_cost_inr ──────────────────────────────────────────────────
        for pattern in [
            r"(?:total\s+amount|net\s+payable|amount\s+due|total\s+due|grand\s+total)[^\d]*(?:rs\.?|inr|₹)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            r"(?:rs\.?|inr|₹)\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        ]:
            m = re.search(pattern, text, re.I)
            if m:
                val = float(m.group(1).replace(",", ""))
                if 0 < val < 100_000_000:
                    result["utility_cost_inr"] = val
                    confidence_scores["utility_cost_inr"] = 0.90 if "total" in pattern else 0.78
                    break

        # ── bill_month ────────────────────────────────────────────────────────
        for pattern in [
            r"(?:bill\s+date|invoice\s+date|billing\s+period|for\s+the\s+month)[^\d]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s\-]*\d{2,4})",
        ]:
            m = re.search(pattern, text, re.I)
            if m:
                result["bill_month"] = m.group(1)
                confidence_scores["bill_month"] = 0.88
                break

        # ── vendor_name (fuzzy vendor matching) ───────────────────────────────
        known_vendors = [
            "BESCOM", "TATA POWER", "ADANI ELECTRICITY", "BSES", "MSEDCL",
            "TNEB", "KSEB", "CESC", "WBSEDCL", "JVVNL", "BRPL", "BYPL",
            "BESQ", "RELIANCE ENERGY",
        ]
        text_upper = text.upper()
        for vendor in known_vendors:
            if vendor in text_upper:
                result["vendor_name"] = vendor
                confidence_scores["vendor_name"] = 0.95
                break

        # ── account_id / consumer_number ─────────────────────────────────────
        m = re.search(r"(?:consumer\s*(?:no|number|id)|account\s*(?:no|number|id)|ca\s*no)[^\d]*([A-Z0-9\-]{6,20})", text, re.I)
        if m:
            result["account_id"] = m.group(1)
            confidence_scores["account_id"] = 0.88

        # ── Previous / Current meter readings (Delta Rule cross-validation) ───
        prev_m = re.search(r"(?:previous|prev\.?|opening)\s*(?:reading|meter)[^\d]*([0-9]+(?:\.[0-9]+)?)", text, re.I)
        curr_m = re.search(r"(?:current|present|closing)\s*(?:reading|meter)[^\d]*([0-9]+(?:\.[0-9]+)?)", text, re.I)
        if prev_m and curr_m:
            prev = float(prev_m.group(1))
            curr = float(curr_m.group(1))
            delta = curr - prev
            result["previous_reading"] = prev
            result["current_reading"] = curr
            # Delta Rule: if energy_kwh already extracted, cross-validate
            if "energy_kwh" in result:
                extracted = result["energy_kwh"]
                ratio = abs(extracted - delta) / max(delta, 1.0)
                if ratio < 0.05:
                    # Perfect match — boost confidence
                    confidence_scores["energy_kwh"] = min(confidence_scores.get("energy_kwh", 0.8) + 0.07, 1.0)
                elif ratio > 0.30:
                    # Mismatch — Delta Rule triggered, use calculated value
                    result["energy_kwh"] = delta
                    result["_delta_rule_applied"] = True
                    confidence_scores["energy_kwh"] = 0.82
            else:
                result["energy_kwh"] = delta
                confidence_scores["energy_kwh"] = 0.80

        # ── Net metering (solar import/export) ───────────────────────────────
        imp_m = re.search(r"(?:import|grid\s+import)[^\d]*([0-9][0-9,]*(?:\.[0-9]+)?)\s*kwh", text, re.I)
        exp_m = re.search(r"(?:export|solar\s+export)[^\d]*([0-9][0-9,]*(?:\.[0-9]+)?)\s*kwh", text, re.I)
        if imp_m:
            result["import_kwh"] = float(imp_m.group(1).replace(",", ""))
        if exp_m:
            result["export_kwh"] = float(exp_m.group(1).replace(",", ""))
            result["solar_kwh"] = result.get("export_kwh", 0)

        # ── Overall confidence ────────────────────────────────────────────────
        if confidence_scores:
            avg_conf = sum(confidence_scores.values()) / len(confidence_scores)
        else:
            avg_conf = 0.0
        result["_confidence"] = round(avg_conf, 3)
        result["_confidence_detail"] = confidence_scores
        result["_ocr_engine"] = "CUESG-RegexOCR-v1"

        # Routing logic per CUESG blueprint
        if avg_conf < 0.75:
            result["_status"] = "HARD_STOP — Low confidence. Route to human auditor."
            result["_auto_ingest"] = False
        elif avg_conf < 0.90:
            result["_status"] = "WARN — Accept with yellow flag for monthly audit."
            result["_auto_ingest"] = True
            result["_flag_for_review"] = True
        else:
            result["_status"] = "AUTO-ACCEPT — High confidence."
            result["_auto_ingest"] = True
            result["_flag_for_review"] = False

        return result

    def parse_utility_bill_text(self, text: str) -> dict[str, Any]:
        """Parse any raw text from a utility bill using the CUESG regex OCR engine."""
        return self._regex_ocr_text(text)

    def parse_utility_bill_pdf(self, pdf_bytes: bytes) -> dict[str, Any] | None:
        """
        Extract utility bill data from a PDF.
        Pipeline: pdfplumber (native PDF) → PyMuPDF → raw text fallback.
        Works 100% offline — no API key needed.
        """
        extracted_text = ""

        # Engine 1: pdfplumber (best for native digital PDFs)
        pdfplumber = _optional_import("pdfplumber")
        if pdfplumber is not None:
            try:
                import io as _io
                with pdfplumber.open(_io.BytesIO(pdf_bytes)) as pdf:
                    pages_text = []
                    for page in pdf.pages[:6]:  # Only first 6 pages (summary pages)
                        pt = page.extract_text()
                        if pt:
                            pages_text.append(pt)
                    extracted_text = "\n".join(pages_text)
            except Exception:
                pass

        # Engine 2: PyMuPDF fallback
        if not extracted_text:
            fitz = _optional_import("fitz")  # PyMuPDF
            if fitz is not None:
                try:
                    import io as _io
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    pages_text = [doc[i].get_text() for i in range(min(6, doc.page_count))]
                    extracted_text = "\n".join(pages_text)
                    doc.close()
                except Exception:
                    pass

        # Engine 3: Raw latin-1 decode fallback
        if not extracted_text:
            try:
                raw = pdf_bytes.decode("latin-1", errors="ignore")
                # Extract printable ASCII sequences from binary PDF
                chunks = re.findall(r"[\x20-\x7e]{4,}", raw)
                extracted_text = " ".join(chunks[:300])
            except Exception:
                pass

        if not extracted_text:
            return {"_status": "ERROR — Could not extract text from PDF.", "_confidence": 0.0, "_auto_ingest": False}

        result = self._regex_ocr_text(extracted_text)
        result["_pdf_chars_extracted"] = len(extracted_text)
        result["_source"] = "PDF"
        return result

    def parse_utility_bill_image(self, image_bytes: bytes, mime_type: str) -> dict[str, Any] | None:
        """
        Parse a utility bill image.
        Pipeline: Gemini Vision (if key available) → Regex OCR on extracted text.
        Always returns a result dict — never silently returns None.
        """
        # Engine 1: Gemini Vision (if API key configured)
        if self._gemini_ready and self._genai is not None:
            try:
                model = self._genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content([
                    (
                        "You are a utility bill OCR engine. Parse this bill image and return ONLY a JSON object "
                        "with these fields (omit fields not found): asset_id, bill_month, energy_kwh, "
                        "utility_cost_inr, diesel_litres, water_withdrawal_kl, solar_kwh, vendor_name, "
                        "account_id, previous_reading, current_reading. Return JSON only, no markdown."
                    ),
                    {"mime_type": mime_type, "data": image_bytes},
                ])
                raw = response.text.strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                match = re.search(r"\{.*\}", raw, re.S)
                if match:
                    parsed = json.loads(match.group(0))
                    parsed["_ocr_engine"] = "Gemini-1.5-Flash"
                    parsed["_confidence"] = 0.95
                    parsed["_status"] = "AUTO-ACCEPT — Gemini Vision extraction."
                    parsed["_auto_ingest"] = True
                    parsed["_flag_for_review"] = False
                    return parsed
            except Exception:
                pass  # fall through to regex engine

        # Engine 2: Try to extract text from image using pytesseract
        tesseract_text = ""
        pytesseract = _optional_import("pytesseract")
        if pytesseract is not None:
            try:
                from PIL import Image as PILImage
                import io as _io
                img = PILImage.open(_io.BytesIO(image_bytes))
                tesseract_text = pytesseract.image_to_string(img)
            except Exception:
                pass

        if tesseract_text:
            result = self._regex_ocr_text(tesseract_text)
            result["_ocr_engine"] = "Tesseract+RegexOCR"
            return result

        # Engine 3: Return a structured placeholder that lets the user fill in manually
        return {
            "_status": "MANUAL_ENTRY — No OCR engine available. Please fill in manually.",
            "_confidence": 0.0,
            "_auto_ingest": False,
            "_ocr_engine": "none",
            "energy_kwh": None,
            "diesel_litres": None,
            "water_withdrawal_kl": None,
            "utility_cost_inr": None,
            "bill_month": None,
            "asset_id": None,
        }

    def append_utility_bill_to_df(self, package: ProcessedPortfolio, parsed_bill: dict[str, Any]) -> pd.DataFrame:
        """
        Append an OCR-parsed bill record to the active portfolio dataframe.
        Handles the new rich OCR dict format (with _confidence, _status, etc.).
        """
        template = package.df.iloc[-1].to_dict()

        # asset_id — use parsed value or keep existing
        if parsed_bill.get("asset_id") and str(parsed_bill["asset_id"]).strip() not in ("", "None", "null"):
            template["asset_id"] = str(parsed_bill["asset_id"])

        # month — handle bill_month or month key
        bill_month_raw = parsed_bill.get("bill_month") or parsed_bill.get("month")
        if bill_month_raw:
            parsed_month = pd.to_datetime(str(bill_month_raw), errors="coerce")
            if pd.notna(parsed_month):
                template["month"] = parsed_month
            else:
                template["month"] = pd.to_datetime(package.df["month"]).max() + pd.offsets.MonthEnd(1)
        else:
            template["month"] = pd.to_datetime(package.df["month"]).max() + pd.offsets.MonthEnd(1)

        # Numeric fields — map both old and new OCR field names
        field_map = {
            "energy_kwh": ["energy_kwh", "import_kwh"],
            "utility_cost_inr": ["utility_cost_inr", "total_amount"],
            "diesel_litres": ["diesel_litres", "quantity_litres"],
            "water_withdrawal_kl": ["water_withdrawal_kl", "total_consumption_kl"],
            "solar_kwh": ["solar_kwh", "export_kwh"],
        }
        for canonical, keys in field_map.items():
            for key in keys:
                val = parsed_bill.get(key)
                if val is not None:
                    try:
                        template[canonical] = float(val)
                        break
                    except (ValueError, TypeError):
                        pass

        # Tag the row so it can be identified as OCR-sourced
        if "source_tag" in template:
            template["source_tag"] = "OCR"

        new_row = pd.DataFrame([template])
        return pd.concat([package.df, new_row], ignore_index=True).reset_index(drop=True)

    def apply_natural_language_update(self, prompt: str, package: ProcessedPortfolio) -> tuple[pd.DataFrame | None, str | None]:
        match = re.search(r"(?:set|add|increase|reduce|decrease)\s+([A-Za-z0-9\-]+).*(diesel|energy|kwh|water|solar|waste|hazardous waste).*(?:to|by)?\s*([0-9]+(?:\.[0-9]+)?)", prompt, re.I)
        if not match:
            match = re.search(r"system,\s*add\s*([0-9]+(?:\.[0-9]+)?)\s*(?:liters|litres|kg|kwh)\s+of\s+(diesel|hazardous waste|waste|energy|water|solar).*(?:to)\s*([A-Za-z0-9\-]+)", prompt, re.I)
            if not match:
                return None, None
            value = float(match.group(1))
            metric_text = match.group(2).lower()
            asset_id = match.group(3).upper()
            mode = "add"
        else:
            asset_id = match.group(1).upper()
            metric_text = match.group(2).lower()
            value = float(match.group(3))
            mode = prompt.split()[0].lower()

        metric_map = {"diesel": "diesel_litres", "energy": "energy_kwh", "kwh": "energy_kwh", "water": "water_withdrawal_kl", "solar": "solar_kwh", "waste": "ewaste_kg", "hazardous waste": "hazardous_waste_kg"}
        canonical = next((column for label, column in metric_map.items() if label in metric_text), None)
        if canonical is None:
            return None, None
        mutated = package.df.copy().reset_index(drop=True)
        mask = mutated["asset_id"].astype(str).str.upper() == asset_id
        if not mask.any():
            return None, f"{asset_id} was not found in the active dataset."
        target_idx = mutated.loc[mask, "month"].idxmax()
        if mode == "set" or " set " in f" {prompt.lower()} ":
            mutated.loc[target_idx, canonical] = value
            verb = "set"
        else:
            direction = -1 if any(token in prompt.lower() for token in ["reduce", "decrease"]) else 1
            mutated.loc[target_idx, canonical] = max(float(mutated.loc[target_idx, canonical]) + (direction * value), 0.0)
            verb = "adjusted"
        return mutated, f"[OMNI-SYSTEM] {asset_id} {canonical} {verb}. Updated latest record value is {float(mutated.loc[target_idx, canonical]):.2f}."

    def execute_action(self, action: dict[str, Any], package: ProcessedPortfolio) -> tuple[ProcessedPortfolio, str]:
        mutated = self.math_engine.auto_clean_outliers(package.df, [action])
        recalculated, detected = self.math_engine.calculate(mutated)
        updated = self.ai_engine.process(recalculated, detected, package.source_name)
        return updated, "[OMNI-SYSTEM] Auto-clean imputation executed. Portfolio metrics refreshed."

    @staticmethod
    def _history_grounded_advice(package: ProcessedPortfolio) -> list[str]:
        advice: list[str] = []
        if package.asset_rankings.empty:
            return advice
        worst_asset = str(package.asset_rankings.iloc[-1]["asset_id"])
        worst_history = package.df.loc[package.df["asset_id"] == worst_asset].sort_values("month").reset_index(drop=True)
        if not worst_history.empty:
            diesel_avg = float(worst_history["diesel_litres"].mean())
            hvac_avg = float(worst_history["hvac_load_kw"].mean())
            water_breaches = int(worst_history["Water_Benchmark_Breach"].sum())
            if diesel_avg > package.df["diesel_litres"].median():
                advice.append(f"{worst_asset} has a sustained diesel baseline above the portfolio median. Audit DG dispatch logs and backup runtime before the next reporting cycle.")
            if hvac_avg > package.df["hvac_load_kw"].median():
                advice.append(f"{worst_asset} is carrying elevated HVAC load across its monthly history. Re-tune setpoints and sequencing before adding new equipment.")
            if water_breaches > 0:
                advice.append(f"{worst_asset} breached its climate water benchmark {water_breaches} times. Investigate cooling tower makeup water, leaks, and controls.")
        if package.summary["anomaly_count"] > 0:
            advice.append("Anomalies are active in the historical ledger. Prioritize root-cause review before using the current period as a benchmark narrative.")
        return advice[:4]

    @staticmethod
    def _asset_match(prompt: str, package: ProcessedPortfolio) -> str | None:
        upper_prompt = prompt.upper()
        for asset_id in sorted(package.df["asset_id"].astype(str).unique(), key=len, reverse=True):
            if asset_id.upper() in upper_prompt:
                return asset_id
        token_match = re.search(r"BLD[-\s]?[A-Z]{3}[-\s]?\d+", upper_prompt)
        if token_match:
            token = token_match.group(0).replace(" ", "-")
            if token in package.df["asset_id"].astype(str).str.upper().tolist():
                return token
        return None

    @staticmethod
    def _extract_assets_in_prompt(prompt: str, package: ProcessedPortfolio) -> list[str]:
        upper_prompt = prompt.upper()
        matches = [asset_id for asset_id in package.df["asset_id"].astype(str).unique() if asset_id.upper() in upper_prompt]
        return list(dict.fromkeys(matches))

    @staticmethod
    def _metric_aliases() -> dict[str, list[str]]:
        return {
            "ESG_Score": ["esg", "score", "compliance score"],
            "Energy_Intensity_kWh_per_sqm": ["energy intensity", "intensity", "kwh/sqm", "kwh per sqm", "efficiency"],
            "energy_kwh": ["energy", "electricity", "kwh", "power"],
            "diesel_litres": ["diesel", "fuel", "dg fuel"],
            "solar_kwh": ["solar", "renewable", "offset"],
            "water_withdrawal_kl": ["water", "withdrawal", "kl"],
            "Scope1_tCO2e": ["scope 1", "direct emissions"],
            "Scope2_tCO2e": ["scope 2", "indirect emissions", "grid emissions"],
            "Scope3_tCO2e": ["scope 3", "value chain"],
            "Total_tCO2e": ["total carbon", "total emissions", "carbon footprint"],
            "hvac_load_kw": ["hvac", "cooling", "chiller load"],
            "occupancy_percent": ["occupancy", "utilization"],
            "Anomaly_Score": ["anomaly", "anomaly score", "fault"],
        }

    @classmethod
    def _match_metric_column(cls, prompt: str) -> str | None:
        prompt_lower = prompt.lower()
        best_metric = None
        best_score = 0
        for metric, aliases in cls._metric_aliases().items():
            score = sum(1 for alias in aliases if alias in prompt_lower)
            if score > best_score:
                best_metric = metric
                best_score = score
        return best_metric

    @staticmethod
    def _recent_memory_reply(messages: list[dict[str, Any]], rolling_context: str) -> str:
        recent_users = [str(m.get("content", "")) for m in messages if m.get("role") == "user"]
        if recent_users:
            recent_slice = recent_users[-3:]
            return "[OMNI-SYSTEM] Recent user intents in memory:\n" + "\n".join(f"- {item}" for item in recent_slice)
        if rolling_context:
            lines = [line for line in rolling_context.splitlines() if line.strip()][-3:]
            return "[OMNI-SYSTEM] Compressed memory summary:\n" + "\n".join(lines)
        return "[OMNI-SYSTEM] There is no meaningful prior conversation state yet."

    @staticmethod
    def _format_asset_metric(asset_id: str, row: pd.Series) -> str:
        return (
            f"- {asset_id}: ESG {float(row.get('ESG_Score', 0.0)):.1f}, "
            f"energy intensity {float(row.get('Energy_Intensity_kWh_per_sqm', 0.0)):.3f}, "
            f"Scope 2 {float(row.get('Scope2_tCO2e', 0.0)):.2f} tCO2e, "
            f"anomalies {int(row.get('Anomalies', 0))}"
        )

    def _generic_dataset_answer(self, prompt: str, package: ProcessedPortfolio, recent_messages: list[dict[str, Any]], rolling_context: str) -> tuple[str | None, go.Figure | None]:
        prompt_lower = prompt.lower()
        rankings = package.asset_rankings.reset_index(drop=True)
        metric = self._match_metric_column(prompt)
        mentioned_assets = self._extract_assets_in_prompt(prompt, package)

        if any(token in prompt_lower for token in ["your name", "who are you", "what's your name", "whats your name"]):
            return "CUESG. I am the dataset analyst inside this workspace, built to turn messy ESG data into operational answers, rankings, anomalies, forecasts, and compliance-ready summaries.", None

        if any(token in prompt_lower for token in ["remember", "what did i ask", "previous question", "earlier", "before that"]):
            return self._recent_memory_reply(recent_messages, rolling_context), None

        if any(token in prompt_lower for token in ["columns", "schema", "fields", "what data do we have"]):
            schema = ", ".join(package.df.columns.tolist())
            return f"[DATA-QUERY] Active dataset columns:\n{schema}", None

        if any(token in prompt_lower for token in ["most sustainable", "best building", "best asset", "best performing", "top building"]) and not rankings.empty:
            best = rankings.iloc[0]
            return (
                f"[DATA-QUERY] The most sustainable asset in the active dataset is {best['asset_id']}.\n"
                f"{self._format_asset_metric(str(best['asset_id']), best)}\n"
                f"It leads because it combines stronger ESG score, lower normalized energy intensity, and fewer anomalies than the laggard cohort."
            ), self.ai_engine.build_trend_figure(package)

        if any(token in prompt_lower for token in ["least sustainable", "worst building", "worst asset", "laggard asset"]) and not rankings.empty:
            worst = rankings.iloc[-1]
            return (
                f"[DATA-QUERY] The weakest asset in the active dataset is {worst['asset_id']}.\n"
                f"{self._format_asset_metric(str(worst['asset_id']), worst)}\n"
                f"It is dragging performance through a weaker score, heavier intensity profile, and/or anomaly recurrence."
            ), self.ai_engine.build_trend_figure(package)

        if len(mentioned_assets) >= 2:
            subset = rankings.loc[rankings["asset_id"].isin(mentioned_assets)].reset_index(drop=True)
            if not subset.empty:
                rows = [self._format_asset_metric(str(row.asset_id), row._asdict()) for row in subset.itertuples()]
                return "[DATA-QUERY] Comparison across the mentioned assets:\n" + "\n".join(rows), None

        if metric is not None and any(token in prompt_lower for token in ["top", "highest", "best", "most"]):
            if metric in rankings.columns:
                ascending = metric == "Energy_Intensity_kWh_per_sqm"
                top_rows = rankings.sort_values(metric, ascending=ascending).head(5)
                lines = [f"- {row.asset_id}: {metric} = {float(getattr(row, metric)):.3f}" for row in top_rows.itertuples()]
                return f"[DATA-QUERY] Top assets for {metric}:\n" + "\n".join(lines), None

        if metric is not None and any(token in prompt_lower for token in ["bottom", "lowest", "worst", "least"]):
            if metric in rankings.columns:
                ascending = metric != "Energy_Intensity_kWh_per_sqm"
                bottom_rows = rankings.sort_values(metric, ascending=ascending).head(5)
                lines = [f"- {row.asset_id}: {metric} = {float(getattr(row, metric)):.3f}" for row in bottom_rows.itertuples()]
                return f"[DATA-QUERY] Lowest-performing assets for {metric}:\n" + "\n".join(lines), None

        matched_asset_for_metric = self._asset_match(prompt, package)
        if metric is not None and matched_asset_for_metric is not None:
            asset_df = package.df.loc[package.df["asset_id"].astype(str) == matched_asset_for_metric].sort_values("month").reset_index(drop=True)
            if metric in asset_df.columns:
                latest = float(asset_df.iloc[-1][metric])
                avg = float(asset_df[metric].mean())
                return (
                    f"[DATA-QUERY] {matched_asset_for_metric} for {metric}:\n"
                    f"- Latest value: {latest:.3f}\n"
                    f"- Historical average: {avg:.3f}\n"
                    f"- Minimum: {float(asset_df[metric].min()):.3f}\n"
                    f"- Maximum: {float(asset_df[metric].max()):.3f}"
                ), None

        if any(token in prompt_lower for token in ["insight", "insights", "analyze", "analyse", "summary", "summarise", "summarize"]):
            best = rankings.iloc[0] if not rankings.empty else None
            worst = rankings.iloc[-1] if not rankings.empty else None
            lines = [
                f"Portfolio ESG score is {package.esg_score:.1f}/100 across {package.summary['assets']} assets.",
                f"Scope 2 is the largest emissions block at {package.summary['scope2_total']:.2f} tCO2e.",
            ]
            if best is not None:
                lines.append(f"Best performer is {best['asset_id']} with ESG {best['ESG_Score']:.1f}.")
            if worst is not None:
                lines.append(f"Laggard asset is {worst['asset_id']} with ESG {worst['ESG_Score']:.1f}.")
            if package.summary["anomaly_count"] > 0:
                lines.append(f"There are {package.summary['anomaly_count']} anomaly records worth reviewing before using the period as a benchmark.")
            return "[DATA-QUERY] Portfolio insight summary:\n- " + "\n- ".join(lines), self.ai_engine.build_trend_figure(package)

        return None, None

    def _proactive_brsr_alerts(self, package: ProcessedPortfolio) -> list[str]:
        """
        Autonomous SEBI BRSR Principle 6 compliance scanner.
        Runs on every chat response — surfaces alerts before the user asks.
        Implements the CUESG Omni-Agent Continuous Compliance blueprint.
        """
        alerts: list[str] = []
        df = package.df

        # Water intensity benchmark breach
        breach_count = int(df["Water_Benchmark_Breach"].sum())
        if breach_count > 0:
            breach_assets = df.loc[df["Water_Benchmark_Breach"], "asset_id"].unique().tolist()[:3]
            alerts.append(
                f"⚠ BRSR P6 EI-6 [WATER]: {breach_count} records in {len(breach_assets)} asset(s) "
                f"({', '.join(breach_assets)}) exceed BEE water intensity benchmark "
                f"(Bengaluru: 1.0 kL/m²/yr, Mumbai: 1.5 kL/m²/yr). "
                f"SEBI may require water-stress mitigation disclosure."
            )

        # Scope 1 escalation
        if package.summary["scope1_total"] > 0 and package.summary["total_carbon"] > 0:
            s1_pct = package.summary["scope1_total"] / package.summary["total_carbon"] * 100
            if s1_pct > 25:
                alerts.append(
                    f"⚠ BRSR P6 EI-2 [DIESEL]: Scope 1 is {s1_pct:.1f}% of total carbon "
                    f"({package.summary['scope1_total']:.1f} tCO2e). High DG dependency threatens "
                    f"Scope 1 reduction targets. Factor: 2.68 kg CO2/L (IPCC AR4)."
                )

        # Anomaly rate above SRD threshold
        anomaly_rate = package.summary["anomaly_count"] / max(package.summary["records"], 1)
        if anomaly_rate > 0.06:
            alerts.append(
                f"⚠ SRD v3.0 UC-2 [ANOMALY]: Anomaly rate {anomaly_rate*100:.1f}% exceeds the 4% "
                f"Isolation Forest contamination baseline. {package.summary['anomaly_count']} flagged "
                f"records should be resolved before using this period as an SEBI benchmark."
            )

        # Solar integration low
        solar_ratio = package.summary["solar_total"] / max(package.summary["energy_total"], 1)
        if solar_ratio < 0.05 and package.summary["energy_total"] > 100_000:
            alerts.append(
                f"⚠ BRSR P6 EI-4 [RENEWABLE]: Solar integration ratio is only "
                f"{solar_ratio*100:.1f}%. SEBI BRSR Leadership Indicators reward renewable "
                f"energy disclosure. Consider a solar uplift scenario (+50 kW → +{50*0.727*2400/1000:.0f} tCO2 reduction/yr)."
            )

        # EUI above BEE baseline
        avg_eui = float(df["Energy_Intensity_kWh_per_sqm"].mean()) * 12  # annualise
        if avg_eui > 180:
            alerts.append(
                f"⚠ BEE ECSBC 2024 [EUI]: Annualised portfolio EUI is {avg_eui:.0f} kWh/m²/yr, "
                f"above the BEE Star Rating baseline of 140-180 kWh/m²/yr. "
                f"BEE 5-Star buildings average <140 kWh/m²/yr."
            )

        return alerts

    def _build_brsr_compliance_answer(self, package: ProcessedPortfolio) -> str:
        """Full BRSR P6 compliance status report — triggered on 'brsr', 'compliance', 'sebi' queries."""
        s = package.summary
        df = package.df
        solar_pct = s["solar_total"] / max(s["energy_total"], 1) * 100
        water_breaches = int(df["Water_Benchmark_Breach"].sum())
        avg_eui = float(df["Energy_Intensity_kWh_per_sqm"].mean()) * 12

        lines = [
            "[BRSR P6 COMPLIANCE AUDIT — SEBI Principle 6 Essential Indicators]",
            "",
            f"EI-1  Energy Consumed:     {s['energy_total']:,.0f} kWh  |  Solar offset: {s['solar_total']:,.0f} kWh ({solar_pct:.1f}%)",
            f"EI-2  Scope 1 Emissions:   {s['scope1_total']:.3f} tCO2e  |  Source: Diesel {df['diesel_litres'].sum():,.0f} L × 2.68 kg/L (IPCC AR4)",
            f"EI-3  Scope 2 Emissions:   {s['scope2_total']:.3f} tCO2e  |  Grid × 0.727 kg CO2/kWh (CEA v20 Dec 2024)",
            f"EI-4  Renewable Energy:    {solar_pct:.1f}% of total — {'✓ Disclosed' if solar_pct > 0 else '✗ No renewable data'}",
            f"EI-5  Energy Intensity:    {avg_eui:.1f} kWh/m²/yr annualised  |  BEE baseline: 140-180",
            f"EI-6  Water Withdrawal:    {s['water_total']:,.1f} kL total  |  Benchmark breaches: {water_breaches} records",
            f"EI-7  Scope 3 (Proxy):     {s['scope3_total']:.3f} tCO2e  |  Water + E-waste + Hazwaste + Procurement",
            f"EI-8  Waste Generated:     {df['ewaste_kg'].sum():,.1f} kg e-waste, {df['hazardous_waste_kg'].sum():,.1f} kg hazardous",
            f"EI-9  Anomalies:           {s['anomaly_count']} records (IF contamination=4%, SRD v3.0 UC-2)",
            f"EI-10 CUESG ESG Score:     {package.esg_score:.1f}/100",
            "",
            "LEADERSHIP INDICATORS:",
            f"  - Scope 3 as % of total: {s['scope3_total']/max(s['total_carbon'],1)*100:.1f}% (SEBI target: disclose if >15%)",
            f"  - YoY intensity reduction: Requires 2+ years of data to compute",
            "",
            f"FRAMEWORK: SEBI BRSR Principle 6 (May 2021) — Mandatory for Top 1000 listed companies",
            f"EMISSION CONSTANTS: CEA v20 (Dec 2024) | IPCC AR4 | ICAP | BEE ECSBC 2024",
        ]
        return "\n".join(lines)

    def _build_anomaly_story(self, package: ProcessedPortfolio) -> str:
        """Root-cause storytelling for anomalies — cross-referenced with physics signatures."""
        if package.anomaly_records.empty:
            return "[XAI-ALERT] No anomaly records in this dataset. Portfolio is operating within normal parameters."

        top = package.anomaly_records.head(3)
        lines = [f"[XAI-ALERT] {package.summary['anomaly_count']} anomaly records detected by Isolation Forest (contamination=4%, SRD v3.0 UC-2)."]
        lines.append("")

        for _, row in top.iterrows():
            sig = str(row.get("Anomaly_Signature", "General_Operational_Anomaly"))
            asset = str(row.get("asset_id", "UNKNOWN"))
            month = pd.to_datetime(row["month"]).strftime("%b %Y") if "month" in row.index else "N/A"
            score = float(row.get("Anomaly_Score", 0))
            energy = float(row.get("energy_kwh", 0))
            diesel = float(row.get("diesel_litres", 0))

            if "Refrigerant" in sig:
                explanation = (
                    f"Positive residual in Energy vs CDD regression + elevated kW/TR. "
                    f"Root cause: likely refrigerant leak or chiller degradation. "
                    f"Threshold: >12% energy spike vs baseline (GreenLens SRD v3.0). "
                    f"Action: inspect chiller refrigerant charge and compressor efficiency."
                )
            elif "Sensor" in sig or "HVAC" in sig:
                explanation = (
                    f"x-intercept shift in Energy vs Outdoor Temp correlation. "
                    f"Root cause: likely BMS sensor drift or AHU calibration failure. "
                    f"Action: recalibrate return-air and supply-air sensors (ASHRAE Guideline 14)."
                )
            elif "Theft" in sig or "Diesel" in sig:
                explanation = (
                    f"Diesel consumption {diesel:,.0f} L with low/zero generator output. "
                    f"Root cause: likely fuel pilferage (dV/dt > 0.5 L/hr at idle, SRD v3.0). "
                    f"Action: install digital flow meters and lock tanks."
                )
            else:
                explanation = (
                    f"General operational anomaly — {energy:,.0f} kWh consumption pattern "
                    f"deviates from Isolation Forest baseline. Review operational logs for {month}."
                )

            lines.append(f"Asset: {asset}  |  Month: {month}  |  Severity: {score:.3f}")
            lines.append(f"Signature: {sig}")
            lines.append(f"Analysis: {explanation}")
            lines.append("")

        if package.latest_shap is not None:
            top_drivers = list(package.latest_shap["values"].items())[:4]
            driver_str = " | ".join(f"{k}: {v:+.3f}" for k, v in top_drivers)
            lines.append(f"[SHAP ATTRIBUTION] Top drivers in highest-severity anomaly: {driver_str}")

        return "\n".join(lines)

    def _build_capex_recommendation(self, package: ProcessedPortfolio) -> str:
        """Automated CapEx strategy — CUESG Omni-Agent blueprint."""
        s = package.summary
        rankings = package.asset_rankings
        if rankings.empty:
            return "[ADVISORY] No asset data available for CapEx analysis."

        worst = rankings.iloc[-1]
        worst_eui = float(worst.get("Energy_Intensity_kWh_per_sqm", 0.5)) * 12
        worst_asset = str(worst["asset_id"])
        total_gap_tco2 = max(s["scope2_total"] - s["scope2_total"] * 0.80, 0)

        # Solar ROI calculation
        solar_kw_needed = max(total_gap_tco2 * 1000 / (0.727 * 2400), 50)
        solar_capex = solar_kw_needed * 45_000
        annual_savings = solar_kw_needed * 2400 * 9.0  # INR 9/kWh
        payback_yrs = solar_capex / max(annual_savings, 1)
        scope2_drop = solar_kw_needed * 2400 * 0.727 / 1000

        lines = [
            "[ADVISORY] CapEx Strategy — Automated Remediation Analysis",
            "",
            f"Current total carbon: {s['total_carbon']:.1f} tCO2e across {s['assets']} assets.",
            f"Worst performer: {worst_asset} (EUI: {worst_eui:.0f} kWh/m²/yr vs BEE baseline 140-180).",
            "",
            "RECOMMENDATION 1 — Solar PV Expansion:",
            f"  Target: {solar_kw_needed:.0f} kW of rooftop solar across portfolio",
            f"  Estimated CapEx: ₹{solar_capex/100000:.1f} Lakhs (@ ₹45,000/kW installed)",
            f"  Annual energy savings: {solar_kw_needed*2400:,.0f} kWh",
            f"  Annual cost savings: ₹{annual_savings/100000:.1f} Lakhs (@ ₹9/kWh)",
            f"  Scope 2 reduction: {scope2_drop:.1f} tCO2e/yr (CEA v20: 0.727 kg CO2/kWh)",
            f"  Payback period: {payback_yrs:.1f} years",
            "",
            "RECOMMENDATION 2 — HVAC Chiller Optimisation:",
            f"  Focus: {worst_asset} — energy intensity above BEE 2-Star threshold.",
            f"  Action: Re-calibrate chiller sequencing, deploy VFDs on secondary CHW pumps.",
            f"  Expected reduction: 8-15% of HVAC load (HVAC Audit Std India).",
            f"  Estimated saving: {s['energy_total']*0.10:.0f} kWh/yr → {s['scope2_total']*0.10:.2f} tCO2e/yr",
            "",
            "RECOMMENDATION 3 — BMS Sensor Calibration:",
            f"  {package.summary['anomaly_count']} anomaly records suggest sensor drift or operational faults.",
            f"  Action: ASHRAE Guideline 14 re-commissioning on flagged assets.",
            "",
            "BRSR COMPLIANCE NOTE: Implementing Recommendations 1+2 is expected to improve",
            f"your CUESG ESG Score from {package.esg_score:.1f} to approximately {min(package.esg_score+12, 100):.1f}/100.",
        ]
        return "\n".join(lines)

    def _fallback_response(self, prompt: str, package: ProcessedPortfolio, recent_messages: list[dict[str, Any]] | None = None, rolling_context: str = "") -> dict[str, Any]:
        prompt_lower = prompt.lower()
        content = ""
        plotly_fig = None
        actions = package.summary.get("outlier_actions", [])[:2]
        advice = self._history_grounded_advice(package)
        asset_match = self._asset_match(prompt, package)

        # ── Generic dataset query handler (rankings, comparisons, metrics) ────
        generic_content, generic_fig = self._generic_dataset_answer(prompt, package, recent_messages or [], rolling_context)
        if generic_content is not None:
            content = generic_content
            plotly_fig = generic_fig

        # ── Specialised intent routing ─────────────────────────────────────────
        if not content:

            # BRSR / SEBI compliance audit
            if any(t in prompt_lower for t in ["brsr", "sebi", "compliance", "principle 6", "mandatory", "disclosure"]):
                content = self._build_brsr_compliance_answer(package)

            # Anomaly root-cause storytelling
            elif any(t in prompt_lower for t in ["anomaly", "anomalies", "root cause", "shap", "why did", "spike", "fault", "signature", "refrigerant", "sensor drift", "diesel theft"]):
                content = self._build_anomaly_story(package)
                plotly_fig = self.ai_engine.build_shap_figure(package)

            # CapEx / recommendations / mitigation
            elif any(t in prompt_lower for t in ["capex", "invest", "recommend", "fix", "remediate", "mitigation", "improve score", "reduce carbon", "what should we do", "action plan", "budget"]):
                content = self._build_capex_recommendation(package)

            # Forecast
            elif any(t in prompt_lower for t in ["forecast", "predict", "next 12", "next year", "future", "trend"]):
                forecast = self.ai_engine.forecast_portfolio(package.df, 12)
                plotly_fig = self.ai_engine.build_forecast_figure(forecast)
                monthly_f = forecast.groupby("month")[["Forecast_kWh", "Forecast_Scope2_tCO2e"]].sum()
                peak = monthly_f["Forecast_kWh"].idxmax()
                peak_kwh = float(monthly_f.loc[peak, "Forecast_kWh"])
                total_forecast_s2 = float(monthly_f["Forecast_Scope2_tCO2e"].sum())
                content = (
                    f"[DATA-QUERY] 12-month GBR/XGBoost forecast (MAPE <15%, SRD v3.0 UC-3).\n\n"
                    f"Peak month: {peak.strftime('%b %Y')} at {peak_kwh:,.0f} kWh (seasonal cooling load).\n"
                    f"Forecast total Scope 2 (12 months): {total_forecast_s2:.2f} tCO2e (CEA v20: 0.727 kg/kWh).\n"
                    f"Baseline used: rolling 3-month asset average with seasonal sinusoidal correction.\n"
                    f"To reduce forecast Scope 2, run a solar/setpoint simulation or ask for a CapEx strategy."
                )

            # Scenario simulation (what-if)
            elif any(t in prompt_lower for t in ["solar", "what if", "simulate", "scenario", "install", "setpoint", "hvac reduction", "what happens"]):
                sim_req = self._parse_simulation_request(prompt)
                if sim_req is not None:
                    solar_kw, setpoint_delta = sim_req
                    forecast, metrics = self.ai_engine.simulate_scenario(package, solar_kw=solar_kw, setpoint_delta_c=setpoint_delta)
                    plotly_fig = self.ai_engine.build_forecast_figure(forecast)
                    content = (
                        f"[OMNI-SYSTEM] Monte Carlo scenario simulation completed.\n\n"
                        f"Inputs:  Solar +{solar_kw:.0f} kW  |  HVAC setpoint -{setpoint_delta:.1f}°C\n"
                        f"Annual energy delta:    {metrics['annual_energy_delta_kwh']:+,.0f} kWh\n"
                        f"Annual Scope 2 delta:   {metrics['annual_carbon_delta_tco2e']:+.2f} tCO2e  (CEA v20: 0.727 kg/kWh)\n"
                        f"Annual cost saving:     ₹{metrics['annual_cost_saving_inr']:,.0f}\n"
                        f"Estimated ROI:          {metrics['estimated_roi_pct']:.1f}%\n\n"
                        f"[CITED-RESEARCH] Scope 2 calculation: CEA CO2 Baseline v20.0 (Dec 2024)\n"
                        f"Solar CapEx assumption: ₹45,000/kW installed (Indian market FY24)"
                    )
                else:
                    content = (
                        "[OMNI-SYSTEM] To run a simulation, specify solar kW and/or HVAC setpoint reduction.\n"
                        "Examples:\n"
                        "  'What happens if we install 100kW of solar next month?'\n"
                        "  'Simulate 50kW solar + reduce HVAC setpoint by 2°C'\n"
                        "  'What if we add 200kW solar?'"
                    )

            # Scope-level emissions deep dive
            elif any(t in prompt_lower for t in ["scope 1", "scope 2", "scope 3", "emission", "carbon", "tco2", "ghg"]):
                s = package.summary
                solar_pct = s["solar_total"] / max(s["energy_total"], 1) * 100
                content = (
                    f"[DATA-QUERY] GHG Emissions — SEBI BRSR P6 + CEA v20 + IPCC AR4\n\n"
                    f"Scope 1 (Direct):   {s['scope1_total']:>10.3f} tCO2e\n"
                    f"  Source: {package.df['diesel_litres'].sum():,.0f} L diesel × 2.68 kg CO2/L (IPCC AR4)\n\n"
                    f"Scope 2 (Indirect): {s['scope2_total']:>10.3f} tCO2e\n"
                    f"  Source: ({s['energy_total']:,.0f} kWh grid − {s['solar_total']:,.0f} kWh solar) × 0.727 kg/kWh (CEA v20)\n"
                    f"  Solar offset: {solar_pct:.1f}% of consumption\n\n"
                    f"Scope 3 (Proxy):    {s['scope3_total']:>10.3f} tCO2e\n"
                    f"  Source: Water × 0.000344 + E-waste + Hazwaste + Procurement spend (SEBI BRSR P6)\n\n"
                    f"TOTAL:              {s['total_carbon']:>10.3f} tCO2e\n\n"
                    f"[CITED-RESEARCH] SEBI BRSR Principle 6; CEA Version 20; IPCC AR4."
                )

            # Water analysis
            elif any(t in prompt_lower for t in ["water", "kl", "withdrawal", "tanker"]):
                df = package.df
                breaches = df.loc[df["Water_Benchmark_Breach"]].groupby("asset_id").size().sort_values(ascending=False)
                total_kl = package.summary["water_total"]
                breach_list = [f"  - {aid}: {int(cnt)} periods above benchmark" for aid, cnt in breaches.head(5).items()] or ["  - No breaches detected."]
                content = (
                    f"[DATA-QUERY] Water Withdrawal Analysis — BRSR P6 EI-6\n\n"
                    f"Total withdrawal: {total_kl:,.1f} kL across {package.summary['assets']} assets.\n"
                    f"Scope 3 water contribution: {total_kl * 0.000344:.3f} tCO2e\n\n"
                    f"BEE Benchmarks (Grade-A Offices):\n"
                    f"  Bengaluru (Temperate): 0.80 – 1.00 kL/m²/yr\n"
                    f"  Mumbai (Warm-Humid):   1.25 – 1.50 kL/m²/yr\n\n"
                    f"Benchmark breaches ({int(df['Water_Benchmark_Breach'].sum())} records total):\n"
                    + "\n".join(breach_list) + "\n\n"
                    f"[CITED-RESEARCH] BEE Performance Benchmarking Grade-A Offices; SEBI BRSR P6."
                )
                plotly_fig = self.ai_engine.build_water_figure(package)

            # Monthly / ledger
            elif any(t in prompt_lower for t in ["monthly", "ledger", "month by month", "period"]):
                monthly = (
                    package.df.assign(month_period=package.df["month"].dt.to_period("M").astype(str))
                    .groupby("month_period", as_index=False)
                    .agg(energy_kwh=("energy_kwh", "sum"), scope2=("Scope2_tCO2e", "sum"), solar=("solar_kwh", "sum"), anomalies=("Is_Anomaly", "sum"))
                    .reset_index(drop=True)
                )
                rows = [
                    f"  {row.month_period}  |  {row.energy_kwh:>10,.0f} kWh  |  Scope2: {row.scope2:>6.2f} tCO2e  |  Solar: {row.solar:>8,.0f} kWh  |  Anomalies: {int(row.anomalies)}"
                    for row in monthly.head(18).itertuples()
                ]
                content = "[DATA-QUERY] Monthly Portfolio Ledger:\n\n" + "\n".join(rows)
                plotly_fig = self.ai_engine.build_energy_figure(package)

            # Named asset deep-dive
            elif asset_match is not None:
                asset_df = package.df.loc[package.df["asset_id"].astype(str) == asset_match].sort_values("month").reset_index(drop=True)
                latest = asset_df.iloc[-1]
                avg_energy = float(asset_df["energy_kwh"].mean())
                trend = "↑" if float(latest["energy_kwh"]) > avg_energy else "↓"
                rank_row = package.asset_rankings.loc[package.asset_rankings["asset_id"] == asset_match]
                esg_score = float(rank_row["ESG_Score"].iloc[0]) if not rank_row.empty else 0.0
                bee = str(rank_row["BEE_Rating"].iloc[0]) if not rank_row.empty else "N/A"
                content = (
                    f"[DATA-QUERY] {asset_match} — Operational Intelligence\n\n"
                    f"ESG Score:         {esg_score:.1f}/100  |  BEE Rating: {bee}\n"
                    f"Energy (latest):   {float(latest['energy_kwh']):,.0f} kWh {trend} vs avg {avg_energy:,.0f} kWh\n"
                    f"Solar offset:      {float(latest['solar_kwh']):,.0f} kWh ({float(latest['solar_kwh'])/max(float(latest['energy_kwh']),1)*100:.1f}%)\n"
                    f"Diesel:            {float(latest['diesel_litres']):,.0f} L  → Scope 1: {float(latest['Scope1_tCO2e']):.3f} tCO2e\n"
                    f"Grid energy:       {max(float(latest['energy_kwh'])-float(latest['solar_kwh']),0):,.0f} kWh  → Scope 2: {float(latest['Scope2_tCO2e']):.3f} tCO2e\n"
                    f"Water:             {float(latest['water_withdrawal_kl']):,.1f} kL  |  Benchmark breach: {'Yes ⚠' if bool(latest.get('Water_Benchmark_Breach', False)) else 'No ✓'}\n"
                    f"Energy intensity:  {float(latest['Energy_Intensity_kWh_per_sqm']):.4f} kWh/m²  (BEE 5-Star: <0.45)\n"
                    f"HVAC kW/TR:        {float(latest.get('kW_per_TR', 0.70)):.3f}  (Benchmark: 0.65-0.75, HVAC Audit Std India)\n"
                    f"Anomaly status:    {'🔴 FLAGGED — review SHAP drivers' if bool(latest.get('Is_Anomaly', False)) else '🟢 NORMAL'}"
                )

            # Default — full portfolio summary with proactive alerts
            else:
                s = package.summary
                best = package.asset_rankings.iloc[0]["asset_id"] if not package.asset_rankings.empty else "N/A"
                worst = package.asset_rankings.iloc[-1]["asset_id"] if not package.asset_rankings.empty else "N/A"
                content = (
                    f"[OMNI-SYSTEM] CUESG Portfolio Intelligence — {s['assets']} assets | {s['records']} records\n\n"
                    f"ESG Score:    {package.esg_score:.1f}/100\n"
                    f"Total Carbon: {s['total_carbon']:.1f} tCO2e  (S1: {s['scope1_total']:.1f} | S2: {s['scope2_total']:.1f} | S3: {s['scope3_total']:.1f})\n"
                    f"Best Asset:   {best}  |  Worst Asset: {worst}\n"
                    f"Anomalies:    {s['anomaly_count']} records flagged  (Isolation Forest contamination=4%)\n\n"
                    "[ADVISORY] Try asking:\n"
                    "  • 'Run a full BRSR compliance audit'\n"
                    "  • 'Explain the anomalies and root causes'\n"
                    "  • 'What CapEx investments will improve our score?'\n"
                    "  • 'What happens if we install 100kW solar?'\n"
                    "  • 'Show me the 12-month forecast'\n"
                    "  • 'Compare top and bottom buildings'\n"
                    "  • 'Export the compliance dossier'"
                )

        # ── Proactive BRSR alerts (appended to every substantive response) ────
        brsr_alerts = self._proactive_brsr_alerts(package)
        if brsr_alerts:
            alert_block = "\n\n[OMNI-SYSTEM AUTONOMOUS ALERT]\n" + "\n".join(brsr_alerts[:2])
            content += alert_block

        # ── Advisory suffix ───────────────────────────────────────────────────
        if advice and any(t in prompt_lower for t in ["advice", "recommend", "improve", "fix", "mitigation", "why", "drop", "laggard", "insight", "analyze", "analyse", "summary"]):
            content += "\n[ADVISORY] " + " | ".join(advice[:2])

        # ── Outlier action buttons ─────────────────────────────────────────────
        if actions and any(t in prompt_lower for t in ["anomaly", "outlier", "clean", "impute", "score drop", "root cause", "fix", "repair"]):
            content += "\n[XAI-ALERT] Extreme outlier signatures detected. Use the action buttons below to auto-impute."

        return {"role": "assistant", "content": self._cite_if_needed(content, prompt), "plotly_fig": plotly_fig, "actions": actions}

    def _parse_simulation_request(self, prompt: str) -> tuple[float, float] | None:
        prompt_lower = prompt.lower()
        solar_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*kw\s+of\s+solar", prompt_lower)
        setpoint_match = re.search(r"(?:reduce|lower).*(?:setpoint|set point).*(?:by)?\s*([0-9]+(?:\.[0-9]+)?)\s*°?c", prompt_lower)
        solar_kw = float(solar_match.group(1)) if solar_match else 0.0
        setpoint_delta = float(setpoint_match.group(1)) if setpoint_match else 0.0
        if solar_kw == 0.0 and setpoint_delta == 0.0:
            return None
        return solar_kw, setpoint_delta

    def chat_agent(self, messages: list[dict[str, Any]], package: ProcessedPortfolio, rolling_context: str = "") -> tuple[dict[str, Any], ProcessedPortfolio, str]:
        prompt = str(messages[-1]["content"]).strip()
        rolling_context, recent_messages = MemoryManager.compress(messages, rolling_context)

        mutated_df, note = self.apply_natural_language_update(prompt, package)
        if mutated_df is not None:
            recalculated, detected = self.math_engine.calculate(mutated_df)
            updated = self.ai_engine.process(recalculated, detected, package.source_name)
            return {"role": "assistant", "content": note + f"\n[OMNI-SYSTEM] Updated portfolio ESG score: {updated.esg_score:.1f}/100."}, updated, rolling_context

        simulation_request = self._parse_simulation_request(prompt)
        if simulation_request is not None:
            solar_kw, setpoint_delta = simulation_request
            forecast, metrics = self.ai_engine.simulate_scenario(package, solar_kw=solar_kw, setpoint_delta_c=setpoint_delta)
            return {
                "role": "assistant",
                "content": (
                    "[OMNI-SYSTEM] Scenario simulation completed.\n"
                    f"- Solar addition: {solar_kw:.1f} kW\n"
                    f"- HVAC setpoint shift: {setpoint_delta:.1f} C\n"
                    f"- Annual energy delta: {metrics['annual_energy_delta_kwh']:.2f} kWh\n"
                    f"- Annual Scope 2 delta: {metrics['annual_carbon_delta_tco2e']:.2f} tCO2e\n"
                    f"- Estimated annual savings: INR {metrics['annual_cost_saving_inr']:,.2f}\n"
                    f"- Estimated ROI: {metrics['estimated_roi_pct']:.2f}%"
                ),
                "plotly_fig": self.ai_engine.build_forecast_figure(forecast),
            }, package, rolling_context

        system_prompt = self._system_prompt(recent_messages, rolling_context, package, prompt)
        if self._gemini_ready and self._genai is not None:
            try:
                model = self._genai.GenerativeModel(model_name="gemini-1.5-pro", system_instruction=system_prompt)
                history = [{"role": "user" if m["role"] == "user" else "model", "parts": [str(m["content"])]} for m in recent_messages[:-1]]
                chat = model.start_chat(history=history)
                reply = chat.send_message(prompt)
                if reply.text.strip():
                    return {"role": "assistant", "content": self._cite_if_needed(reply.text.strip(), prompt)}, package, rolling_context
            except Exception:
                pass
        if self._openai_ready and self._openai_cls is not None:
            try:
                client = self._openai_cls(api_key=self.openai_key)
                reply = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_prompt}] + [{"role": m["role"], "content": str(m["content"])} for m in recent_messages],
                    temperature=0.15,
                )
                content = reply.choices[0].message.content or ""
                if content.strip():
                    return {"role": "assistant", "content": self._cite_if_needed(content.strip(), prompt)}, package, rolling_context
            except Exception:
                pass
        return self._fallback_response(prompt, package, recent_messages, rolling_context), package, rolling_context

    def build_markdown_report(self, package: ProcessedPortfolio, forecast_df: pd.DataFrame, chat_messages: list[dict[str, Any]] | None = None, rolling_context: str = "") -> str:
        rankings = package.asset_rankings.reset_index(drop=True)
        history = (
            package.df.assign(month_period=package.df["month"].dt.to_period("M").astype(str))
            .groupby("month_period", as_index=False)
            .agg(energy_kwh=("energy_kwh", "sum"), scope1=("Scope1_tCO2e", "sum"), scope2=("Scope2_tCO2e", "sum"), scope3=("Scope3_tCO2e", "sum"), anomalies=("Is_Anomaly", "sum"))
            .reset_index(drop=True)
        )
        anomalies = package.anomaly_records.head(12).reset_index(drop=True)
        forecast_monthly = forecast_df.groupby("month", as_index=False)[["Forecast_kWh", "Forecast_Scope2_tCO2e"]].sum().reset_index(drop=True)
        top_table = "\n".join(f"| {row.asset_id} | {row.Rank_Label} | {row.ESG_Score:.1f} | {row.BEE_Rating} | {int(row.Anomalies)} |" for row in rankings.head(10).itertuples()) or "| None | None | 0.0 | None | 0 |"
        ledger = "\n".join(f"| {row.month_period} | {row.energy_kwh:,.2f} | {row.scope1:.2f} | {row.scope2:.2f} | {row.scope3:.2f} | {int(row.anomalies)} |" for row in history.itertuples())
        anomaly_table = "\n".join(f"| {pd.to_datetime(row.month).strftime('%Y-%m')} | {row.asset_id} | {row.Anomaly_Signature} | {row.energy_kwh:,.2f} | {row.water_withdrawal_kl:,.2f} | {row.diesel_litres:,.2f} |" for row in anomalies.itertuples()) or "| None | None | Normal | 0.00 | 0.00 | 0.00 |"
        forecast_rows = "\n".join(f"| {row.month.strftime('%Y-%m')} | {row.Forecast_kWh:,.2f} | {row.Forecast_Scope2_tCO2e:.2f} |" for row in forecast_monthly.itertuples())
        shap_text = "\n".join(f"- {name}: {value:+.3f}" for name, value in (package.latest_shap["values"].items() if package.latest_shap else [])) or "- No SHAP payload available."
        chat_log = "\n".join(f"- [{m.get('role', 'assistant').upper()}] {m.get('content', '')}" for m in (chat_messages or [])) or "- No chat log."
        return f"""# CUESG ESG INTELLIGENCE DOSSIER

## 1. METADATA
| Field | Value |
| --- | --- |
| Source File | {package.source_name} |
| Reporting Period | {package.summary['reporting_start'].strftime('%Y-%m-%d')} to {package.summary['reporting_end'].strftime('%Y-%m-%d')} |
| Assets | {package.summary['assets']} |
| Records | {package.summary['records']} |
| Framework | SEBI BRSR Principle 6 |
| Scope 2 Benchmark | CEA factor {DeterministicMath.CEA_FACTOR:.3f} kg CO2/kWh |

## 2. EXECUTIVE SUMMARY
- CUESG ESG Score: {package.esg_score:.1f}/100
- Total Scope 1: {package.summary['scope1_total']:.2f} tCO2e
- Total Scope 2: {package.summary['scope2_total']:.2f} tCO2e
- Total Scope 3: {package.summary['scope3_total']:.2f} tCO2e
- Total Carbon: {package.summary['total_carbon']:.2f} tCO2e

## 3. ASSET LEADERBOARD
| Asset | Category | ESG Score | BEE Rating | Anomalies |
| --- | --- | --- | --- | --- |
{top_table}

## 4. MONTHLY PORTFOLIO LEDGER
| Month | Energy kWh | Scope 1 | Scope 2 | Scope 3 | Anomalies |
| --- | --- | --- | --- | --- | --- |
{ledger}

## 5. ANOMALY REVIEW
| Month | Asset | Signature | Energy kWh | Water kL | Diesel L |
| --- | --- | --- | --- | --- | --- |
{anomaly_table}

## 6. SHAP ROOT-CAUSE SNAPSHOT
{shap_text}

## 7. FORECAST OUTLOOK
| Forecast Month | Forecast Energy kWh | Forecast Scope 2 tCO2e |
| --- | --- | --- |
{forecast_rows}

## 8. ROLLING MEMORY SUMMARY
{rolling_context or "- No prior compressed context."}

## 9. CHAT AUDIT LOG
{chat_log}
"""

    def build_pdf_report(
        self,
        dossier_md: str,
        figures: dict[str, go.Figure],
        chat_messages: list[dict[str, Any]],
        rolling_context: str = "",
        package: ProcessedPortfolio | None = None,
        forecast_df: Any = None,
    ) -> bytes:
        """
        Generate the compliance PDF dossier.

        Uses greenlens_pdf_engine.build_pdf_report_v2 (ReportLab) when available.
        Falls back to the built-in plain-text PDF builder if the engine is missing.
        """
        if package is None:
            return self._build_plaintext_pdf(dossier_md)

        if _PDF_ENGINE_AVAILABLE and _build_pdf_report_v2 is not None:
            import inspect as _inspect
            _kwargs: dict = dict(
                dossier_md=dossier_md,
                figures=figures,
                chat_messages=chat_messages,
                rolling_context=rolling_context,
                package=package,
                reg_engine=self,
                ai_engine=self.ai_engine,
                forecast_df=forecast_df,
            )
            if "mitigation_recs" in _inspect.signature(_build_pdf_report_v2).parameters:
                _kwargs["mitigation_recs"] = list(GREENLENS_SYSTEM_METADATA.get("Mitigation_Recommendations", []))
            return _build_pdf_report_v2(**_kwargs)

        # greenlens_pdf_engine not installed — use plaintext fallback
        return self._build_plaintext_pdf(dossier_md)

    @staticmethod
    def _build_plaintext_pdf(text: str) -> bytes:
        def esc(value: str) -> str:
            return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        raw_lines = [line if line.strip() else " " for line in normalized.split("\n")]
        wrapped_lines: list[str] = []
        for line in raw_lines:
            while len(line) > 95:
                split_at = line.rfind(" ", 0, 95)
                split_at = split_at if split_at > 20 else 95
                wrapped_lines.append(line[:split_at])
                line = line[split_at:].lstrip()
            wrapped_lines.append(line)
        if not wrapped_lines:
            wrapped_lines = ["CUESG dossier export"]

        lines_per_page = 48
        pages = [wrapped_lines[i:i + lines_per_page] for i in range(0, len(wrapped_lines), lines_per_page)]
        if not pages:
            pages = [["CUESG dossier export"]]

        objects: list[bytes] = []

        def add_object(payload: str | bytes) -> int:
            data = payload.encode("latin-1", errors="replace") if isinstance(payload, str) else payload
            objects.append(data)
            return len(objects)

        font_obj = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        page_ids: list[int] = []
        content_ids: list[int] = []
        pages_placeholder = add_object("<< /Type /Pages /Kids [] /Count 0 >>")

        for page_index, page_lines in enumerate(pages, start=1):
            content_lines = ["BT", "/F1 10 Tf", "50 790 Td", "12 TL"]
            content_lines.append(f"({esc('CUESG Compliance Report')}) Tj")
            content_lines.append("T*")
            content_lines.append(f"({esc(f'Page {page_index} of {len(pages)}')}) Tj")
            content_lines.append("T*")
            content_lines.append("( ) Tj")
            for line in page_lines:
                content_lines.append("T*")
                content_lines.append(f"({esc(line)}) Tj")
            content_lines.append("ET")
            stream = "\n".join(content_lines).encode("latin-1", errors="replace")
            content_id = add_object(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
            content_ids.append(content_id)
            page_obj = add_object(f"<< /Type /Page /Parent {pages_placeholder} 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {content_id} 0 R >>")
            page_ids.append(page_obj)

        kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
        objects[pages_placeholder - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1")
        catalog_obj = add_object(f"<< /Type /Catalog /Pages {pages_placeholder} 0 R >>")

        pdf_parts = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
        offsets = [0]
        for idx, obj in enumerate(objects, start=1):
            offsets.append(sum(len(part) for part in pdf_parts))
            pdf_parts.append(f"{idx} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
        xref_offset = sum(len(part) for part in pdf_parts)
        pdf_parts.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        pdf_parts.append(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            pdf_parts.append(f"{offset:010d} 00000 n \n".encode("ascii"))
        pdf_parts.append(f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_obj} 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode("ascii"))
        return b"".join(pdf_parts)

    def _synthesize_executive_summary(self, package: ProcessedPortfolio) -> list[str]:
        rankings = package.asset_rankings.reset_index(drop=True)
        worst = rankings.iloc[-1] if not rankings.empty else None
        best = rankings.iloc[0] if not rankings.empty else None
        top_anomaly = package.anomaly_records.iloc[0] if not package.anomaly_records.empty else None
        worst_asset_df = (
            package.df.loc[package.df["asset_id"] == worst["asset_id"]].reset_index(drop=True)
            if worst is not None and not rankings.empty
            else pd.DataFrame()
        )
        worst_diesel = float(worst_asset_df["diesel_litres"].sum()) if not worst_asset_df.empty else 0.0
        p1 = (
            f"The portfolio currently spans {package.summary['assets']} assets and {package.summary['records']} time-series records, "
            f"with a CUESG ESG Score of {package.esg_score:.1f}/100. Scope 2 remains the dominant emissions block at "
            f"{package.summary['scope2_total']:.2f} tCO2e, which keeps electricity performance at the center of the compliance narrative."
        )
        p2 = (
            f"The strongest-performing asset is {best['asset_id']} with a BEE-equivalent rating of {best['BEE_Rating']} and "
            f"an ESG score of {best['ESG_Score']:.1f}." if best is not None else "No best-performing asset could be identified from the current portfolio."
        )
        if worst is not None:
            p2 += (
                f" The weakest-performing asset is {worst['asset_id']}, where energy intensity and anomaly recurrence are "
                f"materially pulling down the portfolio benchmark."
            )
        p3 = (
            f"The highest-pressure anomaly is centered on {top_anomaly['asset_id']} with signature {top_anomaly['Anomaly_Signature']}, "
            f"indicating a likely operational root cause that should be addressed before the next reporting cycle."
            if top_anomaly is not None
            else "No acute anomaly pressure is currently visible, so the focus should remain on efficiency tightening and renewable uplift."
        )
        if worst is not None and worst_diesel > 0:
            p3 += f" In addition, {worst['asset_id']} logged {worst_diesel:,.0f} litres of diesel over the reporting window, which threatens quarter-end Scope 1 positioning if not corrected."
        return [p1, p2, p3]
