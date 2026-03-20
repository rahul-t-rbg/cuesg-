from __future__ import annotations

import random

import numpy as np
import pandas as pd


RNG = np.random.default_rng(42)
random.seed(42)


def _dirty_format(value: float, style: str) -> str | float:
    if style == "currency":
        return f"INR {value:,.0f}"
    if style == "comma":
        return f"{value:,.2f}"
    if style == "percent":
        return f"{value:.2f}%"
    if style == "text":
        return f"approx {value:.1f}"
    return round(value, 3)


def _to_float(value: object) -> float:
    return float(str(value).replace(",", "").replace("INR", "").replace("%", "").replace("approx", "").strip())


def build_apex_big_data() -> pd.DataFrame:
    months = pd.date_range("2020-01-31", periods=72, freq="ME")
    cities = ["BLR", "MUM", "DEL", "HYD", "CHE", "PNQ", "KOL", "AHM"]
    assets = [f"BLD-{city}-{idx:03d}" for city in cities for idx in range(1, 26)]

    rows: list[dict[str, object]] = []
    for asset_idx, asset_id in enumerate(assets, start=1):
        area = float(RNG.integers(14_000, 185_000))
        baseline_occupancy = float(RNG.uniform(35, 96))
        solar_capacity_factor = float(RNG.uniform(0.03, 0.30))
        climate_bias = float(RNG.uniform(-3.1, 4.8))
        water_stress = float(RNG.uniform(0.90, 1.35))
        age = int(RNG.integers(2, 35))

        for month_no, month in enumerate(months, start=1):
            season = np.sin((2 * np.pi * month.month) / 12.0)
            occupancy = float(np.clip(baseline_occupancy + RNG.normal(0, 8), 18, 100))
            outdoor_temp = float(np.clip(28.5 + climate_bias + season * 6.5 + RNG.normal(0, 1.4), 14, 44))
            indoor_temp = float(np.clip(23.8 + RNG.normal(0, 0.9), 18, 28))
            humidity = float(np.clip(52 + season * 11 + RNG.normal(0, 6), 18, 95))
            floor_area = max(area + RNG.normal(0, area * 0.012), 8_000)

            hvac_kw = max((outdoor_temp - indoor_temp) * (floor_area / 1200.0) * 0.52 + RNG.normal(0, 45), 25)
            lighting_kwh = max((floor_area / 22.0) * (0.65 + occupancy / 100.0) + RNG.normal(0, 180), 220)
            plug_load_kwh = max((floor_area / 18.0) * (0.55 + occupancy / 95.0) + RNG.normal(0, 220), 240)
            energy_kwh = max((hvac_kw * 38) + lighting_kwh + plug_load_kwh + RNG.normal(0, 2800), 800)
            diesel_l = max(RNG.normal(240 + max(season, 0) * 140 + age * 1.5, 65), 20)
            solar_kwh = max((energy_kwh * solar_capacity_factor) + RNG.normal(0, 650), 0)
            water_kl = max((occupancy * floor_area / 2600.0) * water_stress + RNG.normal(0, 60), 30)
            waste_kg = max(RNG.normal(640 + age * 8 + month_no * 2.5, 115), 20)
            hazardous_waste_kg = max(RNG.normal(32 + age * 0.8, 9), 1)
            spend_inr = max(RNG.normal(7_500_000 + floor_area * 21 + month_no * 18_000, 1_250_000), 300_000)
            utility_cost_inr = max((energy_kwh - solar_kwh) * RNG.uniform(7.5, 12.8), 12_000)

            generator_output_kw = max(diesel_l * RNG.uniform(0.0, 1.6), 0)

            if asset_idx % 13 == 0 and month_no in {10, 22, 37, 58}:
                energy_kwh *= RNG.uniform(1.9, 2.8)
                hvac_kw *= RNG.uniform(1.5, 2.3)
                diesel_l *= RNG.uniform(1.6, 2.5)
            if asset_idx % 17 == 0 and month_no in {14, 29, 43, 65}:
                solar_kwh *= RNG.uniform(0.1, 0.35)
            if asset_idx % 19 == 0 and month_no in {8, 19, 41, 60}:
                water_kl *= RNG.uniform(2.0, 3.6)
            if asset_idx % 23 == 0 and month_no in {5, 34, 47}:
                diesel_l *= RNG.uniform(4.0, 8.5)
                generator_output_kw = 0.0

            rows.append(
                {
                    "Reporting Month": month.strftime("%Y-%m-%d"),
                    "Building Code": asset_id,
                    "Campus Name": f"{asset_id} Campus",
                    "Gross Floor Area sqm": round(floor_area, 2),
                    "Energy Cons. (kWh)": round(energy_kwh, 2),
                    "Diesel Usage [L]": round(diesel_l, 2),
                    "Solar Offset kWh": round(solar_kwh, 2),
                    "Water Draw (KL)": round(water_kl, 2),
                    "E-Waste KG": round(waste_kg, 2),
                    "Haz Waste KG": round(hazardous_waste_kg, 2),
                    "Procurement Spend INR": round(spend_inr, 2),
                    "Utility Bill Cost": round(utility_cost_inr, 2),
                    "Outdoor Temp C": round(outdoor_temp, 2),
                    "Indoor Temp C": round(indoor_temp, 2),
                    "Humidity %": round(humidity, 2),
                    "Occupancy %": round(occupancy, 2),
                    "HVAC Demand KW": round(hvac_kw, 2),
                    "Lighting Load KWh": round(lighting_kwh, 2),
                    "Plug Load KWh": round(plug_load_kwh, 2),
                    "Generator Output kW": round(generator_output_kw, 2),
                    "Region": asset_id.split("-")[1],
                }
            )

    df = pd.DataFrame(rows)

    dirty_cols = [
        "Energy Cons. (kWh)",
        "Diesel Usage [L]",
        "Solar Offset kWh",
        "Water Draw (KL)",
        "Procurement Spend INR",
        "Utility Bill Cost",
        "Occupancy %",
        "Gross Floor Area sqm",
        "Haz Waste KG",
    ]
    df[dirty_cols] = df[dirty_cols].astype(object)

    for column in ["Energy Cons. (kWh)", "Diesel Usage [L]", "Solar Offset kWh", "Water Draw (KL)", "Procurement Spend INR", "Utility Bill Cost"]:
        style = "currency" if column == "Procurement Spend INR" else "comma"
        sample_index = df.sample(frac=0.08, random_state=abs(hash(column)) % 10_000).index
        df.loc[sample_index, column] = df.loc[sample_index, column].map(lambda value: _dirty_format(float(value), style))

    for column in ["Occupancy %", "Gross Floor Area sqm", "Haz Waste KG"]:
        sample_index = df.sample(frac=0.04, random_state=(abs(hash(column)) + 333) % 10_000).index
        style = "percent" if column == "Occupancy %" else "comma"
        df.loc[sample_index, column] = df.loc[sample_index, column].map(lambda value: _dirty_format(float(value), style))

    for column in ["Energy Cons. (kWh)", "Diesel Usage [L]", "Solar Offset kWh", "Water Draw (KL)", "E-Waste KG", "Haz Waste KG", "Procurement Spend INR", "Occupancy %"]:
        nan_index = df.sample(frac=0.065, random_state=(abs(hash(column)) + 29) % 10_000).index
        df.loc[nan_index, column] = np.nan

    outlier_index = df.sample(frac=0.025, random_state=911).index
    for column, factor_low, factor_high in [
        ("Energy Cons. (kWh)", 3.0, 5.2),
        ("Water Draw (KL)", 2.5, 4.2),
        ("Diesel Usage [L]", 4.0, 9.0),
    ]:
        df.loc[outlier_index, column] = df.loc[outlier_index, column].apply(
            lambda value: _dirty_format(
                float(str(value).replace(",", "").replace("INR", "").replace("%", "").replace("approx", "").strip()) * RNG.uniform(factor_low, factor_high),
                "comma",
            )
            if pd.notna(value)
            else value
        )

    weird_text_idx = df.sample(frac=0.02, random_state=1234).index
    df.loc[weird_text_idx, "Utility Bill Cost"] = df.loc[weird_text_idx, "Utility Bill Cost"].map(
        lambda value: _dirty_format(_to_float(value), "text") if pd.notna(value) else value
    )

    return df


def save_outputs(df: pd.DataFrame) -> None:
    df.to_csv("master_esg_data.csv", index=False)
    df.to_excel("master_esg_data.xlsx", index=False)


if __name__ == "__main__":
    generated_df = build_apex_big_data()
    save_outputs(generated_df)
    print(
        f"[OMNI-SYSTEM] Generated {len(generated_df):,} rows across "
        f"{generated_df['Building Code'].nunique()} assets with NaNs, dirty formatting, and injected anomalies."
    )
