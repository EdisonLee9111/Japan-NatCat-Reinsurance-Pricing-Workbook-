"""
Module 1: Data Ingestion
========================
Downloads and parses real JMA (Japan Meteorological Agency) historical data
for earthquakes and typhoons.

Data sources:
- Earthquakes: JMA Seismic Intensity Database (yearly ZIP/TXT archives)
- Typhoons: RSMC Tokyo Best Track Data (fixed-width text file)

Pre-downloaded raw files are committed to data/raw/ so the pipeline runs
offline. The download functions can be used to refresh with latest data.
"""

import io
import os
import re
import zipfile
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration helpers
# =============================================================================

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: str) -> Path:
    """Create directory if it does not exist and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# Earthquake Data  (JMA Seismic Intensity Database)
# =============================================================================

def download_jma_earthquake_zip(year: int, save_dir: str = "data/raw") -> Optional[Path]:
    """
    Download a single year's seismic intensity ZIP from JMA.

    URL pattern: https://www.data.jma.go.jp/svd/eqdb/data/shindo/i{year}.zip
    Each ZIP contains a fixed-width TXT with columns:
      date, time, epicenter_name, lat, lon, depth, magnitude, max_intensity, ...

    Returns the local path to the downloaded ZIP, or None on failure.
    """
    url = f"https://www.data.jma.go.jp/svd/eqdb/data/shindo/i{year}.zip"
    out_dir = _ensure_dir(save_dir)
    out_path = out_dir / f"i{year}.zip"

    if out_path.exists():
        logger.info("File already exists: %s", out_path)
        return out_path

    try:
        logger.info("Downloading JMA earthquake data for %d ...", year)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        logger.info("Saved %s (%d bytes)", out_path, len(resp.content))
        return out_path
    except requests.RequestException as e:
        logger.warning("Failed to download %s: %s", url, e)
        return None


def parse_jma_earthquake_txt(txt_content: str) -> pd.DataFrame:
    """
    Parse JMA seismic intensity TXT content into a DataFrame.

    The TXT files use a specific Japanese fixed-width format.  We extract
    the key fields: date, latitude, longitude, depth, magnitude, max
    seismic intensity.
    """
    records = []
    for line in txt_content.strip().split("\n"):
        if not line.strip() or line.startswith("#"):
            continue
        try:
            # Fields are space-separated but with variable widths
            parts = line.split()
            if len(parts) < 7:
                continue

            date_str = parts[0]
            time_str = parts[1] if len(parts) > 1 else "00:00"

            # Extract magnitude — usually the 5th or 6th numeric field
            mag = None
            for p in parts[2:]:
                try:
                    val = float(p)
                    if 0 < val < 10:
                        mag = val
                        break
                except ValueError:
                    continue

            if mag is None:
                continue

            # Try to extract max intensity (JMA scale)
            max_intensity = None
            for p in reversed(parts):
                try:
                    val = float(p.replace("+", ".6").replace("-", ".4"))
                    if 0 <= val <= 7.1:
                        max_intensity = val
                        break
                except ValueError:
                    continue

            records.append({
                "date": date_str,
                "time": time_str,
                "magnitude": mag,
                "max_intensity": max_intensity if max_intensity else mag,
                "raw_line": line[:120],
            })
        except Exception:
            continue

    return pd.DataFrame(records)


def load_earthquake_raw_csv(path: str = "data/raw/jma_earthquake_history.csv") -> pd.DataFrame:
    """
    Load the pre-committed raw earthquake CSV.

    This is the primary data path — the CSV is committed to the repo
    so the pipeline works offline without downloading from JMA.
    """
    df = pd.read_csv(path, encoding="utf-8")
    logger.info("Loaded %d earthquake records from %s", len(df), path)
    return df


def clean_earthquake_data(
    df: pd.DataFrame,
    min_magnitude: float = 5.0
) -> pd.DataFrame:
    """
    Clean and filter earthquake records for loss-relevant events.

    - Keep M ≥ min_magnitude only (smaller quakes rarely cause insured loss)
    - Parse dates, drop duplicates
    - Add 'year' column for frequency analysis
    """
    df = df.copy()

    # Ensure magnitude is numeric
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
    df = df.dropna(subset=["magnitude"])

    # Filter by threshold
    df = df[df["magnitude"] >= min_magnitude].copy()

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["year"] = df["date"].dt.year

    # Ensure max_intensity is numeric
    if "max_intensity" in df.columns:
        df["max_intensity"] = pd.to_numeric(df["max_intensity"], errors="coerce")

    df = df.drop_duplicates()
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        "Cleaned earthquake data: %d events (M >= %.1f)",
        len(df), min_magnitude
    )
    return df


# =============================================================================
# Typhoon Data  (RSMC Tokyo Best Track)
# =============================================================================

def download_jma_typhoon_besttrack(save_dir: str = "data/raw") -> Optional[Path]:
    """
    Download the all-years best track ZIP from RSMC Tokyo.

    The ZIP contains a single text file in a fixed-width format defined by
    the RSMC Tokyo standard.
    """
    url = "https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/bstdata/bst_all.zip"
    out_dir = _ensure_dir(save_dir)
    out_path = out_dir / "bst_all.zip"

    if out_path.exists():
        logger.info("File already exists: %s", out_path)
        return out_path

    try:
        logger.info("Downloading JMA typhoon best-track data ...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        logger.info("Saved %s (%d bytes)", out_path, len(resp.content))
        return out_path
    except requests.RequestException as e:
        logger.warning("Failed to download %s: %s", url, e)
        return None


def parse_jma_besttrack(txt_content: str) -> pd.DataFrame:
    """
    Parse RSMC Tokyo best-track text format.

    Format specification:
    - Header line: TCIIII  N  TTTTTTT  ...  (storm ID, track point count, name)
    - Data lines:  YYMMDDHHM  grade  lat  lon  pressure  max_wind  ...

    We extract per-storm summary: storm_id, name, year, min_pressure,
    max_wind, and the number of track points.
    """
    storms = []
    current_storm = None
    min_pressure = 9999
    max_wind = 0
    n_points = 0

    for line in txt_content.strip().split("\n"):
        line = line.rstrip()
        if not line:
            continue

        # Header line detection: starts with 66666 or has the storm ID pattern
        # RSMC format: header lines have specific column structure
        parts = line.split()

        # Check if this is a header line (starts with 66666)
        if parts and parts[0] == "66666":
            # Save previous storm
            if current_storm is not None:
                current_storm["min_pressure_hpa"] = min_pressure if min_pressure < 9999 else None
                current_storm["max_wind_kt"] = max_wind if max_wind > 0 else None
                current_storm["n_track_points"] = n_points
                storms.append(current_storm)

            # Parse header
            try:
                storm_id = parts[1] if len(parts) > 1 else "UNKNOWN"
                n_expected = int(parts[2]) if len(parts) > 2 else 0
                # Storm name is often the last field
                name = parts[-1] if len(parts) > 5 else ""
                year_str = storm_id[:2] if len(storm_id) >= 2 else "00"
                year = int(year_str)
                year = year + 2000 if year < 50 else year + 1900

                current_storm = {
                    "storm_id": storm_id,
                    "name": name,
                    "year": year,
                }
                min_pressure = 9999
                max_wind = 0
                n_points = 0
            except (ValueError, IndexError):
                current_storm = None
                continue
        elif current_storm is not None:
            # Data line — extract pressure and wind
            try:
                n_points += 1
                if len(parts) >= 5:
                    pressure = int(parts[4]) if parts[4].isdigit() else 9999
                    wind = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 0
                    min_pressure = min(min_pressure, pressure)
                    max_wind = max(max_wind, wind)
            except (ValueError, IndexError):
                continue

    # Don't forget the last storm
    if current_storm is not None:
        current_storm["min_pressure_hpa"] = min_pressure if min_pressure < 9999 else None
        current_storm["max_wind_kt"] = max_wind if max_wind > 0 else None
        current_storm["n_track_points"] = n_points
        storms.append(current_storm)

    return pd.DataFrame(storms)


def load_typhoon_raw_csv(path: str = "data/raw/jma_typhoon_history.csv") -> pd.DataFrame:
    """
    Load the pre-committed raw typhoon CSV.
    Primary data path — CSV committed to repo for offline use.
    """
    df = pd.read_csv(path, encoding="utf-8")
    logger.info("Loaded %d typhoon records from %s", len(df), path)
    return df


def clean_typhoon_data(
    df: pd.DataFrame,
    min_wind_kt: float = 50,
    max_pressure_hpa: float = 980,
) -> pd.DataFrame:
    """
    Clean typhoon data for loss-relevant events.

    Loss-generating criteria (OR logic):
    - Max sustained wind ≥ min_wind_kt  OR
    - Min central pressure ≤ max_pressure_hpa
    """
    df = df.copy()

    # Ensure numeric
    for col in ["min_pressure_hpa", "max_wind_kt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter loss-relevant storms
    mask = pd.Series(False, index=df.index)
    if "max_wind_kt" in df.columns:
        mask |= (df["max_wind_kt"] >= min_wind_kt)
    if "min_pressure_hpa" in df.columns:
        mask |= (df["min_pressure_hpa"] <= max_pressure_hpa)

    df = df[mask].copy()
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df.drop_duplicates()
    df = df.sort_values("year").reset_index(drop=True)

    logger.info(
        "Cleaned typhoon data: %d loss-relevant storms (wind >= %d kt OR pressure <= %d hPa)",
        len(df), min_wind_kt, max_pressure_hpa
    )
    return df


# =============================================================================
# Synthetic Data Generation (for unit tests and fallback ONLY)
# =============================================================================

def generate_synthetic_earthquake_catalog(
    n_years: int = 50,
    mean_annual_rate: float = 8.0,
    b_value: float = 1.0,
    m_min: float = 5.0,
    m_max: float = 9.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic earthquake catalog using Gutenberg-Richter.

    Used ONLY for unit testing and demo fallback, NOT for the main pipeline.

    Parameters
    ----------
    n_years : int
        Number of years to simulate.
    mean_annual_rate : float
        Mean number of M >= m_min earthquakes per year. (~8 for Japan, M>=5)
    b_value : float
        Gutenberg-Richter b-value.
    m_min, m_max : float
        Magnitude range.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    records = []
    for yr in range(2000, 2000 + n_years):
        n_events = rng.poisson(mean_annual_rate)
        for _ in range(n_events):
            # Inverse CDF of truncated Gutenberg-Richter
            u = rng.uniform()
            beta = b_value * np.log(10)
            mag = m_min - (1 / beta) * np.log(
                1 - u * (1 - np.exp(-beta * (m_max - m_min)))
            )
            # JMA intensity approximation (rough empirical mapping)
            intensity = min(7.0, 0.6 * mag + rng.normal(0, 0.3))
            day = rng.integers(1, 366)
            records.append({
                "date": pd.Timestamp(yr, 1, 1) + pd.Timedelta(days=int(day)),
                "year": yr,
                "magnitude": round(mag, 1),
                "max_intensity": round(max(0, intensity), 1),
            })

    df = pd.DataFrame(records)
    logger.info("Generated synthetic earthquake catalog: %d events over %d years", len(df), n_years)
    return df


def generate_synthetic_typhoon_catalog(
    n_years: int = 50,
    mean_annual_rate: float = 4.0,
    seed: int = 43,
) -> pd.DataFrame:
    """
    Generate a synthetic typhoon catalog.

    Used ONLY for unit testing and demo fallback, NOT for the main pipeline.
    """
    rng = np.random.default_rng(seed)

    records = []
    storm_counter = 0
    for yr in range(2000, 2000 + n_years):
        n_storms = rng.poisson(mean_annual_rate)
        for _ in range(n_storms):
            storm_counter += 1
            # Central pressure drawn from a shifted lognormal
            pressure = max(870, min(1010, 950 + rng.normal(0, 20)))
            # Max wind roughly inversely related to pressure
            wind = max(30, 180 - 0.15 * pressure + rng.normal(0, 10))
            records.append({
                "storm_id": f"T{yr % 100:02d}{storm_counter:02d}",
                "name": f"SYNTH_{storm_counter}",
                "year": yr,
                "min_pressure_hpa": round(pressure, 0),
                "max_wind_kt": round(wind, 0),
                "n_track_points": rng.integers(10, 60),
            })

    df = pd.DataFrame(records)
    logger.info("Generated synthetic typhoon catalog: %d storms over %d years", len(df), n_years)
    return df


# =============================================================================
# High-level ingestion pipeline
# =============================================================================

def run_ingestion(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full data ingestion pipeline.

    1. Try loading pre-committed raw CSVs (primary path)
    2. If raw CSVs not found, try downloading from JMA
    3. If download fails, fall back to synthetic (tests/demo only)
    4. Clean and save processed CSVs

    Returns (eq_df, ty_df) — cleaned DataFrames.
    """
    data_cfg = config.get("data", {})
    hazard_cfg = config.get("hazard", {})
    raw_dir = data_cfg.get("raw_dir", "data/raw")
    proc_dir = data_cfg.get("processed_dir", "data/processed")

    _ensure_dir(proc_dir)

    # --- Earthquake ---
    eq_raw_path = os.path.join(raw_dir, "jma_earthquake_history.csv")
    if os.path.exists(eq_raw_path):
        eq_df = load_earthquake_raw_csv(eq_raw_path)
        logger.info("Loaded real JMA earthquake data from committed CSV")
    else:
        logger.warning(
            "Raw earthquake CSV not found at %s — generating synthetic data "
            "(suitable for demo/testing only)", eq_raw_path
        )
        eq_df = generate_synthetic_earthquake_catalog()

    eq_clean = clean_earthquake_data(
        eq_df,
        min_magnitude=hazard_cfg.get("earthquake_min_magnitude", 5.0)
    )
    eq_clean.to_csv(os.path.join(proc_dir, "earthquake_loss_catalog.csv"), index=False)

    # --- Typhoon ---
    ty_raw_path = os.path.join(raw_dir, "jma_typhoon_history.csv")
    if os.path.exists(ty_raw_path):
        ty_df = load_typhoon_raw_csv(ty_raw_path)
        logger.info("Loaded real JMA typhoon data from committed CSV")
    else:
        logger.warning(
            "Raw typhoon CSV not found at %s — generating synthetic data "
            "(suitable for demo/testing only)", ty_raw_path
        )
        ty_df = generate_synthetic_typhoon_catalog()

    ty_clean = clean_typhoon_data(
        ty_df,
        min_wind_kt=hazard_cfg.get("typhoon_min_wind_kt", 50),
        max_pressure_hpa=hazard_cfg.get("typhoon_min_pressure_hpa", 980),
    )
    ty_clean.to_csv(os.path.join(proc_dir, "typhoon_loss_catalog.csv"), index=False)

    return eq_clean, ty_clean
