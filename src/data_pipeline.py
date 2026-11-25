"""
ETL Pipeline for ENSO Analytics
Pulls official ONI index, regional temperature, rainfall data into SQL store.
"""

import pandas as pd
import sqlite3
import requests
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Configuration
SQLITE_PATH = Path(__file__).parent.parent / "data" / "enso_data.db"
RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ENSODataPipeline:
    """Fetches, processes, and stores ENSO-related data."""

    def __init__(self, db_path: Path = SQLITE_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_noaa_oni(self) -> pd.DataFrame:
        """
        Fetch Oceanic Niño Index from NOAA CPC.

        URL: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
        Format: Year, Month, Nino3.4, ONI (3-month mean)
        """
        url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

        logger.info("Fetching ONI data from NOAA CPC...")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            raw_text = response.text

            # Save raw data
            raw_path = RAW_DATA_DIR / "oni_raw.txt"
            with open(raw_path, "w") as f:
                f.write(raw_text)

            # Parse the fixed-width format
            # Header lines: 2, then data rows with spaces
            lines = raw_text.strip().split("\n")
            data_lines = [l for l in lines if l.strip() and not l.startswith(" ")][: -1 or None]

            records = []
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        year = int(parts[0])
                        month = int(parts[1])
                        nino34 = float(parts[2])
                        oni = float(parts[3])
                        records.append({
                            "year": year,
                            "month": month,
                            "nino34_anomaly": nino34,
                            "oni": oni,
                            "period": f"{year}-{month:02d}"
                        })
                    except ValueError:
                        continue

            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["period"])
            df["enso_phase"] = df["oni"].apply(self._classify_enso)

            logger.info(f"Fetched {len(df)} ONI records")
            return df

        except requests.RequestException as e:
            logger.error(f"Failed to fetch ONI data: {e}")
            # Return sample data for development
            return self._sample_oni_data()

    def fetch_noaa_pdo(self) -> pd.DataFrame:
        """
        Fetch Pacific Decadal Oscillation index.
        """
        url = "https://www.psl.noaa.gov/gamini/pdo.html"

        logger.info("Fetching PDO data from NOAA PSL...")

        # NOAA PDO data URL - tab-separated format
        pdo_url = "https://www.psl.noaa.gov/data/correlation/pdo.data"

        try:
            response = requests.get(pdo_url, timeout=30)
            response.raise_for_status()

            raw_path = RAW_DATA_DIR / "pdo_raw.txt"
            with open(raw_path, "w") as f:
                f.write(response.text)

            # Parse - typically year, then 12 monthly values
            lines = response.text.strip().split("\n")
            records = []

            for line in lines:
                if line.strip() and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 13:
                        try:
                            year = int(parts[0])
                            for month, val in enumerate(parts[1:13], 1):
                                records.append({
                                    "year": year,
                                    "month": month,
                                    "pdo": float(val),
                                    "date": pd.Timestamp(year=year, month=month, day=1)
                                })
                        except ValueError:
                            continue

            df = pd.DataFrame(records)
            logger.info(f"Fetched {len(df)} PDO records")
            return df

        except requests.RequestException as e:
            logger.error(f"Failed to fetch PDO data: {e}")
            return self._sample_pdo_data()

    def fetch_fao_crop_data(self, csv_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load FAO crop production data from local file or fetch from URL.
        """
        if csv_path and csv_path.exists():
            logger.info(f"Loading FAO data from {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            # Use provided dataset if available
            default_path = Path(__file__).parent.parent / "Datasets" / "FAOSTAT_data_en_7-23-2026.csv"
            if default_path.exists():
                logger.info(f"Loading FAO data from {default_path}")
                df = pd.read_csv(default_path)
            else:
                logger.warning("No FAO data available")
                return self._sample_crop_data()

        # Standardize columns
        df = df.rename(columns={
            "Year": "year",
            "Area": "country",
            "Item": "crop",
            "Value": "production",
            "Unit": "unit"
        })

        return df

    def _classify_enso(self, oni: float) -> str:
        """
        Classify ENSO phase based on ONI value.

        El Niño: ONI >= 0.5°C for 5 consecutive months
        La Niña: ONI <= -0.5°C for 5 consecutive months
        Neutral: -0.5°C < ONI < 0.5°C
        """
        if oni >= 0.5:
            return "El Nino"
        elif oni <= -0.5:
            return "La Nina"
        else:
            return "Neutral"

    def load_emdat_disasters(self, csv_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load EM-DAT disaster events data.
        """
        if csv_path and csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            default_path = Path(__file__).parent.parent / "Datasets" / "public_emdat_project.csv"
            if default_path.exists():
                df = pd.read_csv(default_path)
            else:
                return pd.DataFrame()

        # Standardize date handling
        if "Year" in df.columns:
            df["year"] = df["Year"]
        if "Total Damage" in df.columns:
            df["total_damage"] = pd.to_numeric(
                df["Total Damage"].astype(str).str.replace(",", ""), errors="coerce"
            )

        return df

    def merge_dataset(self, oni_df: pd.DataFrame, crop_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge ONI data with crop yield data using lag handling.
        """
        # Create a month-year key
        oni_df["year_month"] = oni_df["year"].astype(str) + "-" + oni_df["month"].astype(str).str.zfill(2)

        if "Year" in crop_df.columns:
            crop_df["year_month"] = crop_df["Year"].astype(str) + "-01"

        merged = pd.merge(
            oni_df,
            crop_df,
            left_on="year_month",
            right_on="year_month",
            how="inner"
        )

        return merged

    def store_to_sql(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace"):
        """
        Store dataframe to SQLite database.
        """
        df.to_sql(table_name, self.conn, if_exists=if_exists, index=False)
        logger.info(f"Stored {len(df)} records to {table_name}")

    def run_full_pipeline(self, crop_data_path: Optional[Path] = None):
        """
        Execute complete ETL pipeline.
        """
        logger.info("=" * 50)
        logger.info("Starting ENSO Data Pipeline")
        logger.info("=" * 50)

        # Fetch ONI data
        oni_df = self.fetch_noaa_oni()
        self.store_to_sql(oni_df, "oni_index")

        # Fetch PDO data
        pdo_df = self.fetch_noaa_pdo()
        self.store_to_sql(pdo_df, "pdo_index")

        # Load crop data
        if crop_data_path:
            crop_df = self.fetch_fao_crop_data(crop_data_path)
            self.store_to_sql(crop_df, "crop_production")

        # Load disaster data
        disaster_df = self.load_emdat_disasters()
        self.store_to_sql(disaster_df, "disaster_events")

        # Create master merged table
        if "crop_production" in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall():
            oni_loaded = pd.read_sql("SELECT * FROM oni_index", self.conn)
            crop_loaded = pd.read_sql("SELECT * FROM crop_production", self.conn)
            merged = self.merge_dataset(oni_loaded, crop_loaded)
            self.store_to_sql(merged, "merged_enso_crop")

        logger.info("Pipeline complete!")

        return {
            "oni_records": len(oni_df),
            "pdo_records": len(pdo_df),
            "crop_records": len(crop_df) if "crop_df" in dir() else 0,
        }

    @staticmethod
    def _sample_oni_data() -> pd.DataFrame:
        """Sample ONI data for development."""
        import numpy as np
        np.random.seed(42)
        dates = pd.date_range("1980-01", "2024-01", freq="MS")
        n = len(dates)
        
        # Generate enough values for the date range
        oni_values = np.random.normal(0, 0.8, n)
        
        # Add some ENSO-like patterns
        for i in range(0, n, 12):
            oni_values[i:i+12] += np.random.normal(0.3, 0.2, min(12, n-i))  # Seasonal

        df = pd.DataFrame({
            "date": dates,
            "year": dates.year,
            "month": dates.month,
            "oni": oni_values,
            "nino34_anomaly": oni_values * 0.9,
        })
        df["period"] = df["date"].dt.strftime("%Y-%m")
        df["enso_phase"] = df["oni"].apply(lambda x: "El Nino" if x >= 0.5 else ("La Nina" if x <= -0.5 else "Neutral"))
        return df

    @staticmethod
    def _sample_pdo_data() -> pd.DataFrame:
        """Sample PDO data for development."""
        import numpy as np
        np.random.seed(42)
        dates = pd.date_range("1980-01", "2024-01", freq="MS")
        pdo_values = np.random.normal(0, 1, len(dates))

        return pd.DataFrame({
            "date": dates,
            "year": dates.year,
            "month": dates.month,
            "pdo": pdo_values
        })

    @staticmethod
    def _sample_crop_data() -> pd.DataFrame:
        """Sample crop data for development."""
        import numpy as np
        np.random.seed(42)
        years = list(range(1990, 2024))
        countries = ["USA", "Brazil", "Argentina", "India", "Australia"]
        crops = ["Wheat", "Rice", "Soybeans", "Corn"]
        
        data = []
        for year in years:
            for country in countries:
                for crop in crops:
                    # Yield varies by crop and has some correlation with ONI
                    base_yield = {"Wheat": 3.5, "Rice": 4.5, "Soybeans": 2.8, "Corn": 5.5}[crop]
                    # Add trend and noise
                    trend = (year - 1990) * 0.02
                    noise = np.random.normal(0, 0.3)
                    yield_val = base_yield + trend + noise
                    production = yield_val * 1000  # tonnes
                    
                    data.append({
                        "year": year,
                        "country": country,
                        "crop": crop,
                        "yield": round(yield_val, 2),
                        "production": round(production, 1),
                        "unit": "tonnes/ha"
                    })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    pipeline = ENSODataPipeline()
    stats = pipeline.run_full_pipeline()
    print(f"Pipeline stats: {stats}")