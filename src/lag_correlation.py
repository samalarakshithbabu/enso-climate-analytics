"""
Lag Correlation Analysis for ENSO-Crop Relationships
Computes correlation between ONI and crop yields at various time lags.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LagCorrelationResult:
    """Results from lag correlation analysis."""
    lag_months: int
    correlation: float
    p_value: float
    significant: bool
    crop: str
    region: str
    n_observations: int


class ENSOLagCorrelator:
    """
    Computes and analyzes lag correlations between ENSO (ONI) and crop yields.

    Key insight: ENSO effects on agriculture are not instantaneous.
    El Niño/La Niña effects manifest with a delay of 3-12 months depending
    on crop type and region.
    """

    def __init__(self, oni_df: pd.DataFrame, crop_df: pd.DataFrame):
        """
        Initialize with ONI and crop dataframes.

        Args:
            oni_df: DataFrame with 'date', 'oni', 'year', 'month' columns
            crop_df: DataFrame with 'year', 'country', 'crop', 'production'/'yield' columns
        """
        self.oni = oni_df.copy()
        self.crop = crop_df.copy()
        self._prepare_data()

    def _prepare_data(self):
        """Align datasets for correlation analysis."""
        # Ensure date columns are datetime
        if "date" not in self.oni.columns and "year" in self.oni.columns:
            self.oni["date"] = pd.to_datetime(
                self.oni["year"].astype(str) + "-" +
                self.oni.get("month", 1).astype(str).str.zfill(2) + "-01"
            )

        # Create year-month key for merging
        self.oni["year_month"] = self.oni["date"].dt.to_period("M")

    def compute_lag_correlations(
        self,
        crop_type: str,
        region: str,
        max_lag: int = 24,
        min_overlap_months: int = 36
    ) -> pd.DataFrame:
        """
        Compute correlations at multiple lag periods.

        Args:
            crop_type: Type of crop to analyze
            region: Country/region name
            max_lag: Maximum lag in months to test
            min_overlap_months: Minimum overlapping observations required

        Returns:
            DataFrame with lag correlations and significance
        """
        # Filter crop data
        crop_filtered = self.crop[
            (self.crop["crop"] == crop_type) &
            (self.crop["country"] == region)
        ].copy()

        if crop_filtered.empty:
            logger.warning(f"No crop data for {region} - {crop_type}")
            return pd.DataFrame()

        # Get crop years and map to season midpoint
        crop_filtered = crop_filtered.rename(columns={"Year": "year", "Value": "yield"})

        # Aggregate to annual if needed
        crop_annual = crop_filtered.groupby("year").agg({
            "yield": "mean",
            "production": "mean" if "production" in crop_filtered.columns else "sum"
        }).reset_index()

        results = []

        for lag in range(0, max_lag + 1):
            # Shift ONI data to create lagged values
            # For annual crop data, we use the ONI value from several months before harvest
            oni_lagged = self._create_lagged_oni(lag, crop_annual["year"].values)

            # Merge on year
            merged = pd.merge(crop_annual, oni_lagged, on="year", how="inner")

            if len(merged) < 10:  # Need minimum sample
                continue

            # Compute correlation
            if "yield" in merged.columns and "oni_lagged" in merged.columns:
                corr, pval = stats.pearsonr(
                    merged["oni_lagged"].dropna(),
                    merged.loc[merged["oni_lagged"].notna(), "yield"]
                )
            elif "yield" in merged.columns and "oni_mean" in merged.columns:
                corr, pval = stats.pearsonr(
                    merged["oni_mean"].dropna(),
                    merged.loc[merged["oni_mean"].notna(), "yield"]
                )
            else:
                logger.warning("Required columns not found for correlation")
                continue

            results.append({
                "lag_months": lag,
                "correlation": corr,
                "p_value": pval,
                "significant": pval < 0.05,
                "n_observations": len(merged),
                "crop": crop_type,
                "region": region
            })

        return pd.DataFrame(results)

    def _create_lagged_oni(
        self,
        lag_months: int,
        crop_years: np.ndarray
    ) -> pd.DataFrame:
        """
        Create lagged ONI values aligned with crop years.

        For a lag of N months, we use the ONI value from N months
        before the typical harvest time.
        """
        oni_data = []

        for year in crop_years:
            # Use growing season average (e.g., 3 months before harvest)
            # This is a simplification - in practice, you'd want growing season dates
            harvest_month = 9  # September is common for many crops
            target_year = year
            target_month = harvest_month - lag_months

            if target_month <= 0:
                target_year -= 1
                target_month += 12

            # Find closest ONI observation
            mask = (
                (self.oni["year"] == target_year) &
                (self.oni["month"] == target_month)
            )

            if mask.any():
                oni_val = self.oni.loc[mask, "oni"].values[0]
            else:
                # Use annual average
                year_mask = self.oni["year"] == target_year
                oni_val = self.oni.loc[year_mask, "oni"].mean()

            oni_data.append({
                "year": year,
                "oni_lagged": oni_val,
                "oni_mean": self.oni[self.oni["year"] == year]["oni"].mean()
            })

        return pd.DataFrame(oni_data)

    def find_optimal_lag(
        self,
        crop_type: str,
        region: str,
        max_lag: int = 24
    ) -> Dict[str, any]:
        """
        Find the optimal lag with strongest correlation.

        Returns dictionary with optimal lag info.
        """
        correlations = self.compute_lag_correlations(crop_type, region, max_lag)

        if correlations.empty:
            return {"error": "No correlation data available"}

        # Find lag with strongest absolute correlation
        correlations["abs_corr"] = correlations["correlation"].abs()
        optimal = correlations.loc[correlations["abs_corr"].idxmax()]

        # Also identify significant lags
        sig_lags = correlations[correlations["significant"]]

        return {
            "optimal_lag_months": int(optimal["lag_months"]),
            "optimal_correlation": float(optimal["correlation"]),
            "optimal_p_value": float(optimal["p_value"]),
            "significant_lags": sig_lags["lag_months"].tolist() if not sig_lags.empty else [],
            "all_correlations": correlations.to_dict("records")
        }

    def create_lag_summary_report(
        self,
        crop_region_pairs: list,
        max_lag: int = 24
    ) -> pd.DataFrame:
        """
        Generate comprehensive lag correlation summary for multiple crops/regions.
        """
        all_results = []

        for crop, region in crop_region_pairs:
            result = self.find_optimal_lag(crop, region, max_lag)

            if "error" not in result:
                all_results.append({
                    "crop": crop,
                    "region": region,
                    "optimal_lag_months": result["optimal_lag_months"],
                    "optimal_correlation": result["optimal_correlation"],
                    "optimal_p_value": result["optimal_p_value"],
                    "significant": result["optimal_p_value"] < 0.05,
                    "n_significant_lags": len(result["significant_lags"]),
                    "interpretation": self._interpret_correlation(
                        result["optimal_correlation"],
                        result["optimal_p_value"] < 0.05
                    )
                })

        summary_df = pd.DataFrame(all_results)
        return summary_df

    @staticmethod
    def _interpret_correlation(corr: float, significant: bool) -> str:
        """Provide interpretation of correlation coefficient."""
        if not significant:
            return "Not statistically significant (p >= 0.05)"

        strength = "weak"
        if abs(corr) >= 0.7:
            strength = "strong"
        elif abs(corr) >= 0.4:
            strength = "moderate"

        direction = "positive" if corr > 0 else "negative"

        return f"{strength.capitalize()} {direction} correlation"

    def plot_lag_correlation(
        self,
        crop_type: str,
        region: str,
        max_lag: int = 24,
        save_path: Optional[Path] = None
    ):
        """
        Generate lag correlation plot.
        """
        correlations = self.compute_lag_correlations(crop_type, region, max_lag)

        if correlations.empty:
            logger.warning("No data to plot")
            return

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot correlations
        ax.bar(correlations["lag_months"], correlations["correlation"], color="steelblue")

        # Highlight significant lags
        sig_mask = correlations["significant"]
        if sig_mask.any():
            ax.bar(
                correlations.loc[sig_mask, "lag_months"],
                correlations.loc[sig_mask, "correlation"],
                color="darkgreen" if correlations.loc[sig_mask, "correlation"].mean() > 0 else "darkred"
            )

        # Add significance threshold lines
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        ax.set_xlabel("Lag (months)")
        ax.set_ylabel("Correlation with ONI")
        ax.set_title(f"ONI-Crop Yield Lag Correlation: {crop_type} in {region}")
        ax.set_xlim(-1, max_lag + 1)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved lag correlation plot to {save_path}")

        plt.close()


if __name__ == "__main__":
    # Example usage
    from data_pipeline import ENSODataPipeline

    pipeline = ENSODataPipeline()
    oni_sample = pipeline._sample_oni_data()
    crop_sample = pipeline._sample_crop_data()

    correlator = ENSOLagCorrelator(oni_sample, crop_sample)

    # Compute lag correlations for wheat in USA
    results = correlator.find_optimal_lag("Wheat", "USA", max_lag=24)
    print(f"Optimal lag: {results}")

    # Generate summary for multiple crops/regions
    pairs = [("Wheat", "USA"), ("Rice", "India"), ("Soybeans", "Brazil")]
    summary = correlator.create_lag_summary_report(pairs, max_lag=24)
    print("\nLag Summary:")
    print(summary.to_string())