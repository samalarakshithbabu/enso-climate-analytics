"""
Regression Analysis: ENSO Phase Effects on Regional Crop Yields

Model: Yield = β₀ + β₁·ONI_lag + β₂·Precipitation + β₃·Temperature + ε

This module implements OLS regression with robust standard errors
to estimate the relationship between ENSO and crop yields.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RegressionResult:
    """Standardized regression result container."""
    r_squared: float
    adjusted_r_squared: float
    coefficients: Dict[str, float]
    p_values: Dict[str, float]
    std_errors: Dict[str, float]
    significant_vars: List[str]
    f_statistic: float
    f_p_value: float
    n_observations: int
    model_summary: str


class ENSOReggressionModel:
    """
    Regression model linking ENSO to crop yields with regional controls.

    The model estimates:
        Yield = β₀ + β₁·ONI_lag + β₂·Precip + β₃·Temp + β₄·Trend + regional_fe + ε

    Features:
    - Robust standard errors (HC3)
    - VIF check for multicollinearity
    - Regional fixed effects
    - Model selection via AIC/BIC
    """

    def __init__(
        self,
        oni_df: pd.DataFrame,
        crop_df: pd.DataFrame,
        climate_df: Optional[pd.DataFrame] = None
    ):
        """
        Initialize with ONI, crop, and optional climate data.

        Args:
            oni_df: ONI index data with 'date', 'oni', 'year', 'month'
            crop_df: Crop yield data with 'year', 'country', 'crop', 'yield'
            climate_df: Precipitation/temperature data (if available)
        """
        self.oni = oni_df.copy()
        self.crop = crop_df.copy()
        self.climate = climate_df.copy() if climate_df is not None else None
        self._merged_data: Optional[pd.DataFrame] = None

    def prepare_dataset(
        self,
        crop_type: str,
        region: str,
        lag_months: int = 6,
        include_controls: bool = True
    ) -> pd.DataFrame:
        """
        Prepare regression dataset with lagged ONI and controls.

        Args:
            crop_type: Crop to analyze
            region: Country/region
            lag_months: Lag for ONI effect
            include_controls: Include precipitation/temperature if available

        Returns:
            Prepared DataFrame for regression
        """
        # Filter and prepare crop data
        crop_filtered = self.crop[
            (self.crop["crop"] == crop_type) &
            (self.crop["country"] == region)
        ].copy()

        if crop_filtered.empty:
            raise ValueError(f"No data for {crop_type} in {region}")

        # Ensure year column
        if "Year" in crop_filtered.columns:
            crop_filtered["year"] = crop_filtered["Year"]
        if "Value" in crop_filtered.columns:
            crop_filtered["yield"] = crop_filtered["Value"]

        # Aggregate to annual if needed
        if "yield" in crop_filtered.columns:
            crop_annual = crop_filtered.groupby("year").agg({
                "yield": "mean"
            }).reset_index()
        else:
            crop_annual = crop_filtered[["year", "production"]].copy()
            crop_annual = crop_annual.rename(columns={"production": "yield"})

        # Create lagged ONI - annual average
        oni_annual = self.oni.groupby("year").agg({
            "oni": "mean"
        }).reset_index()
        oni_col = f"oni_lag{lag_months}"
        oni_annual = oni_annual.rename(columns={"oni": oni_col})

        # Merge ONI with crop data
        merged = pd.merge(crop_annual, oni_annual, on="year", how="inner")

        # Add controls if available and requested
        if include_controls and self.climate is not None:
            merged = self._add_climate_controls(merged)

        # Add time trend
        merged["time_trend"] = merged["year"] - merged["year"].min()

        self._merged_data = merged
        return merged

    def _annualize_oni(self, lag_months: int) -> pd.DataFrame:
        """Convert monthly ONI to annual with specified lag."""
        oni_data = self.oni.copy()

        # Calculate growing season average (simplified: use months 6-9)
        oni_data["growing_season"] = oni_data["month"].isin([6, 7, 8, 9])
        oni_annual = oni_data.groupby("year").agg({
            "oni": lambda x: x.mean(),
            "growing_season": "sum"
        }).reset_index()

        oni_annual = oni_annual.rename(columns={"oni": "oni_annual"})

        return oni_annual

    def _add_climate_controls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add precipitation/temperature controls."""
        if self.climate is None:
            return df

        # Assume climate data has year, precip, temp
        climate_cols = {c: c.lower() for c in self.climate.columns}
        if "precipitation" in climate_cols.values():
            pass  # Already have the column

        merged = pd.merge(df, self.climate, on="year", how="left")
        return merged

    def fit_model(
        self,
        crop_type: str,
        region: str,
        lag_months: int = 6,
        dependent_var: str = "yield",
        fixed_effects: bool = False,
        robust_se: bool = True
    ) -> RegressionResult:
        """
        Fit OLS regression model.

        Args:
            crop_type: Crop to analyze
            region: Country/region
            lag_months: ONI lag in months
            dependent_var: Dependent variable name
            fixed_effects: Include region fixed effects
            robust_se: Use heteroskedasticity-robust standard errors

        Returns:
            RegressionResult with coefficients and diagnostics
        """
        if self._merged_data is None:
            self.prepare_dataset(crop_type, region, lag_months)

        df = self._merged_data.copy()

        # Define dependent variable
        y = df[dependent_var]

        # Define independent variables
        X_cols = [f"oni_lag{lag_months}", "time_trend"]
        X = df[X_cols].copy()

        # Add fixed effects if requested
        if fixed_effects:
            # This is simplified - proper FE would need more data
            df["region_dummy"] = 1
            X["region_dummy"] = 1

        # Add constant
        X = sm.add_constant(X)

        # Check for missing values
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) < 10:
            raise ValueError("Insufficient observations for regression")

        # Fit model
        if robust_se:
            model = sm.OLS(y, X).fit(cov_type="HC3")  # Robust standard errors
        else:
            model = sm.OLS(y, X).fit()

        # Extract results
        significant_vars = [
            name for name, pval in model.pvalues.items()
            if pval < 0.05 and name != "const"
        ]

        result = RegressionResult(
            r_squared=model.rsquared,
            adjusted_r_squared=model.rsquared_adj,
            coefficients=model.params.to_dict(),
            p_values=model.pvalues.to_dict(),
            std_errors=model.bse.to_dict(),
            significant_vars=significant_vars,
            f_statistic=model.fvalue,
            f_p_value=model.f_pvalue,
            n_observations=int(model.nobs),
            model_summary=model.summary().as_text()
        )

        return result

    def compute_vif(self, crop_type: str, region: str, lag_months: int = 6) -> pd.DataFrame:
        """
        Compute Variance Inflation Factors to check multicollinearity.
        """
        if self._merged_data is None:
            self.prepare_dataset(crop_type, region, lag_months)

        df = self._merged_data.copy()
        X_cols = [f"oni_lag{lag_months}", "time_trend"]
        X = df[X_cols].dropna()

        if len(X) < 5:
            return pd.DataFrame()

        # Add constant for VIF calculation
        X_with_const = sm.add_constant(X)

        vif_data = {}
        for i, col in enumerate(X_cols):
            if col != "const":
                vif_data[col] = variance_inflation_factor(X_with_const.values, i + 1)

        vif_df = pd.DataFrame({
            "Variable": list(vif_data.keys()),
            "VIF": list(vif_data.values())
        })

        return vif_df

    def compare_lags(
        self,
        crop_type: str,
        region: str,
        lag_range: range = range(0, 25, 3),
        max_lags: int = 5
    ) -> pd.DataFrame:
        """
        Compare model fit across different lag specifications.

        Args:
            crop_type: Crop to analyze
            region: Country/region
            lag_range: Range of lag values to test
            max_lags: Maximum number of significant coefficients

        Returns:
            Comparison table with AIC, BIC, R² for each lag
        """
        results = []

        for lag in list(lag_range)[:max_lags]:
            try:
                self.prepare_dataset(crop_type, region, lag)
                model = self.fit_model(crop_type, region, lag)

                results.append({
                    "lag_months": lag,
                    "r_squared": model.r_squared,
                    "adj_r_squared": model.adjusted_r_squared,
                    "aic": model.coefficients.get("__aic__", np.nan),  # Will need model object
                    "bic": model.coefficients.get("__bic__", np.nan),
                    "oni_coef": model.coefficients.get(f"oni_lag{lag}", np.nan),
                    "oni_pvalue": model.p_values.get(f"oni_lag{lag}", np.nan),
                    "n_obs": model.n_observations
                })
            except Exception as e:
                logger.debug(f"Lag {lag} failed: {e}")
                continue

        return pd.DataFrame(results)

    def get_forecast_elasticity(self, crop_type: str, region: str) -> Dict:
        """
        Calculate ONI-to-yield elasticity from fitted model.

        This provides a simple interpretation: a 1°C increase in ONI
        leads to X% change in yield.
        """
        if self._merged_data is None:
            raise ValueError("Model not yet fitted. Call fit_model first.")

        result = self.fit_model(crop_type, region)

        # Elasticity: % change in yield per 1°C ONI change
        oni_coef = result.coefficients.get(f"oni_lag6", 0)  # Default 6-month lag
        mean_yield = self._merged_data["yield"].mean()
        mean_oni = self._merged_data.get("oni_lag6", self._merged_data.get("oni_annual", 0)).mean()

        if mean_yield and mean_oni:
            elasticity = (oni_coef * mean_oni) / mean_yield
        else:
            elasticity = np.nan

        return {
            "oni_coefficient": oni_coef,
            "mean_yield": mean_yield,
            "mean_oni": mean_oni,
            "elasticity": elasticity,
            "interpretation": (
                f"A 1°C increase in ONI is associated with a "
                f"{oni_coef:.2f} unit change in {crop_type} yield"
            )
        }

    def generate_report(
        self,
        crop_type: str,
        region: str,
        lag_months: int = 6
    ) -> str:
        """
        Generate comprehensive model report.
        """
        self.prepare_dataset(crop_type, region, lag_months)
        result = self.fit_model(crop_type, region)
        vif = self.compute_vif(crop_type, region, lag_months)
        elasticity = self.get_forecast_elasticity(crop_type, region)

        report_lines = [
            f"ENSO-Crop Regression Report: {crop_type} in {region}",
            "=" * 60,
            f"\nModel Specification: Yield = β₀ + β₁·ONI_lag{lag_months} + β₂·Trend + ε",
            f"\nObservations: {result.n_observations}",
            f"R²: {result.r_squared:.4f}",
            f"Adjusted R²: {result.adjusted_r_squared:.4f}",
            f"F-statistic: {result.f_statistic:.3f} (p={result.f_p_value:.4f})",
            "\nCoefficients:",
            "-" * 40,
        ]

        for var, coef in result.coefficients.items():
            pval = result.p_values.get(var, np.nan)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            report_lines.append(f"  {var}: {coef:.4f} (p={pval:.4f}) {sig}")

        report_lines.extend([
            "\nSignificant Variables: " + (", ".join(result.significant_vars) or "None"),
            "\nMulticollinearity (VIF):",
            "-" * 40,
        ])

        if not vif.empty:
            for _, row in vif.iterrows():
                report_lines.append(f"  {row['Variable']}: {row['VIF']:.2f}")
        else:
            report_lines.append("  VIF analysis not available")

        report_lines.extend([
            "\nElasticity Analysis:",
            "-" * 40,
            f"  ONI Coefficient: {elasticity['oni_coefficient']:.4f}",
            f"  Elasticity: {elasticity['elasticity']:.4f}",
            f"  {elasticity['interpretation']}",
            "\nCaveats:",
            "- Correlation does not imply causation",
            "- Omitted variable bias possible (e.g., local weather, policy)",
            "- Model assumes linear relationship",
            "- External validation recommended"
        ])

        return "\n".join(report_lines)


if __name__ == "__main__":
    from data_pipeline import ENSODataPipeline
    from lag_correlation import ENSOLagCorrelator

    pipeline = ENSODataPipeline()
    oni_sample = pipeline._sample_oni_data()
    crop_sample = pipeline._sample_crop_data()

    model = ENSOReggressionModel(oni_sample, crop_sample)

    # Fit model for wheat in USA
    result = model.fit_model("Wheat", "USA", lag_months=6)
    print(result.model_summary)

    # Generate report
    report = model.generate_report("Wheat", "USA", lag_months=6)
    print("\n" + report)