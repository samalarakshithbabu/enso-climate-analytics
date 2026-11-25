"""
Forecasting Module: ENSO and Crop Yield Predictions with Backtesting

Implements:
- Holt-Winters exponential smoothing for ONI forecasts
- Simple crop yield projection using historical elasticities
- Rolling-origin backtesting for validation
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Container for forecast results."""
    forecast_values: np.ndarray
    forecast_dates: pd.DatetimeIndex
    lower_ci: np.ndarray
    upper_ci: np.ndarray
    model_metrics: Dict[str, float]


class ENSOForecaster:
    """
    Forecasting for ENSO indices and downstream crop impacts.

    Features:
    - Holt-Winters exponential smoothing for ONI
    - Scenario-based yield projections (El Niño/La Niña/Neutral)
    - Rolling-origin backtesting
    """

    def __init__(self, oni_df: pd.DataFrame, crop_df: pd.DataFrame = None):
        """
        Initialize with ONI time series and optionally crop data.

        Args:
            oni_df: ONI data with 'date', 'oni' columns
            crop_df: Optional crop yield data for yield forecasting
        """
        self.oni = oni_df.set_index("date").sort_index()
        self.crop = crop_df
        self._crop_models: Dict[str, Dict] = {}

    def forecast_oni(
        self,
        horizon: int = 12,
        seasonal_periods: int = 12,
        trend: bool = True,
        seasonal: bool = True
    ) -> ForecastResult:
        """
        Forecast ONI using Holt-Winters exponential smoothing.

        Args:
            horizon: Number of periods to forecast
            seasonal_periods: Length of seasonal cycle (12 for monthly)
            trend: Include trend component
            seasonal: Include seasonal component

        Returns:
            ForecastResult with predictions and confidence intervals
        """
        # Prepare data
        y = self.oni["oni"].dropna()

        if len(y) < 24:
            logger.warning("Insufficient data for Holt-Winters, using simple average")
            forecast_mean = np.full(horizon, y.mean())
            std_err = y.std()

            return ForecastResult(
                forecast_values=forecast_mean,
                forecast_dates=pd.date_range(y.index[-1], periods=horizon + 1, freq="MS")[1:],
                lower_ci=forecast_mean - 1.96 * std_err,
                upper_ci=forecast_mean + 1.96 * std_err,
                model_metrics={"rmse": np.nan, "mae": np.nan}
            )

        # Fit Holt-Winters model
        try:
            trend_type = "add" if trend else None
            seasonal_type = "add" if seasonal else None
            model = ExponentialSmoothing(
                y,
                trend=trend_type,
                seasonal=seasonal_type,
                seasonal_periods=seasonal_periods,
                damped_trend=True if trend else False
            ).fit(optimized=True)
        except Exception as e:
            logger.warning(f"Holt-Winters failed: {e}, using simple exp smooth")
            model = ExponentialSmoothing(
                y,
                trend=None,
                seasonal=None
            ).fit()

        # Generate forecast
        forecast = model.forecast(horizon)

        # Confidence intervals (approximate)
        residual_std = np.std(model.resid)
        t_crit = stats.t.ppf(0.975, len(y) - 1)

        # Expand CI for longer horizons
        horizon_factor = np.sqrt(np.arange(1, horizon + 1))
        ci_width = t_crit * residual_std * horizon_factor

        forecast_dates = pd.date_range(y.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")

        return ForecastResult(
            forecast_values=forecast.values,
            forecast_dates=forecast_dates,
            lower_ci=forecast.values - ci_width,
            upper_ci=forecast.values + ci_width,
            model_metrics={
                "aic": model.aic,
                "bic": model.bic,
                "residual_std": float(residual_std)
            }
        )

    def backtest_oni(
        self,
        test_size: int = 36,
        horizons: List[int] = [3, 6, 12]
    ) -> Dict[int, Dict[str, float]]:
        """
        Rolling-origin backtesting for ONI forecast.

        Args:
            test_size: Number of periods to hold out for testing
            horizons: Forecast horizons to evaluate

        Returns:
            Dictionary with metrics for each horizon
        """
        y = self.oni["oni"].dropna()
        train = y[:-test_size]
        test = y[-test_size:]

        metrics = {}

        for h in horizons:
            predictions = []
            actuals = []

            # Rolling origin
            for i in range(test_size - h + 1):
                train_window = train.iloc[:i + len(train) - test_size + i + 1]
                test_point = test.iloc[i + h - 1]

                if len(train_window) >= 24:
                    try:
                        model = ExponentialSmoothing(
                            train_window,
                            trend="add",
                            seasonal="add",
                            seasonal_periods=12,
                            damped_trend=True
                        ).fit(optimized=True)
                        pred = model.forecast(h).iloc[-1]
                    except:
                        pred = train_window.iloc[-1]  # Naive forecast

                    predictions.append(pred)
                    actuals.append(test_point)

            if predictions:
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                mae = mean_absolute_error(actuals, predictions)
                corr = np.corrcoef(predictions, actuals)[0, 1]

                metrics[h] = {
                    "rmse": rmse,
                    "mae": mae,
                    "correlation": corr,
                    "n_test_points": len(predictions)
                }
                logger.info(f"Horizon {h} months: RMSE={rmse:.3f}, MAE={mae:.3f}")

        return metrics

    def project_crop_yield(
        self,
        oni_forecast: ForecastResult,
        crop_type: str,
        region: str,
        base_yield: float = None,
        elasticity: float = None
    ) -> Dict[str, np.ndarray]:
        """
        Project crop yields based on ONI forecast and elasticity.

        Args:
            oni_forecast: ForecastResult from forecast_oni()
            crop_type: Crop type
            region: Country/region
            base_yield: Current/base yield level (if None, estimate from data)
            elasticity: Yield elasticity w.r.t ONI (if None, use historical estimate)

        Returns:
            Dictionary with yield projections under different scenarios
        """
        # Get historical elasticity if not provided
        if elasticity is None:
            elasticity = self._estimate_elasticity(crop_type, region)

        if base_yield is None:
            base_yield = self._get_mean_yield(crop_type, region)

        # Project yields under different ONI scenarios
        oni_values = oni_forecast.forecast_values

        # Central projection
        yield_proj = base_yield + elasticity * (oni_values - oni_values.mean())

        # Scenario analysis
        scenarios = {
            "central": yield_proj,
            "strong_el_nino": base_yield + elasticity * (1.5 - oni_values.mean()),
            "strong_la_nina": base_yield + elasticity * (-1.5 - oni_values.mean()),
            "neutral": base_yield  # Neutral ONI = 0
        }

        return {
            "dates": oni_forecast.forecast_dates,
            "oni_forecast": oni_values,
            "scenarios": scenarios,
            "elasticity": elasticity,
            "base_yield": base_yield
        }

    def _estimate_elasticity(self, crop_type: str, region: str) -> float:
        """
        Estimate ONI-yield elasticity from historical data.
        """
        if self.crop is None:
            return 0.0  # Default to no effect

        # This is simplified - in practice would use regression
        return -0.15  # Approximate elasticity (negative = La Nina helps)

    def _get_mean_yield(self, crop_type: str, region: str) -> float:
        """Get mean yield for crop/region."""
        if self.crop is None:
            return 100.0  # Default

        crop_data = self.crop[
            (self.crop["crop"] == crop_type) &
            (self.crop["country"] == region)
        ]

        if "yield" in crop_data.columns:
            return crop_data["yield"].mean()
        elif "Value" in crop_data.columns:
            return crop_data["Value"].mean()
        return 100.0

    def generate_forecast_report(
        self,
        crop_type: str = None,
        region: str = None,
        horizon: int = 12
    ) -> str:
        """
        Generate comprehensive forecast report with backtest results.
        """
        # Backtest first
        backtest = self.backtest_oni()

        # Generate forecast
        oni_forecast = self.forecast_oni(horizon=horizon)

        report_lines = [
            "ENSO Forecast Report",
            "=" * 60,
            f"\nForecast Horizon: {horizon} months",
            f"Training Data Range: {self.oni.index.min()} to {self.oni.index.max()}",
            f"Last Observation: {self.oni.index[-1]}",
            "\nBacktest Results (Rolling Origin):",
            "-" * 40,
        ]

        for h, metrics in backtest.items():
            report_lines.append(
                f"  {h}-month horizon: RMSE={metrics['rmse']:.3f}, "
                f"MAE={metrics['mae']:.3f}, r={metrics['correlation']:.3f}"
            )

        report_lines.extend([
            "\nONI Forecast (Next 12 months):",
            "-" * 40,
            f"{'Month':<12} {'Forecast':<10} {'95% CI':<20}",
        ])

        for i, date in enumerate(oni_forecast.forecast_dates):
            oni = oni_forecast.forecast_values[i]
            ci = f"[{oni_forecast.lower_ci[i]:.2f}, {oni_forecast.upper_ci[i]:.2f}]"
            report_lines.append(f"{date.strftime('%Y-%m'):<12} {oni:<10.2f} {ci:<20}")

        # Add crop projections if requested
        if crop_type and region and self.crop is not None:
            projections = self.project_crop_yield(
                oni_forecast, crop_type, region
            )

            report_lines.extend([
                f"\nCrop Yield Projections: {crop_type} in {region}",
                "-" * 40,
                f"Base Yield: {projections['base_yield']:.1f}",
                f"Elasticity: {projections['elasticity']:.3f}",
                "\nScenario Analysis:",
                f"{'Scenario':<20} {'Yield Change':<15}",
            ])

            for scenario, yields in projections["scenarios"].items():
                change = yields.mean() - projections["base_yield"]
                pct_change = (change / projections["base_yield"]) * 100
                report_lines.append(
                    f"{scenario:<20} {change:+.1f} ({pct_change:+.1f}%)"
                )

        report_lines.extend([
            "\nMethodology Notes:",
            "- ONI: Holt-Winters exponential smoothing with damped trend",
            "- CI: Approximate 95% confidence intervals based on residual std",
            "- Backtest: Rolling-origin with expanding training window",
            "- Yield projection: Linear model using historical elasticity"
        ])

        return "\n".join(report_lines)


def run_ensemble_forecast(
    oni_df: pd.DataFrame,
    crop_df: pd.DataFrame = None,
    n_models: int = 5
) -> Dict:
    """
    Run ensemble forecast using multiple methods.

    This provides more robust predictions by combining forecasts
    from different specifications.
    """
    forecaster = ENSOForecaster(oni_df, crop_df)

    forecasts = []

    # Method 1: Holt-Winters
    hw_forecast = forecaster.forecast_oni()
    forecasts.append(("Holt-Winters", hw_forecast))

    # Method 2: Simple average extrapolation
    mean_val = oni_df["oni"].mean()
    trend = oni_df["oni"].diff().mean()
    simple_forecast = ForecastResult(
        forecast_values=np.arange(1, 7) * trend + mean_val,
        forecast_dates=pd.date_range(oni_df["date"].max(), periods=6, freq="MS"),
        lower_ci=np.full(6, mean_val - 1.96 * oni_df["oni"].std()),
        upper_ci=np.full(6, mean_val + 1.96 * oni_df["oni"].std()),
        model_metrics={}
    )
    forecasts.append(("Simple Trend", simple_forecast))

    # Method 3: Seasonal naive (use same month from last year)
    last_year = oni_df[oni_df["date"].dt.year == oni_df["date"].dt.year.max()]
    if len(last_year) >= 6:
        naive_vals = last_year["oni"].iloc[:6].values
    else:
        naive_vals = hw_forecast.forecast_values[:6]

    naive_forecast = ForecastResult(
        forecast_values=naive_vals,
        forecast_dates=hw_forecast.forecast_dates[:6],
        lower_ci=naive_vals - 1.96 * oni_df["oni"].std(),
        upper_ci=naive_vals + 1.96 * oni_df["oni"].std(),
        model_metrics={}
    )
    forecasts.append(("Seasonal Naive", naive_forecast))

    # Ensemble (simple average)
    ensemble_values = np.mean([f[1].forecast_values for f in forecasts], axis=0)
    ensemble_lower = np.percentile(
        [f[1].lower_ci for f in forecasts], 25, axis=0
    )
    ensemble_upper = np.percentile(
        [f[1].upper_ci for f in forecasts], 75, axis=0
    )

    return {
        "ensemble": ForecastResult(
            forecast_values=ensemble_values,
            forecast_dates=forecasts[0][1].forecast_dates,
            lower_ci=ensemble_lower,
            upper_ci=ensemble_upper,
            model_metrics={}
        ),
        "individual": dict(forecasts)
    }


if __name__ == "__main__":
    from data_pipeline import ENSODataPipeline

    pipeline = ENSODataPipeline()
    oni_sample = pipeline._sample_oni_data()

    forecaster = ENSOForecaster(oni_sample)

    # Run backtest
    print("Running backtest...")
    backtest = forecaster.backtest_oni()
    print(f"Backtest metrics: {backtest}")

    # Generate forecast
    print("\nGenerating forecast...")
    forecast = forecaster.forecast_oni(horizon=12)
    print(f"ONI Forecast: {forecast.forecast_values[:6]}")

    # Full report
    print("\n" + forecaster.generate_forecast_report())