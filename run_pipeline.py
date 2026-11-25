#!/usr/bin/env python
"""
ENSO Analytics - Main Entry Point

Usage:
    python run_pipeline.py              # Run full ETL pipeline
    python run_pipeline.py --lag        # Run lag correlation analysis
    python run_pipeline.py --regression # Run regression analysis
    python run_pipeline.py --forecast   # Run forecasting with backtest
    python run_pipeline.py --dashboard  # Launch Streamlit dashboard
    python run_pipeline.py --all        # Run all analyses
"""

import argparse
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data_pipeline import ENSODataPipeline
from lag_correlation import ENSOLagCorrelator
from regression_model import ENSOReggressionModel
from forecast import ENSOForecaster


def run_pipeline(args):
    """Run the full ETL pipeline."""
    print("=" * 60)
    print("ENSO Data Pipeline")
    print("=" * 60)

    pipeline = ENSODataPipeline()
    stats = pipeline.run_full_pipeline()

    print(f"\nPipeline complete:")
    print(f"  - ONI records: {stats['oni_records']}")
    print(f"  - PDO records: {stats['pdo_records']}")
    print(f"  - Crop records: {stats['crop_records']}")


def run_lag_analysis(args):
    """Run lag correlation analysis."""
    print("=" * 60)
    print("ENSO-Crop Lag Correlation Analysis")
    print("=" * 60)

    pipeline = ENSODataPipeline()
    oni = pipeline._sample_oni_data()
    crop = pipeline._sample_crop_data()

    correlator = ENSOLagCorrelator(oni, crop)

    # Analyze specified crops/regions
    pairs = [
        ("Wheat", "USA"),
        ("Rice", "India"),
        ("Soybeans", "Brazil"),
        ("Corn", "USA"),
    ]

    if args.crop:
        pairs = [(args.crop, args.region or "USA")]

    summary = correlator.create_lag_summary_report(pairs, max_lag=args.max_lag)
    print("\nLag Correlation Summary:")
    print(summary.to_string(index=False))

    # Save results
    output_path = Path(__file__).parent / "data" / "output" / "lag_correlations.csv"
    output_path.parent.mkdir(exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def run_regression(args):
    """Run regression analysis."""
    print("=" * 60)
    print("ENSO-Crop Regression Analysis")
    print("=" * 60)

    pipeline = ENSODataPipeline()
    oni = pipeline._sample_oni_data()
    crop = pipeline._sample_crop_data()

    model = ENSOReggressionModel(oni, crop)

    # Analyze specified crops
    crop_region_pairs = [
        ("Wheat", "USA"),
        ("Rice", "India"),
        ("Soybeans", "Brazil"),
    ]

    if args.crop:
        crop_region_pairs = [(args.crop, args.region or "USA")]

    for crop_type, region in crop_region_pairs:
        try:
            print(f"\n--- {crop_type} in {region} ---")
            result = model.fit_model(crop_type, region, lag_months=args.lag_months)
            print(f"R²: {result.r_squared:.4f}")
            print(f"ONI coefficient: {result.coefficients.get(f'oni_lag{args.lag}', 'N/A')}")

            # Generate full report
            report = model.generate_report(crop_type, region, lag_months=args.lag_months)
            print("\n" + report)

        except Exception as e:
            print(f"Analysis failed for {crop_type}/{region}: {e}")


def run_forecast(args):
    """Run forecasting with backtesting."""
    print("=" * 60)
    print("ENSO Forecasting & Backtesting")
    print("=" * 60)

    pipeline = ENSODataPipeline()
    oni = pipeline._sample_oni_data()

    forecaster = ENSOForecaster(oni)

    # Run backtest
    print("\nRunning backtest...")
    backtest = forecaster.backtest_oni()
    print("Backtest Results:")
    for h, metrics in backtest.items():
        print(f"  {h}-month horizon: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}")

    # Generate forecast
    print("\nGenerating forecast...")
    forecast = forecaster.forecast_oni(horizon=args.horizon)
    print(f"Forecast horizon: {args.horizon} months")
    print(f"First 6 forecasted values: {forecast.forecast_values[:6]}")

    # Full report
    report = forecaster.generate_forecast_report(
        crop_type=args.crop, region=args.region
    )
    print("\n" + report)


def run_dashboard(args):
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / "notebooks" / "dashboard.py"
    print(f"Launching dashboard: {dashboard_path}")
    subprocess.run(["streamlit", "run", str(dashboard_path)])


def main():
    parser = argparse.ArgumentParser(
        description="ENSO Analytics - Climate Intelligence Dashboard"
    )
    parser.add_argument(
        "--lag", action="store_true",
        help="Run lag correlation analysis"
    )
    parser.add_argument(
        "--regression", action="store_true",
        help="Run regression analysis"
    )
    parser.add_argument(
        "--forecast", action="store_true",
        help="Run forecasting with backtest"
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Launch Streamlit dashboard"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all analyses"
    )
    parser.add_argument(
        "--crop", type=str, default=None,
        help="Crop type for analysis"
    )
    parser.add_argument(
        "--region", type=str, default=None,
        help="Region/country for analysis"
    )
    parser.add_argument(
        "--max-lag", type=int, default=24,
        help="Maximum lag for correlation analysis"
    )
    parser.add_argument(
        "--lag-months", type=int, default=6,
        help="Lag months for regression"
    )
    parser.add_argument(
        "--horizon", type=int, default=12,
        help="Forecast horizon in months"
    )

    args = parser.parse_args()

    # Default: run pipeline
    if not any([args.lag, args.regression, args.forecast, args.dashboard, args.all]):
        run_pipeline(args)
        return

    if args.all or args.lag:
        run_lag_analysis(args)
    if args.all or args.regression:
        run_regression(args)
    if args.all or args.forecast:
        run_forecast(args)
    if args.dashboard:
        run_dashboard(args)


if __name__ == "__main__":
    main()