"""
Interactive Streamlit Dashboard for ENSO Analytics

Features:
- ONI time series with phase annotations
- Lag correlation explorer
- Regression results visualization
- Forecast with confidence intervals
- Scenario analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import ENSODataPipeline
from src.lag_correlation import ENSOLagCorrelator
from src.regression_model import ENSOReggressionModel
from src.forecast import ENSOForecaster


def setup_page():
    """Configure Streamlit page."""
    st.set_page_config(
        page_title="ENSO Analytics Dashboard",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )


@st.cache_data
def load_data():
    """Load and cache all data."""
    pipeline = ENSODataPipeline()
    oni = pipeline._sample_oni_data()
    crop = pipeline._sample_crop_data()
    return oni, crop


def plot_oni_timeseries(oni_df: pd.DataFrame, ax=None):
    """Plot ONI time series with phase annotations."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    colors = {"El Nino": "red", "Neutral": "gray", "La Nina": "blue"}

    for phase, color in colors.items():
        mask = oni_df["enso_phase"] == phase
        ax.scatter(
            oni_df.loc[mask, "date"],
            oni_df.loc[mask, "oni"],
            c=color,
            alpha=0.5,
            s=10,
            label=phase
        )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(y=-0.5, color="blue", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("ONI (°C)")
    ax.set_title("Oceanic Niño Index (ONI) Time Series")
    ax.legend(loc="upper right")

    return ax


def plot_lag_correlations(correlations: pd.DataFrame, ax=None):
    """Plot lag correlation results."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["darkgreen" if sig else "steelblue"
              for sig in correlations["significant"]]

    ax.bar(correlations["lag_months"], correlations["correlation"], color=colors)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.axhline(y=0.3, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(y=-0.3, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Correlation")
    ax.set_title("ONI-Crop Yield Correlation by Lag")
    ax.set_xlim(-1, correlations["lag_months"].max() + 1)

    return ax


def plot_forecast(forecast_result, ax=None):
    """Plot ONI forecast with confidence intervals."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(forecast_result.forecast_dates, forecast_result.forecast_values,
            "b-", linewidth=2, label="Forecast")
    ax.fill_between(
        forecast_result.forecast_dates,
        forecast_result.lower_ci,
        forecast_result.upper_ci,
        alpha=0.2,
        color="blue",
        label="95% CI"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("ONI (°C)")
    ax.set_title("ONI Forecast (12 months)")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    return ax


def main():
    """Main dashboard function."""
    setup_page()

    # Load data
    oni_df, crop_df = load_data()

    # Sidebar
    st.sidebar.title("🌊 ENSO Analytics")
    st.sidebar.markdown("---")

    # Data sources info
    st.sidebar.info(
        "**Data Sources:**\n"
        "- NOAA CPC (ONI Index)\n"
        "- NOAA PSL (PDO, SOI)\n"
        "- FAO (Crop Production)\n"
        "- EM-DAT (Disaster Events)"
    )

    # Tab selection
    tab = st.sidebar.radio(
        "Navigation",
        ["ONI Overview", "Lag Analysis", "Regression", "Forecasting", "Data Documentation"]
    )

    # Main content
    if tab == "ONI Overview":
        st.title("🌊 Oceanic Niño Index Overview")

        # KPI row
        col1, col2, col3, col4 = st.columns(4)

        latest = oni_df.iloc[-1]
        current_phase = latest["enso_phase"]
        current_oni = latest["oni"]

        col1.metric("Current ONI", f"{current_oni:.2f}°C")
        col2.metric("Current Phase", current_phase,
                    delta="El Niño" if current_phase == "El Nino" else ("La Niña" if current_phase == "La Nina" else None))
        col3.metric("Mean ONI", f"{oni_df['oni'].mean():.2f}°C")
        col4.metric("Max El Niño", f"{oni_df[oni_df['oni'] > 0.5]['oni'].max():.2f}°C" if (oni_df['oni'] > 0.5).any() else "N/A")

        # Time series plot
        st.subheader("ONI Time Series")
        fig, ax = plt.subplots(figsize=(12, 4))
        plot_oni_timeseries(oni_df, ax)
        st.pyplot(fig)

        # Phase distribution
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Phase Distribution")
            phase_counts = oni_df["enso_phase"].value_counts()
            fig, ax = plt.subplots()
            ax.pie(phase_counts, labels=phase_counts.index, autopct="%1.1f%%",
                   colors=["blue" if p == "La Nina" else ("red" if p == "El Nino" else "gray")
                           for p in phase_counts.index])
            st.pyplot(fig)

        with col2:
            st.subheader("ONI Statistics")
            st.write(oni_df["oni"].describe())

    elif tab == "Lag Analysis":
        st.title("📊 Lag Correlation Analysis")

        st.markdown("""
        **Understanding ENSO Effects:** The impact of El Niño/La Niña on agriculture
        is not immediate. Effects manifest with a delay of 3-12 months depending on
        crop type, region, and local conditions.
        """)

        # Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            crop_type = st.selectbox("Crop Type", crop_df["crop"].unique())
        with col2:
            region = st.selectbox("Region", crop_df["country"].unique())
        with col3:
            max_lag = st.slider("Maximum Lag (months)", 6, 36, 18)

        # Compute correlations
        correlator = ENSOLagCorrelator(oni_df, crop_df)
        correlations = correlator.compute_lag_correlations(crop_type, region, max_lag)

        if not correlations.empty:
            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            plot_lag_correlations(correlations, ax)
            st.pyplot(fig)

            # Optimal lag info
            correlations["abs_corr"] = correlations["correlation"].abs()
            optimal = correlations.loc[correlations["abs_corr"].idxmax()]

            st.info(f"""
            **Optimal Lag:** {int(optimal['lag_months'])} months
            **Correlation:** {optimal['correlation']:.3f} (p={optimal['p_value']:.4f})
            **Interpretation:** {correlator._interpret_correlation(optimal['correlation'], optimal['significant'])}
            """)

            # Data table
            st.subheader("Correlation at Each Lag")
            st.dataframe(correlations[["lag_months", "correlation", "p_value", "significant", "n_observations"]])

        else:
            st.warning("Insufficient data for this crop/region combination.")

    elif tab == "Regression":
        st.title("📈 ENSO-Crop Regression Model")

        st.markdown("""
        **Model Specification:**
        ```
        Yield = β₀ + β₁·ONI_lag + β₂·Trend + ε
        ```
        OLS regression with robust standard errors (HC3).
        """)

        # Controls
        col1, col2 = st.columns(2)
        with col1:
            crop_type = st.selectbox("Crop", crop_df["crop"].unique(), key="reg_crop")
        with col2:
            region = st.selectbox("Region", crop_df["country"].unique(), key="reg_region")

        lag_months = st.slider("ONI Lag (months)", 0, 18, 6)

        # Fit model
        try:
            model = ENSOReggressionModel(oni_df, crop_df)
            result = model.fit_model(crop_type, region, lag_months)

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R²", f"{result.r_squared:.3f}")
            col2.metric("Adj R²", f"{result.adjusted_r_squared:.3f}")
            col3.metric("F-statistic", f"{result.f_statistic:.2f}")
            col4.metric("Observations", result.n_observations)

            # Coefficients plot
            fig, ax = plt.subplots(figsize=(10, 4))
            vars_list = [k for k in result.coefficients.keys() if k != "const"]
            coefs = [result.coefficients[k] for k in vars_list]
            errors = [result.std_errors[k] * 1.96 for k in vars_list]

            colors = ["green" if result.p_values[k] < 0.05 else "gray" for k in vars_list]
            ax.barh(vars_list, coefs, xerr=errors, color=colors, alpha=0.7)
            ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
            ax.set_xlabel("Coefficient Estimate")
            ax.set_title(f"Regression Coefficients (95% CI)")
            st.pyplot(fig)

            # VIF
            vif = model.compute_vif(crop_type, region, lag_months)
            if not vif.empty:
                st.subheader("Variance Inflation Factors")
                st.dataframe(vif)

            # Model summary
            with st.expander("Full Model Summary"):
                st.text(result.model_summary)

        except Exception as e:
            st.error(f"Model fitting failed: {e}")

    elif tab == "Forecasting":
        st.title("🔮 ENSO & Yield Forecasting")

        st.markdown("""
        **Forecasting Approach:**
        - ONI: Holt-Winters exponential smoothing with damped trend
        - Yield: Linear projection using historical elasticity
        - Backtest: Rolling-origin validation
        """)

        # Forecast horizon
        horizon = st.slider("Forecast Horizon (months)", 3, 24, 12)

        # Run forecast
        forecaster = ENSOForecaster(oni_df, crop_df)
        oni_forecast = forecaster.forecast_oni(horizon=horizon)

        # Plot forecast
        fig, ax = plt.subplots(figsize=(12, 5))

        # Historical
        ax.plot(oni_df["date"].iloc[-60:], oni_df["oni"].iloc[-60:],
                "b-", linewidth=1, label="Historical")
        # Forecast
        ax.plot(oni_forecast.forecast_dates, oni_forecast.forecast_values,
                "r-", linewidth=2, label="Forecast")
        ax.fill_between(
            oni_forecast.forecast_dates,
            oni_forecast.lower_ci,
            oni_forecast.upper_ci,
            alpha=0.2,
            color="red",
            label="95% CI"
        )
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("ONI (°C)")
        ax.set_title("ONI Forecast with Confidence Intervals")
        ax.legend()
        st.pyplot(fig)

        # Backtest results
        st.subheader("Backtest Performance")
        backtest = forecaster.backtest_oni()

        if backtest:
            backtest_df = pd.DataFrame(backtest).T
            backtest_df.index.name = "Horizon"
            st.dataframe(backtest_df.style.format("{:.3f}"))

        # Crop yield projection
        st.subheader("Yield Scenario Analysis")
        col1, col2 = st.columns(2)
        with col1:
            crop_type = st.selectbox("Crop", crop_df["crop"].unique(), key="fc_crop")
        with col2:
            region = st.selectbox("Region", crop_df["country"].unique(), key="fc_region")

        if st.button("Generate Yield Projections"):
            projections = forecaster.project_crop_yield(oni_forecast, crop_type, region)

            st.write(f"**Base Yield:** {projections['base_yield']:.1f}")
            st.write(f"**Elasticity:** {projections['elasticity']:.3f}")

            fig, ax = plt.subplots(figsize=(10, 4))
            for scenario, yields in projections["scenarios"].items():
                ax.plot(projections["dates"], yields, label=scenario, linewidth=2)
            ax.axhline(y=projections["base_yield"], color="gray", linestyle="--",
                       label="Base Yield")
            ax.set_xlabel("Date")
            ax.set_ylabel("Projected Yield")
            ax.set_title(f"{crop_type} Yield Scenarios in {region}")
            ax.legend()
            st.pyplot(fig)

    elif tab == "Data Documentation":
        st.title("📋 Data Sources & Documentation")

        st.markdown("""
        ## Primary Data Sources

        | Source | Data Type | URL | Update Frequency |
        |--------|-----------|-----|------------------|
        | NOAA CPC - ONI Index | Oceanic Niño Index | https://www.cpc.ncep.noaa.gov/data/indices/oni.shtml | Weekly |
        | NOAA PSL | PDO, SOI, AO | https://www.psl.noaa.gov/data/climateindices/list/ | Monthly |
        | FAO - FAOSTAT | Crop production, yields | https://www.fao.org/faostat/en/#data | Annual |
        | EM-DAT | Disaster events | https://www.emdat.be/ | Event-based |

        ## Limitations & Caveats

        1. **ONI Data:** 3-month lag for finalized values; historical discontinuities in ERSSTv5
        2. **Correlation ≠ Causation:** ENSO affects agriculture through multiple pathways
        3. **Regional Variation:** ENSO impacts vary significantly by region and crop
        4. **Model Uncertainty:** Forecasts are probabilistic; actual outcomes may differ

        ## Methodology Notes

        - **Lag Correlation:** Pearson correlation at various lag periods (0-24 months)
        - **Regression:** OLS with robust standard errors (HC3)
        - **Forecasting:** Holt-Winters exponential smoothing
        - **Backtesting:** Rolling-origin with expanding window
        """)

        st.info("""
        **Citation:** If using this analysis, please cite:
        SAI-798 (2026). ENSO Analytics: Monitoring Climate Anomalies,
        Economic Losses, and Population Vulnerability. GitHub Repository.
        """)


if __name__ == "__main__":
    main()