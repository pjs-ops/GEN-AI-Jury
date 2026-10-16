"""
visualization.py
Interactive Plotly charts for the Streamlit dashboard and publication-quality
Matplotlib visualizations for the Jupyter notebook.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None


# Corporate color palette for retail dashboard
PALETTE = {
    "actual": "#1f77b4",       # Solid blue
    "forecast": "#ff7f0e",     # Vibrant amber/orange
    "baseline": "#7f7f7f",     # Charcoal gray
    "holiday": "#d62728",      # Crimson red
    "interval": "rgba(255, 127, 14, 0.2)",
    "rolling": "#2ca02c"       # Forest green
}


def plot_sales_trend_plotly(
    df: pd.DataFrame,
    title: str = "Weekly Sales Trend & Rolling Average"
) -> go.Figure:
    """Creates an interactive Plotly sales trend chart with rolling average."""
    fig = go.Figure()

    # Calculate 4-week rolling average backward using only current and previous observations (no lookahead)
    if "Rolling_Mean_4" in df.columns and not df["Rolling_Mean_4"].isnull().all():
        rolling_series = df["Rolling_Mean_4"]
    else:
        rolling_series = df["Weekly_Sales"].rolling(window=4, min_periods=1).mean()

    # 1. Actual weekly sales line
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Weekly_Sales"],
        mode="lines+markers",
        name="Weekly Sales ($)",
        line=dict(color=PALETTE["actual"], width=2),
        marker=dict(size=4),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Weekly Sales:</b> $%{y:,.2f}<extra></extra>"
    ))

    # 2. 4-Week Rolling Average line (clearly distinguishable emerald line)
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=rolling_series,
        mode="lines",
        name="4-Week Rolling Avg",
        line=dict(color="#00cc96", width=2.5, dash="dash"),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>4-Week Rolling Avg:</b> $%{y:,.2f}<extra></extra>"
    ))

    # 3. Highlight holiday weeks
    if "IsHoliday" in df.columns:
        holidays = df[df["IsHoliday"] == 1]
        fig.add_trace(go.Scatter(
            x=holidays["Date"],
            y=holidays["Weekly_Sales"],
            mode="markers",
            name="Holiday Weeks",
            marker=dict(color=PALETTE["holiday"], size=8, symbol="diamond"),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Holiday Sales:</b> $%{y:,.2f}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Date", showgrid=True),
        yaxis=dict(title="Weekly Sales (USD)", showgrid=True, tickprefix="$", tickformat=","),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_actual_vs_predicted_plotly(
    dates: pd.Series,
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    selected_model: str,
    baseline_model: Optional[str] = None
) -> go.Figure:
    """Plots actual sales vs predicted sales for selected model and optional baseline."""
    fig = go.Figure()

    # Actual Sales
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_true,
        mode="lines+markers",
        name="Actual Sales",
        line=dict(color=PALETTE["actual"], width=2.5),
        marker=dict(size=5)
    ))

    # Baseline Model if provided
    if baseline_model and baseline_model in predictions_dict:
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions_dict[baseline_model],
            mode="lines",
            name=f"Baseline: {baseline_model}",
            line=dict(color=PALETTE["baseline"], width=1.8, dash="dot")
        ))

    # Selected Model
    if selected_model in predictions_dict:
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions_dict[selected_model],
            mode="lines+markers",
            name=f"Selected: {selected_model}",
            line=dict(color=PALETTE["forecast"], width=2.5),
            marker=dict(size=5)
        ))

    fig.update_layout(
        title=f"Holdout Actual vs. Predicted Demand: {selected_model}",
        xaxis_title="Date",
        yaxis_title="Weekly Sales (USD)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_future_forecast_plotly(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    history_tail: int = 26
) -> go.Figure:
    """Plots recent historical sales connected seamlessly to the future forecast with confidence interval."""
    fig = go.Figure()

    recent_history = history_df.tail(history_tail).copy()

    # Historical line
    fig.add_trace(go.Scatter(
        x=recent_history["Date"],
        y=recent_history["Weekly_Sales"],
        mode="lines+markers",
        name="Historical Sales",
        line=dict(color=PALETTE["actual"], width=2.2),
        marker=dict(size=5)
    ))

    # Forecast upper bound
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Upper_Bound ($)"],
        mode="lines",
        name="Upper Bound (95% PI)",
        line=dict(width=0),
        showlegend=False
    ))

    # Forecast lower bound (fills area between upper and lower)
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Lower_Bound ($)"],
        mode="lines",
        name="95% Prediction Interval",
        fill="tonexty",
        fillcolor=PALETTE["interval"],
        line=dict(width=0)
    ))

    # Point Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast ($)"],
        mode="lines+markers",
        name="Point Forecast",
        line=dict(color=PALETTE["forecast"], width=3, dash="dash"),
        marker=dict(size=7, symbol="circle")
    ))

    fig.update_layout(
        title="Multi-Step Future Demand Forecast with 95% Prediction Bounds",
        xaxis_title="Date",
        yaxis_title="Weekly Sales (USD)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_leaderboard_bar_plotly(
    leaderboard_df: pd.DataFrame,
    metric_col: str = "WAPE (%)"
) -> go.Figure:
    """Creates a horizontal ranked bar chart of model evaluation performance."""
    valid_df = leaderboard_df[leaderboard_df["Status"] == "Valid"].copy()
    valid_df = valid_df.sort_values(by=metric_col, ascending=False)

    fig = px.bar(
        valid_df,
        x=metric_col,
        y="Model",
        orientation="h",
        color=metric_col,
        color_continuous_scale="Blues_r",
        text=metric_col,
        title=f"Model Performance Comparison by {metric_col}"
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title=metric_col,
        yaxis_title="",
        coloraxis_showscale=False
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    return fig


def plot_residuals_plotly(analysis_table: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """Generates residual error over time and residual distribution charts."""
    # Chart 1: Residuals over time
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=analysis_table["Date"],
        y=analysis_table["Error"],
        mode="lines+markers",
        name="Residual (Actual - Pred)",
        line=dict(color="#d62728", width=1.8),
        marker=dict(size=6)
    ))
    fig_time.add_hline(y=0, line_dash="dash", line_color="black")
    fig_time.update_layout(
        title="Forecast Residuals Over Time",
        xaxis_title="Date",
        yaxis_title="Residual Error ($)",
        template="plotly_white"
    )

    # Chart 2: Residual distribution
    fig_dist = px.histogram(
        analysis_table,
        x="Error",
        nbins=20,
        marginal="box",
        title="Distribution of Forecast Errors",
        color_discrete_sequence=["#1f77b4"]
    )
    fig_dist.update_layout(
        template="plotly_white",
        xaxis_title="Error ($)",
        yaxis_title="Frequency"
    )

    return fig_time, fig_dist
