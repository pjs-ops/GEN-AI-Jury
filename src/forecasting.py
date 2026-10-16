"""
forecasting.py
Multi-step future forecasting engine with recursive feature state progression
and empirical residual-calibrated prediction intervals.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.forecasting_models import BaseForecaster, get_all_models
from src.feature_engineering import update_recursive_features


# Standard US Retail Holidays in Walmart historical calendar
KNOWN_HOLIDAY_WEEKS = {
    # Super Bowl: Week ~6
    6: "Super Bowl",
    # Labor Day: Week ~36
    36: "Labor Day",
    # Thanksgiving: Week ~47
    47: "Thanksgiving",
    # Christmas: Week ~51 / 52
    51: "Christmas",
    52: "Christmas / New Year"
}


def is_walmart_holiday(dt: pd.Timestamp) -> Tuple[int, str]:
    """Determines if the given Friday is a major Walmart holiday week."""
    iso_week = int(dt.isocalendar().week)
    if iso_week in KNOWN_HOLIDAY_WEEKS:
        return 1, KNOWN_HOLIDAY_WEEKS[iso_week]
    return 0, "Regular Week"


def generate_future_forecast(
    model: BaseForecaster,
    history_df: pd.DataFrame,
    horizon: int = 8,
    residual_std: Optional[float] = None,
    z_score: float = 1.96
) -> pd.DataFrame:
    """
    Generates multi-step future sales forecasts beyond the end of historical data.
    
    Parameters:
    - model: BaseForecaster instance (already fitted or will be refitted)
    - history_df: Historical DataFrame with 'Date' and 'Weekly_Sales'
    - horizon: Number of weeks ahead to forecast (e.g. 1, 4, 8, 12)
    - residual_std: Empirical standard deviation of forecasting errors (for prediction interval)
    - z_score: Multiplier for prediction interval (1.96 for 95% interval)
    
    Returns:
    - DataFrame with forecast dates, point forecast, lower & upper bounds.
    """
    history_clean = history_df.sort_values("Date").reset_index(drop=True)
    last_date = pd.to_datetime(history_clean["Date"].iloc[-1])
    sales_history = history_clean["Weekly_Sales"].values.tolist()

    # If residual_std not provided, estimate from standard deviation of diff or residuals
    if residual_std is None or residual_std <= 0:
        residual_std = float(np.std(np.diff(sales_history))) if len(sales_history) > 2 else 1000.0

    forecast_records = []
    
    # Generate future dates
    future_dates = [last_date + pd.Timedelta(weeks=step) for step in range(1, horizon + 1)]
    
    # Statistical / time-series models can forecast directly
    if model.model_type == "statistical":
        raw_preds = model.predict(steps=horizon)
        for h, (dt, pred) in enumerate(zip(future_dates, raw_preds), start=1):
            is_hol, hol_name = is_walmart_holiday(dt)
            # Uncertainty expands as sqrt(h) over the horizon
            interval_radius = z_score * residual_std * np.sqrt(h)
            
            p_val = max(0.0, float(pred))
            lower_b = max(0.0, p_val - interval_radius)
            upper_b = p_val + interval_radius
            
            forecast_records.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "Week": int(dt.isocalendar().week),
                "Holiday": hol_name,
                "IsHoliday": is_hol,
                "Forecast ($)": round(p_val, 2),
                "Lower_Bound ($)": round(lower_b, 2),
                "Upper_Bound ($)": round(upper_b, 2),
                "Model": model.name
            })
    else:
        # Recursive multi-step forecasting for feature-based models
        current_history = list(sales_history)
        
        for h, dt in enumerate(future_dates, start=1):
            is_hol, hol_name = is_walmart_holiday(dt)
            feat_dict = update_recursive_features(
                historical_sales=current_history,
                next_date=dt,
                is_holiday=is_hol
            )
            feat_row = pd.DataFrame([feat_dict])
            
            pred = float(model.predict(steps=1, X=feat_row)[0])
            pred = max(0.0, pred)
            
            interval_radius = z_score * residual_std * np.sqrt(h)
            lower_b = max(0.0, pred - interval_radius)
            upper_b = pred + interval_radius
            
            forecast_records.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "Week": int(dt.isocalendar().week),
                "Holiday": hol_name,
                "IsHoliday": is_hol,
                "Forecast ($)": round(pred, 2),
                "Lower_Bound ($)": round(lower_b, 2),
                "Upper_Bound ($)": round(upper_b, 2),
                "Model": model.name
            })
            
            # Append predicted value into current history for next step's lags
            current_history.append(pred)

    return pd.DataFrame(forecast_records)
