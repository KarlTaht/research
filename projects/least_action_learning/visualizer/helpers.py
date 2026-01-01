"""Data cleaning and plotting helper utilities for visualization."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .styles import CATEGORICAL_COLORS


def get_valid_data(df: pd.DataFrame, col: str) -> Tuple[pd.Series, np.ndarray]:
    """Get column data with a mask of valid (non-NaN, finite) values.

    Args:
        df: DataFrame containing the data
        col: Column name to extract

    Returns:
        Tuple of (series, valid_mask boolean array)
        If column doesn't exist, returns empty series and empty mask
    """
    if col not in df.columns:
        return pd.Series(dtype=float), np.array([], dtype=bool)
    y_data = df[col]
    valid_mask = (y_data.notna() & np.isfinite(y_data)).values
    return y_data, valid_mask


def filter_by_epoch_range(
    df: pd.DataFrame,
    min_epoch: int,
    max_epoch: int,
) -> pd.DataFrame:
    """Filter DataFrame by epoch/step range.

    Args:
        df: DataFrame with 'step' column
        min_epoch: Minimum step to include
        max_epoch: Maximum step to include

    Returns:
        Filtered DataFrame
    """
    if df.empty or "step" not in df.columns:
        return df
    return df[(df["step"] >= min_epoch) & (df["step"] <= max_epoch)]


def add_line_trace(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    name: str,
    color: str,
    row: Optional[int] = None,
    col: Optional[int] = None,
    dash: str = "solid",
    width: int = 2,
    fill: Optional[str] = None,
    fillcolor: Optional[str] = None,
    hover_format: str = ".4g",
) -> bool:
    """Add a scatter line trace with standard formatting.

    Args:
        fig: Plotly figure to add trace to
        df: DataFrame containing the data
        x_col: Column name for x-axis data
        y_col: Column name for y-axis data
        name: Legend name for the trace
        color: Line color
        row: Subplot row (for make_subplots)
        col: Subplot column (for make_subplots)
        dash: Line dash style ("solid", "dash", "dot", "dashdot")
        width: Line width in pixels
        fill: Fill mode ("tozeroy", "tonexty", etc.)
        fillcolor: Fill color (with alpha)
        hover_format: Number format for hover template

    Returns:
        True if trace was added (had valid data), False otherwise
    """
    y_data, valid_mask = get_valid_data(df, y_col)
    if not valid_mask.any():
        return False

    trace = go.Scatter(
        x=df.loc[valid_mask, x_col],
        y=df.loc[valid_mask, y_col],
        mode="lines",
        name=name,
        line=dict(color=color, width=width, dash=dash),
        fill=fill,
        fillcolor=fillcolor,
        hovertemplate=f"{name}<br>Step: %{{x}}<br>Value: %{{y:{hover_format}}}<extra></extra>",
    )

    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)
    return True


def create_empty_figure(message: str = "No data available") -> go.Figure:
    """Create an empty figure with a centered message.

    Args:
        message: Text to display in the center

    Returns:
        Plotly figure with annotation
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        height=300,
    )
    return fig


def get_color(index: int) -> str:
    """Get a color from the categorical palette by index.

    Args:
        index: Color index (will wrap around if > palette size)

    Returns:
        Hex color string
    """
    return CATEGORICAL_COLORS[index % len(CATEGORICAL_COLORS)]
