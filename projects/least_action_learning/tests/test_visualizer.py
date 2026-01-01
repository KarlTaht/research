"""Tests for the visualizer module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from projects.least_action_learning.visualizer.app import (
    filter_by_epoch_range,
    create_single_metric_plot,
    create_per_layer_metric_plot,
    update_weight_plots,
    update_gradient_plots,
    update_training_dynamics_plots,
    update_all_plots,
    update_experiment_controls,
)
from projects.least_action_learning.visualizer.plots import create_empty_figure


class TestFilterByEpochRange:
    """Tests for epoch range filtering."""

    def test_filter_with_min_max_epochs(self):
        """Test filtering with separate min and max epochs."""
        df = pd.DataFrame({
            'step': [0, 100, 200, 300, 400, 500],
            'value': [1, 2, 3, 4, 5, 6],
        })

        filtered = filter_by_epoch_range(df, 100, 300)

        assert len(filtered) == 3
        assert list(filtered['step']) == [100, 200, 300]

    def test_filter_full_range(self):
        """Test filtering with full range includes all data."""
        df = pd.DataFrame({
            'step': [0, 100, 200, 300, 400, 500],
            'value': [1, 2, 3, 4, 5, 6],
        })

        filtered = filter_by_epoch_range(df, 0, 500)

        assert len(filtered) == 6

    def test_filter_empty_dataframe(self):
        """Test filtering empty DataFrame."""
        df = pd.DataFrame({'step': [], 'value': []})

        filtered = filter_by_epoch_range(df, 0, 100)

        assert len(filtered) == 0

    def test_filter_missing_step_column(self):
        """Test filtering DataFrame without step column."""
        df = pd.DataFrame({'epoch': [0, 100, 200], 'value': [1, 2, 3]})

        # Should return original DataFrame unchanged
        filtered = filter_by_epoch_range(df, 0, 100)

        assert len(filtered) == 3


class TestCreateSingleMetricPlot:
    """Tests for single metric plotting."""

    def test_plot_with_valid_data(self):
        """Test creating plot with valid data."""
        df = pd.DataFrame({
            'step': [0, 100, 200, 300],
            'total_weight_norm': [1.0, 1.5, 2.0, 2.5],
        })

        fig = create_single_metric_plot(df, 'total_weight_norm', 'Weight Norm')

        assert fig is not None
        assert len(fig.data) == 1

    def test_plot_with_missing_metric(self):
        """Test creating plot with missing metric column."""
        df = pd.DataFrame({
            'step': [0, 100, 200],
            'other_metric': [1.0, 2.0, 3.0],
        })

        fig = create_single_metric_plot(df, 'total_weight_norm', 'Weight Norm')

        # Should return empty figure
        assert fig is not None
        assert len(fig.data) == 0  # Empty figure has no data traces

    def test_plot_with_epoch_range_filter(self):
        """Test plot respects epoch range filter."""
        df = pd.DataFrame({
            'step': [0, 100, 200, 300, 400],
            'jacobian_norm': [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        fig = create_single_metric_plot(
            df, 'jacobian_norm', 'Jacobian', min_epoch=100, max_epoch=300
        )

        assert fig is not None
        # The filtered data should only have 3 points
        if len(fig.data) > 0:
            assert len(fig.data[0].x) == 3

    def test_plot_with_nan_values(self):
        """Test plot handles NaN values correctly."""
        df = pd.DataFrame({
            'step': [0, 100, 200, 300],
            'metric': [1.0, np.nan, 3.0, np.nan],
        })

        fig = create_single_metric_plot(df, 'metric', 'Metric')

        assert fig is not None
        # Should only plot non-NaN values
        if len(fig.data) > 0:
            assert len(fig.data[0].x) == 2


class TestUpdateFunctions:
    """Tests for update functions with various input types."""

    def test_update_weight_plots_with_none_path(self):
        """Test update_weight_plots handles None path."""
        result = update_weight_plots(None, 0, 10000, 'All')

        # 6 weight plots: embed, b0_attn, b0_ffn, b1_attn, b1_ffn, output
        assert len(result) == 6

    def test_update_gradient_plots_with_none_path(self):
        """Test update_gradient_plots handles None path."""
        result = update_gradient_plots(None, 0, 10000)

        assert len(result) == 2

    def test_update_training_dynamics_plots_with_none_path(self):
        """Test update_training_dynamics_plots handles None path."""
        result = update_training_dynamics_plots(None, 0, 10000)

        # 3 plots: loss curves, accuracy curves, learning rate
        assert len(result) == 3

    def test_update_all_plots_with_none_path(self):
        """Test update_all_plots handles None path."""
        result = update_all_plots(None, 0, 10000, 'All')

        # 3 dynamics + 6 weight + 5 curvature = 14 total
        assert len(result) == 14

    def test_update_experiment_controls_with_none_path(self):
        """Test update_experiment_controls handles None path."""
        result = update_experiment_controls(None)

        assert len(result) == 3  # min_epoch, max_epoch, layer dropdown
        # Should return default slider and layer dropdown updates


class TestEpochRangeInputTypes:
    """Tests for different epoch range input types."""

    def test_epoch_range_with_integers(self):
        """Test epoch range works with integer values."""
        df = pd.DataFrame({
            'step': [0, 100, 200, 300],
            'metric': [1, 2, 3, 4],
        })

        filtered = filter_by_epoch_range(df, 100, 200)
        assert len(filtered) == 2

    def test_epoch_range_with_floats(self):
        """Test epoch range works when Gradio passes float values."""
        df = pd.DataFrame({
            'step': [0, 100, 200, 300],
            'metric': [1, 2, 3, 4],
        })

        # Gradio sliders might pass float values
        filtered = filter_by_epoch_range(df, 100.0, 200.0)
        assert len(filtered) == 2

    def test_epoch_range_min_equals_max(self):
        """Test epoch range when min equals max (single point)."""
        df = pd.DataFrame({
            'step': [0, 100, 200, 300],
            'metric': [1, 2, 3, 4],
        })

        filtered = filter_by_epoch_range(df, 100, 100)
        assert len(filtered) == 1
        assert filtered['step'].iloc[0] == 100
