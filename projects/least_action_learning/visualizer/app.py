"""Gradio application for routing visualization."""

from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch

from .data import (
    ExperimentRun,
    analyze_all_experiments,
    discover_experiments,
    get_experiment_display_name,
    get_routing_at_step,
    load_experiment,
)
from .plots import (
    create_accuracy_comparison,
    create_empty_figure,
    create_grokking_summary_table,
    create_head_utilization,
    create_multi_experiment_comparison,
    create_routing_entropy_curve,
    create_routing_heatmap,
    create_spectral_plot,
    create_training_curves,
)


# Cache for loaded experiments
_experiment_cache: dict[str, ExperimentRun] = {}


def get_experiment(exp_path: str) -> Optional[ExperimentRun]:
    """Load experiment with caching."""
    if not exp_path:
        return None

    if exp_path not in _experiment_cache:
        try:
            _experiment_cache[exp_path] = load_experiment(Path(exp_path))
        except Exception as e:
            print(f"Failed to load experiment: {e}")
            return None

    return _experiment_cache[exp_path]


def get_experiment_choices() -> list[tuple[str, str]]:
    """Get experiment choices as (display_name, path) pairs."""
    experiments = discover_experiments()
    choices = []
    for exp in experiments:
        display = get_experiment_display_name(exp)
        choices.append((display, str(exp)))
    return choices


def refresh_experiments() -> dict:
    """Refresh the experiment dropdown choices."""
    _experiment_cache.clear()
    choices = get_experiment_choices()
    return gr.update(choices=choices, value=None)


def update_all_plots(exp_path: str) -> Tuple:
    """
    Update all plots when experiment selection changes.

    Returns:
        Tuple of (training_fig, entropy_fig, heatmap_fig, utilization_fig,
                  layer_slider_update, step_slider_update, spectral_fig)
    """
    if not exp_path:
        empty = create_empty_figure("Select an experiment")
        return (
            empty,
            empty,
            empty,
            empty,
            gr.update(maximum=3, value=0),
            gr.update(maximum=100, value=100),
            empty,
        )

    exp = get_experiment(exp_path)
    if exp is None:
        empty = create_empty_figure("Failed to load experiment")
        return (
            empty,
            empty,
            empty,
            empty,
            gr.update(maximum=3, value=0),
            gr.update(maximum=100, value=100),
            empty,
        )

    # Training curves
    training_fig = create_training_curves(exp.history_df)
    entropy_fig = create_routing_entropy_curve(exp.history_df)

    # Routing heatmaps (initial: layer 0, final step)
    if exp.has_routing:
        final_snapshot = get_routing_at_step(exp, 100.0)
        if final_snapshot:
            heatmap_fig = create_routing_heatmap(
                final_snapshot["weights"],
                exp.p,
                layer_idx=0,
                n_heads=exp.n_heads,
            )
            utilization_fig = create_head_utilization(
                final_snapshot["weights"],
                n_heads=exp.n_heads,
            )
        else:
            heatmap_fig = create_empty_figure("No routing snapshots")
            utilization_fig = create_empty_figure("No routing snapshots")
    else:
        heatmap_fig = create_empty_figure("No routing data (baseline model?)")
        utilization_fig = create_empty_figure("No routing data")

    # Spectral analysis (requires model inference)
    spectral_fig = create_spectral_from_experiment(exp)

    # Update slider ranges
    layer_update = gr.update(maximum=exp.n_layers - 1, value=0)
    step_update = gr.update(maximum=100, value=100)

    return (
        training_fig,
        entropy_fig,
        heatmap_fig,
        utilization_fig,
        layer_update,
        step_update,
        spectral_fig,
    )


def update_heatmaps(exp_path: str, layer_idx: int, step_percent: float) -> Tuple:
    """
    Update heatmaps when layer or step slider changes.

    Returns:
        Tuple of (heatmap_fig, utilization_fig)
    """
    if not exp_path:
        empty = create_empty_figure("Select an experiment")
        return empty, empty

    exp = get_experiment(exp_path)
    if exp is None or not exp.has_routing:
        empty = create_empty_figure("No routing data")
        return empty, empty

    snapshot = get_routing_at_step(exp, step_percent)
    if snapshot is None:
        empty = create_empty_figure("No snapshot at this step")
        return empty, empty

    heatmap_fig = create_routing_heatmap(
        snapshot["weights"],
        exp.p,
        layer_idx=int(layer_idx),
        n_heads=exp.n_heads,
    )
    utilization_fig = create_head_utilization(
        snapshot["weights"],
        n_heads=exp.n_heads,
    )

    return heatmap_fig, utilization_fig


def load_all_experiments() -> dict[str, ExperimentRun]:
    """Load all experiments into cache and return the cache."""
    experiments = discover_experiments()
    for exp_path in experiments:
        path_str = str(exp_path)
        if path_str not in _experiment_cache:
            try:
                _experiment_cache[path_str] = load_experiment(exp_path)
            except Exception as e:
                print(f"Failed to load experiment {exp_path}: {e}")
    return _experiment_cache


def update_grokking_analysis() -> Tuple:
    """
    Update the grokking quality analysis table and experiment choices.

    Returns:
        Tuple of (table_fig, experiment_choices)
    """
    experiments = load_all_experiments()

    if not experiments:
        empty = create_empty_figure("No experiments found")
        return empty, gr.update(choices=[])

    # Analyze all experiments (returns DataFrame sorted by clean + variance)
    analysis_df = analyze_all_experiments(experiments)
    table_fig = create_grokking_summary_table(analysis_df)

    # Create name -> path mapping
    name_to_path = {exp.name: path for path, exp in experiments.items()}

    # Create choices in rank order with rank labels
    choices = []
    for rank, row in enumerate(analysis_df.itertuples(), start=1):
        name = row.name
        path = name_to_path.get(name)
        if path:
            label = f"#{rank} {name}"
            choices.append((label, path))

    return table_fig, gr.update(choices=choices)


def update_all_comparisons(selected_paths: list[str]) -> Tuple:
    """
    Update all multi-experiment comparison plots.

    Args:
        selected_paths: List of experiment path strings

    Returns:
        Tuple of 6 Plotly figures (acc, loss, wnorm, jacobian, hessian, repnorm)
    """
    empty = create_empty_figure("Select experiments to compare")

    if not selected_paths:
        return empty, empty, empty, empty, empty, empty

    # Load selected experiments
    experiments = []
    for path in selected_paths:
        exp = get_experiment(path)
        if exp is not None:
            experiments.append(exp)

    if not experiments:
        return empty, empty, empty, empty, empty, empty

    # Create all comparison plots
    acc_fig = create_accuracy_comparison(experiments)  # Train + Test accuracy
    loss_fig = create_multi_experiment_comparison(experiments, "test_loss")
    wnorm_fig = create_multi_experiment_comparison(experiments, "total_weight_norm")
    jacobian_fig = create_multi_experiment_comparison(experiments, "jacobian_norm")
    hessian_fig = create_multi_experiment_comparison(experiments, "hessian_trace")
    repnorm_fig = create_multi_experiment_comparison(experiments, "representation_norm")

    return acc_fig, loss_fig, wnorm_fig, jacobian_fig, hessian_fig, repnorm_fig


def create_spectral_from_experiment(exp: ExperimentRun):
    """
    Create spectral plot by running inference on the model.

    Args:
        exp: Loaded experiment

    Returns:
        Plotly figure
    """
    if exp.model_path is None or not exp.model_path.exists():
        return create_empty_figure("No model checkpoint found")

    try:
        # Import model class
        import sys

        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from src.models import BaselineMLP, RoutedNetwork

        # Load model
        checkpoint = torch.load(exp.model_path, map_location="cpu", weights_only=False)
        model_type = exp.config.get("model_type", "routed")

        p = exp.p
        input_dim = 2 * p
        hidden_dim = exp.config.get("hidden_dim", 128)
        n_layers = exp.n_layers

        if model_type == "baseline":
            model = BaselineMLP(input_dim, hidden_dim, p, n_layers)
        else:
            model = RoutedNetwork(input_dim, hidden_dim, p, n_layers, exp.n_heads)

        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        # Run inference on all inputs
        with torch.no_grad():
            a_vals = torch.arange(p)
            b_vals = torch.arange(p)
            aa, bb = torch.meshgrid(a_vals, b_vals, indexing="ij")
            pairs = torch.stack([aa.flatten(), bb.flatten()], dim=1)

            batch_size = p * p
            inputs = torch.zeros(batch_size, 2 * p)
            inputs[torch.arange(batch_size), pairs[:, 0]] = 1.0
            inputs[torch.arange(batch_size), p + pairs[:, 1]] = 1.0

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            predicted = logits.argmax(dim=-1).view(p, p).numpy()

        return create_spectral_plot(predicted, p)

    except Exception as e:
        return create_empty_figure(f"Spectral analysis failed: {e}")


def create_app() -> gr.Blocks:
    """Create the Gradio application interface."""

    with gr.Blocks(title="Routing Visualizer") as app:
        gr.Markdown("# Least Action Learning: Routing Visualizer")
        gr.Markdown(
            "Visualize routing patterns and training dynamics from saved experiments."
        )

        # ─── Experiment Selection ───
        with gr.Row():
            with gr.Column(scale=5):
                exp_dropdown = gr.Dropdown(
                    choices=get_experiment_choices(),
                    label="Select Experiment",
                    info="Choose a completed training run to analyze",
                )
            with gr.Column(scale=1):
                refresh_btn = gr.Button("Refresh", variant="secondary", size="sm")

        # ─── Training Progress ───
        gr.Markdown("## Training Progress")
        with gr.Row():
            with gr.Column(scale=1):
                training_plot = gr.Plot(label="Loss & Accuracy")
            with gr.Column(scale=1):
                entropy_plot = gr.Plot(label="Routing Entropy")

        # ─── Routing Patterns ───
        gr.Markdown("## Routing Patterns")
        gr.Markdown(
            "*Use the sliders to explore routing at different layers and training stages.*"
        )

        with gr.Row():
            layer_slider = gr.Slider(
                minimum=0,
                maximum=3,
                step=1,
                value=0,
                label="Layer",
                info="Select which layer to visualize",
            )
            step_slider = gr.Slider(
                minimum=0,
                maximum=100,
                step=5,
                value=100,
                label="Training Progress (%)",
                info="Scrub through training to see routing evolution",
            )

        with gr.Row():
            heatmap_plot = gr.Plot(label="Dominant Head per Input (a, b)")
            utilization_plot = gr.Plot(label="Head Utilization by Layer")

        # ─── Spectral Analysis ───
        with gr.Accordion("Spectral Analysis", open=False):
            gr.Markdown(
                "Spectral analysis shows the frequency content of the learned function. "
                "Smooth functions have energy concentrated in low frequencies."
            )
            spectral_plot = gr.Plot(label="Output Function & Power Spectrum")

        # ─── Grokking Quality Analysis ───
        gr.Markdown("## Grokking Quality Analysis")
        gr.Markdown(
            "*Compare all experiments by grokking quality. "
            "Shows % of training steps where train≥98% and test≥95%. "
            "Sorted by test%, then train%.*"
        )

        grokking_table = gr.Plot(label="Grokking Quality Summary")

        gr.Markdown("### Compare Experiments")
        experiment_checkboxes = gr.CheckboxGroup(
            choices=[],
            label="Select experiments to compare",
            info="Select experiments to overlay their metrics (ordered by grokking quality)",
        )

        # Accuracy comparison
        comparison_acc_plot = gr.Plot(label="Test Accuracy Comparison")

        # Loss comparison
        comparison_loss_plot = gr.Plot(label="Loss Comparison (log scale)")

        # Additional metrics in a row
        with gr.Row():
            comparison_wnorm_plot = gr.Plot(label="Weight Norm")
            comparison_jacobian_plot = gr.Plot(label="Jacobian Norm")

        with gr.Row():
            comparison_hessian_plot = gr.Plot(label="Hessian Trace")
            comparison_repnorm_plot = gr.Plot(label="Representation Norm")

        # ─── Event Handlers ───
        exp_dropdown.change(
            fn=update_all_plots,
            inputs=[exp_dropdown],
            outputs=[
                training_plot,
                entropy_plot,
                heatmap_plot,
                utilization_plot,
                layer_slider,
                step_slider,
                spectral_plot,
            ],
        )

        def refresh_all():
            """Refresh experiments and update grokking analysis."""
            dropdown_update = refresh_experiments()
            table_fig, checkboxes_update = update_grokking_analysis()
            return dropdown_update, table_fig, checkboxes_update

        refresh_btn.click(
            fn=refresh_all,
            outputs=[exp_dropdown, grokking_table, experiment_checkboxes],
        )

        # Update heatmaps when sliders change
        layer_slider.change(
            fn=update_heatmaps,
            inputs=[exp_dropdown, layer_slider, step_slider],
            outputs=[heatmap_plot, utilization_plot],
        )

        step_slider.change(
            fn=update_heatmaps,
            inputs=[exp_dropdown, layer_slider, step_slider],
            outputs=[heatmap_plot, utilization_plot],
        )

        # ─── Grokking Analysis Event Handlers ───
        experiment_checkboxes.change(
            fn=update_all_comparisons,
            inputs=[experiment_checkboxes],
            outputs=[
                comparison_acc_plot,
                comparison_loss_plot,
                comparison_wnorm_plot,
                comparison_jacobian_plot,
                comparison_hessian_plot,
                comparison_repnorm_plot,
            ],
        )

        # Initialize grokking analysis on app load
        app.load(
            fn=update_grokking_analysis,
            outputs=[grokking_table, experiment_checkboxes],
        )

    return app


def launch(share: bool = False, port: int = 7860):
    """
    Launch the Gradio application.

    Args:
        share: Whether to create a public shareable link
        port: Port to run on (default: 7860)
    """
    app = create_app()
    app.launch(share=share, server_port=port, theme=gr.themes.Soft())
