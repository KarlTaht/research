"""Gradio application for routing visualization."""

from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch

from .data import (
    ExperimentRun,
    discover_experiments,
    get_experiment_display_name,
    get_routing_at_step,
    load_experiment,
)
from .plots import (
    create_empty_figure,
    create_head_utilization,
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

        refresh_btn.click(
            fn=refresh_experiments,
            outputs=[exp_dropdown],
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
