"""
Experiment analysis package for grokking research.

Provides a notebook-friendly API for exploring and analyzing training experiments.

Example usage:

    from src.analysis import ExperimentStore, ExperimentLoader, detect_phases

    # Query experiments with SQL
    store = ExperimentStore()
    best = store.query('''
        SELECT experiment_name, MIN(test_loss) as best_loss
        FROM experiments WHERE step > 10000
        GROUP BY experiment_name ORDER BY best_loss
    ''')

    # Load single experiment
    loader = ExperimentLoader()
    exp = loader.load('p17_lr3e-4_wd1.0')
    print(exp.history_df.columns)

    # Detect grokking phases
    phases = detect_phases(exp.history_df)
    print(f"Grokked at step {phases.grokking_start}")

    # Compute derived metrics
    enriched = compute_derived_metrics(exp.history_df)
    print(enriched[['step', 'generalization_gap', 'loss_ratio']].tail())
"""

from .loader import ExperimentData, ExperimentLoader
from .architecture import (
    get_layer_names,
    get_layer_groups,
    get_layer_display_name,
    infer_model_type,
)
from .store import ExperimentStore
from .phases import (
    GrokkingPhases,
    GrokkingAnalysis,
    detect_phases,
    analyze_grokking,
    analyze_all_experiments,
)
from .metrics import (
    compute_aggregates_over_history,
    compute_derived_metrics,
)
from .comparison import (
    align_by_epoch,
    compute_metric_deltas,
    group_by_hyperparameter,
    create_sweep_summary,
)

__all__ = [
    # loader
    "ExperimentData",
    "ExperimentLoader",
    # architecture
    "get_layer_names",
    "get_layer_groups",
    "get_layer_display_name",
    "infer_model_type",
    # store
    "ExperimentStore",
    # phases
    "GrokkingPhases",
    "GrokkingAnalysis",
    "detect_phases",
    "analyze_grokking",
    "analyze_all_experiments",
    # metrics
    "compute_aggregates_over_history",
    "compute_derived_metrics",
    # comparison
    "align_by_epoch",
    "compute_metric_deltas",
    "group_by_hyperparameter",
    "create_sweep_summary",
]
