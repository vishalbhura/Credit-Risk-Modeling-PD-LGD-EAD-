"""
src/utils.py
============
Shared utilities used across all notebooks and the dashboard.

Provides:
- Logging setup      : Consistent log format for all modules.
- Plot style         : Uniform matplotlib/seaborn theme for all charts.
- Figure saving      : Persist plots to the reports/ directory.
- Model persistence  : Save and load pickle artifacts.

Import pattern (at the top of every notebook):
    from src.utils import setup_logging, set_plot_style
    setup_logging()
    set_plot_style()
"""

import logging
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a consistent format.

    Call this once at the top of each notebook or script before any
    other imports that use logging.

    Args:
        level: Logging level constant (default: logging.INFO).
               Use logging.DEBUG for more verbose output.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,   # Override any previously set handlers (important in notebooks)
    )


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

# Consistent colour palette used throughout the project
PLOT_COLOR = "#2c7bb6"        # Primary blue — single-series charts
ACCENT_COLOR = "#d7191c"      # Red — highlights / bad loans
PALETTE = ["#2c7bb6", "#d7191c", "#1a9641", "#fdae61", "#abd9e9"]
FIGURE_DPI = 120


def set_plot_style() -> None:
    """Apply a clean, consistent style to all matplotlib / seaborn figures.

    Call this once per notebook after setup_logging().
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": FIGURE_DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "figure.titlesize": 14,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })


def save_figure(fig: plt.Figure, filename: str, reports_dir: str = "reports") -> None:
    """Save a matplotlib figure to the reports/ directory.

    Args:
        fig:         Matplotlib Figure object to save.
        filename:    Output filename including extension (e.g. 'roc_curve.png').
        reports_dir: Directory to save into (default: 'reports/').
    """
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(reports_dir, filename)
    fig.savefig(out_path, bbox_inches="tight", dpi=FIGURE_DPI)
    logging.getLogger(__name__).info("Figure saved → %s", out_path)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(obj: object, filepath: str) -> None:
    """Serialize any Python object (model, preprocessor, dict) to pickle.

    Creates the parent directory automatically if it does not exist.

    Args:
        obj:      Object to serialize (sklearn estimator, dict, etc.).
        filepath: Destination path (e.g. 'models/pd/lr_model.pkl').
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    logging.getLogger(__name__).info("Model saved → %s", filepath)


def load_model(filepath: str) -> object:
    """Deserialize a Python object from a pickle file.

    Args:
        filepath: Path to the .pkl file.

    Returns:
        The deserialized object.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    logging.getLogger(__name__).info("Model loaded ← %s", filepath)
    return obj
