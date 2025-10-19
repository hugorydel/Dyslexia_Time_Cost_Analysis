"""
Visualization utilities for dyslexia analysis
Contains plotting functions for exploratory and summary visualizations
"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_exploratory_plots(data: pd.DataFrame, output_path: Path) -> None:
    """
    Create exploratory plots for word-level data

    Args:
        data: DataFrame with word-level eye-tracking data
        output_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Word-Level Exploratory Data Analysis")

        # Total reading time distribution (only for fixated words)
        if "total_reading_time" in data.columns:
            fixated_data = data[data["n_fixations"] > 0]["total_reading_time"]
            axes[0, 0].hist(fixated_data, bins=50, alpha=0.7)
            axes[0, 0].set_title("Total Reading Time Distribution (Fixated Words)")
            axes[0, 0].set_xlabel("Duration (ms)")

        # Group comparison with normalized histograms (only for fixated words)
        if "dyslexic" in data.columns and "total_reading_time" in data.columns:
            fixated_mask = data["n_fixations"] > 0
            for group, label in [(True, "Dyslexic"), (False, "Control")]:
                group_data = data[(data["dyslexic"] == group) & fixated_mask][
                    "total_reading_time"
                ]
                # Create normalized histogram
                if len(group_data) > 0:
                    counts, bins, patches = axes[0, 1].hist(
                        group_data, alpha=0.7, label=label, bins=30, density=True
                    )

            axes[0, 1].set_title("Reading Time by Group (Fixated Words Only)")
            axes[0, 1].set_xlabel("Duration (ms)")
            axes[0, 1].set_ylabel("Density (normalized)")
            axes[0, 1].legend()

        # Word length distribution
        if "word_length" in data.columns:
            axes[1, 0].hist(data["word_length"], bins=range(1, 16), alpha=0.7)
            axes[1, 0].set_title("Word Length Distribution")
            axes[1, 0].set_xlabel("Characters")

        # Skipping vs Fixation patterns
        if "n_fixations" in data.columns:
            skipped_count = (data["n_fixations"] == 0).sum()
            fixated_count = (data["n_fixations"] > 0).sum()

            labels = ["Skipped", "Fixated"]
            counts = [skipped_count, fixated_count]
            colors = ["lightcoral", "skyblue"]

            axes[1, 1].pie(counts, labels=labels, colors=colors, autopct="%1.1f%%")
            axes[1, 1].set_title("Word Skipping vs Fixation")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Exploratory plots saved to {output_path}")

    except Exception as e:
        logger.warning(f"Could not create exploratory plots: {e}")


def create_group_summary_plots(
    data: pd.DataFrame, output_path: Path, skipping_analysis: Dict[str, Any] = None
) -> None:
    """
    Create comprehensive group summary plots

    Args:
        data: DataFrame with word-level eye-tracking data
        output_path: Path to save the plot
        skipping_analysis: Optional skipping analysis results
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style for better-looking plots
        plt.style.use("default")
        sns.set_palette("Set2")

        # Create figure with subplots for different measures
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(
            "Group Summary Statistics: Dyslexic vs Control Readers",
            fontsize=16,
            fontweight="bold",
        )

        if "dyslexic" not in data.columns:
            logger.warning(
                "No dyslexic column found - cannot create group comparison plots"
            )
            return

        # Calculate subject-level averages for cleaner group comparisons
        # Only use fixated words for reading time measures
        fixated_data = data[data["n_fixations"] > 0]

        subject_averages = (
            fixated_data.groupby(["subject_id", "dyslexic"])
            .agg(
                {
                    measure: "mean"
                    for measure in [
                        "total_reading_time",
                        "first_fixation_duration",
                        "gaze_duration",
                        "n_fixations",
                        "regression_probability",
                    ]
                    if measure in fixated_data.columns
                }
            )
            .reset_index()
        )

        plot_idx = 0

        # 1. Total Reading Time by Group
        if "total_reading_time" in subject_averages.columns:
            ax = axes[plot_idx // 2, plot_idx % 2]
            _create_bar_plot_with_error_bars(
                ax,
                subject_averages,
                "total_reading_time",
                "Total Reading Time by Group",
                "Reading Time (ms)",
            )
            plot_idx += 1

        # 2. First Fixation Duration by Group
        if "first_fixation_duration" in subject_averages.columns:
            ax = axes[plot_idx // 2, plot_idx % 2]
            _create_bar_plot_with_error_bars(
                ax,
                subject_averages,
                "first_fixation_duration",
                "First Fixation Duration by Group",
                "Duration (ms)",
            )
            plot_idx += 1

        # 3. Gaze Duration by Group
        if "gaze_duration" in subject_averages.columns:
            ax = axes[plot_idx // 2, plot_idx % 2]
            _create_bar_plot_with_error_bars(
                ax,
                subject_averages,
                "gaze_duration",
                "Gaze Duration by Group",
                "Duration (ms)",
            )
            plot_idx += 1

        # 4. Number of Fixations by Group
        if "n_fixations" in subject_averages.columns:
            ax = axes[plot_idx // 2, plot_idx % 2]
            _create_bar_plot_with_error_bars(
                ax,
                subject_averages,
                "n_fixations",
                "Number of Fixations per Word by Group",
                "Number of Fixations",
            )
            plot_idx += 1

        # 5. Skipping Probability by Group
        if skipping_analysis and "subject_level_skipping_by_group" in skipping_analysis:
            ax = axes[plot_idx // 2, plot_idx % 2]
            _create_skipping_plot(ax, skipping_analysis)
            plot_idx += 1

        # 6. Word Length Effects by Group (if remaining space)
        if (
            plot_idx < 6
            and "word_length" in data.columns
            and "total_reading_time" in data.columns
        ):
            ax = axes[plot_idx // 2, plot_idx % 2]
            _create_word_length_effects_plot(ax, data)
            plot_idx += 1

        # Hide any unused subplots
        for i in range(plot_idx, 6):
            axes[i // 2, i % 2].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Group summary plots saved to {output_path}")

    except Exception as e:
        logger.warning(f"Could not create group summary plots: {e}")


def _create_bar_plot_with_error_bars(ax, data, measure, title, ylabel):
    """Helper function to create bar plots with error bars"""
    groups = ["Control", "Dyslexic"]
    means = [
        data[data["dyslexic"] == False][measure].mean(),
        data[data["dyslexic"] == True][measure].mean(),
    ]
    stds = [
        data[data["dyslexic"] == False][measure].std(),
        data[data["dyslexic"] == True][measure].std(),
    ]

    # Color mapping for different measures
    color_map = {
        "total_reading_time": ["skyblue", "lightcoral"],
        "first_fixation_duration": ["lightgreen", "orange"],
        "gaze_duration": ["mediumpurple", "gold"],
        "n_fixations": ["lightsteelblue", "lightsalmon"],
    }
    colors = color_map.get(measure, ["lightblue", "lightpink"])

    bars = ax.bar(
        groups,
        means,
        yerr=stds,
        capsize=5,
        alpha=0.7,
        color=colors,
    )
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        offset = std + (height * 0.05)  # Adaptive offset based on bar height

        if measure == "n_fixations":
            label = f"{mean:.2f}±{std:.2f}"
        else:
            label = f"{mean:.1f}±{std:.1f}"

        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            label,
            ha="center",
            va="bottom",
        )


def _create_skipping_plot(ax, skipping_analysis):
    """Helper function to create skipping probability plot"""
    groups = ["Control", "Dyslexic"]
    skipping_data = skipping_analysis["subject_level_skipping_by_group"]

    means = [
        skipping_data.get("control", {}).get("mean", 0),
        skipping_data.get("dyslexic", {}).get("mean", 0),
    ]
    stds = [
        skipping_data.get("control", {}).get("std", 0),
        skipping_data.get("dyslexic", {}).get("std", 0),
    ]

    bars = ax.bar(
        groups,
        means,
        yerr=stds,
        capsize=5,
        alpha=0.7,
        color=["darkseagreen", "plum"],
    )
    ax.set_title("Word Skipping Probability by Group", fontweight="bold")
    ax.set_ylabel("Skipping Probability")
    ax.set_ylim(0, max(means) * 1.3 if max(means) > 0 else 0.5)

    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 0.01,
            f"{mean:.3f}±{std:.3f}",
            ha="center",
            va="bottom",
        )


def _create_word_length_effects_plot(ax, data):
    """Helper function to create word length effects plot"""
    # Use only fixated words for reading time analysis
    fixated_data = data[data["n_fixations"] > 0].copy()

    # Create word length bins
    fixated_data["word_length_bin"] = pd.cut(
        fixated_data["word_length"],
        bins=[0, 3, 5, 7, float("inf")],
        labels=[
            "Short (1-3)",
            "Medium (4-5)",
            "Long (6-7)",
            "Very Long (8+)",
        ],
    )

    length_group_means = (
        fixated_data.groupby(["word_length_bin", "dyslexic"], observed=True)[
            "total_reading_time"
        ]
        .mean()
        .unstack()
    )
    length_group_stds = (
        fixated_data.groupby(["word_length_bin", "dyslexic"], observed=True)[
            "total_reading_time"
        ]
        .std()
        .unstack()
    )

    x = np.arange(len(length_group_means.index))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        length_group_means[False],
        width,
        yerr=length_group_stds[False],
        label="Control",
        alpha=0.7,
        capsize=3,
    )
    bars2 = ax.bar(
        x + width / 2,
        length_group_means[True],
        width,
        yerr=length_group_stds[True],
        label="Dyslexic",
        alpha=0.7,
        capsize=3,
    )

    ax.set_title("Reading Time by Word Length and Group", fontweight="bold")
    ax.set_ylabel("Reading Time (ms)")
    ax.set_xlabel("Word Length")
    ax.set_xticks(x)
    ax.set_xticklabels(length_group_means.index)
    ax.legend()
