"""
Diagnostic tool for investigating zipf-length interaction effects
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_length_bins(data: pd.DataFrame, n_bins: int = 10) -> tuple:
    """
    Create length bins with approximately equal number of observations
    and generate appropriate labels based on the actual length ranges

    Args:
        data: DataFrame with 'length' column
        n_bins: Number of bins to create (default: 10)

    Returns:
        tuple: (bin_edges, bin_labels)
    """
    # Calculate quantiles to get approximately equal-sized bins
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = data["length"].quantile(quantiles).values

    # Ensure edges are unique and properly ordered
    bin_edges = np.unique(bin_edges)

    # Generate labels based on actual integer length ranges in each bin
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        start = int(np.ceil(bin_edges[i]))
        end = int(np.floor(bin_edges[i + 1]))

        # Adjust for the first bin to include the minimum
        if i == 0:
            start = int(np.floor(bin_edges[i]))

        # Adjust for the last bin to include the maximum
        if i == len(bin_edges) - 2:
            end = int(np.ceil(bin_edges[i + 1]))
            # For last bin, use "X+" format if it's open-ended
            if end > start:
                bin_labels.append(f"{start}-{end}")
            else:
                bin_labels.append(f"{start}+")
        elif start == end:
            bin_labels.append(f"{start}")
        else:
            bin_labels.append(f"{start}-{end}")

    return bin_edges, bin_labels


def diagnose_zipf_length_interaction(
    data: pd.DataFrame,
    n_bins: int = 10,
    output_dir: Path = None,
):
    """
    Diagnostic for zipf effects within length bins - observed data only

    Creates:
    1. Contrast figure comparing dyslexic vs control zipf effects across length bins
    2. JSON export of all diagnostic data

    Args:
        data: DataFrame with columns: length, zipf, ERT, group
        n_bins: Number of bins to create (default: 10)
        output_dir: Directory to save outputs
    """

    if output_dir is None:
        output_dir = Path("./diagnostics")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create bins with approximately equal observations and auto-generate labels
    bin_edges, bin_labels = create_length_bins(data, n_bins)

    # Assign bins
    data = data.copy()
    data["length_bin"] = pd.cut(
        data["length"], bins=bin_edges, labels=bin_labels, include_lowest=True
    )

    diagnostic_results = {
        "bin_edges": bin_edges.tolist(),
        "bin_labels": bin_labels,
        "bins": {},
    }

    # Store contrast data for plotting
    contrast_data = {
        "bin_labels": [],
        "control_effects": [],
        "dyslexic_effects": [],
        "control_se": [],
        "dyslexic_se": [],
    }

    # ========== ANALYSIS PER BIN ==========
    for bin_idx, bin_label in enumerate(bin_labels):
        bin_start = bin_edges[bin_idx]
        bin_end = bin_edges[bin_idx + 1]

        bin_data = data[data["length_bin"] == bin_label]

        if len(bin_data) < 10:
            logger.info(f"  Skipping (only {len(bin_data)} observations)")
            continue

        bin_results = {
            "label": bin_label,
            "length_range": [float(bin_start), float(bin_end)],
            "n_total": len(bin_data),
            "groups": {},
        }

        bin_control_effect = None
        bin_dyslexic_effect = None
        bin_control_se = None
        bin_dyslexic_se = None

        for group in ["control", "dyslexic"]:
            group_data = bin_data[bin_data["group"] == group]

            if len(group_data) < 5:
                continue

            # Zipf statistics
            zipf_q1 = group_data["zipf"].quantile(0.25)
            zipf_q3 = group_data["zipf"].quantile(0.75)
            zipf_iqr = zipf_q3 - zipf_q1

            # Split into 5 zipf quintiles for observed effect
            group_data_sorted = group_data.copy()
            group_data_sorted["zipf_quintile"] = pd.qcut(
                group_data_sorted["zipf"], q=5, labels=False, duplicates="drop"
            )

            observed_ert_by_zipf = []
            zipf_quintile_means = []
            zipf_quintile_se = []

            for q in range(5):
                quintile_data = group_data_sorted[
                    group_data_sorted["zipf_quintile"] == q
                ]
                if len(quintile_data) > 0:
                    observed_ert_by_zipf.append(quintile_data["ERT"].mean())
                    zipf_quintile_means.append(quintile_data["zipf"].mean())
                    zipf_quintile_se.append(quintile_data["ERT"].sem())

            # Calculate observed effect (Q5 - Q1)
            if len(observed_ert_by_zipf) >= 2:
                observed_effect = observed_ert_by_zipf[-1] - observed_ert_by_zipf[0]
                # Standard error of difference
                observed_se = np.sqrt(
                    zipf_quintile_se[0] ** 2 + zipf_quintile_se[-1] ** 2
                )
            else:
                observed_effect = np.nan
                observed_se = np.nan

            if observed_effect > 0:
                logger.info(f"    ⚠️  REVERSAL: Higher frequency -> MORE time")

            bin_results["groups"][group] = {
                "n": len(group_data),
                "zipf_q1": float(zipf_q1),
                "zipf_q3": float(zipf_q3),
                "zipf_iqr": float(zipf_iqr),
                "zipf_min": float(group_data["zipf"].min()),
                "zipf_max": float(group_data["zipf"].max()),
                "length_mean": float(group_data["length"].mean()),
                "observed_ert_by_zipf_quintile": [
                    float(x) for x in observed_ert_by_zipf
                ],
                "zipf_quintile_means": [float(x) for x in zipf_quintile_means],
                "zipf_quintile_se": [float(x) for x in zipf_quintile_se],
                "observed_effect": (
                    float(observed_effect) if not np.isnan(observed_effect) else None
                ),
                "observed_effect_se": (
                    float(observed_se) if not np.isnan(observed_se) else None
                ),
            }

            # Store for contrast plot
            if group == "control":
                bin_control_effect = observed_effect
                bin_control_se = observed_se
            else:
                bin_dyslexic_effect = observed_effect
                bin_dyslexic_se = observed_se

        diagnostic_results["bins"][bin_label] = bin_results

        # Add to contrast data if both groups present
        if bin_control_effect is not None and bin_dyslexic_effect is not None:
            contrast_data["bin_labels"].append(bin_label)
            contrast_data["control_effects"].append(bin_control_effect)
            contrast_data["dyslexic_effects"].append(bin_dyslexic_effect)
            contrast_data["control_se"].append(bin_control_se)
            contrast_data["dyslexic_se"].append(bin_dyslexic_se)

    # ========== CREATE CONTRAST VISUALIZATION ==========
    logger.info("\n" + "=" * 80)
    logger.info("Creating contrast plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(
        "Zipf Frequency Effects within Length Bins:\nDyslexic vs. Control Readers",
        fontsize=14,
        fontweight="bold",
    )

    x_pos = np.arange(len(contrast_data["bin_labels"]))

    # Left panel: Control
    ax1.bar(
        x_pos,
        contrast_data["control_effects"],
        yerr=contrast_data["control_se"],
        color="steelblue",
        alpha=0.7,
        capsize=5,
    )
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax1.set_xlabel("Word Length (characters)", fontsize=11)
    ax1.set_ylabel("Zipf Effect on ERT (ms)\n(High Freq - Low Freq)", fontsize=11)
    ax1.set_title("Control Readers", fontsize=12, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(contrast_data["bin_labels"], rotation=0)
    ax1.grid(axis="y", alpha=0.3)

    # Right panel: Dyslexic
    ax2.bar(
        x_pos,
        contrast_data["dyslexic_effects"],
        yerr=contrast_data["dyslexic_se"],
        color="coral",
        alpha=0.7,
        capsize=5,
    )
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_xlabel("Word Length (characters)", fontsize=11)
    ax2.set_title("Dyslexic Readers", fontsize=12, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(contrast_data["bin_labels"], rotation=0)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "zipf_diagnostic_contrast.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # ========== SAVE JSON ==========
    with open(output_dir / "zipf_diagnostic_data.json", "w") as f:
        json.dump(diagnostic_results, f, indent=2)

    logger.info(f"✅ Diagnostic complete. Saved to {output_dir}/")
    logger.info(f"  - zipf_diagnostic_contrast.png")
    logger.info(f"  - zipf_diagnostic_data.json")

    return diagnostic_results


# Convenience function to run from main pipeline
def run_zipf_diagnostic(data, n_bins=10, results_dir="./results"):
    """Run diagnostic and save to results/diagnostics/"""
    diagnostic_dir = Path(results_dir) / "diagnostics"
    return diagnose_zipf_length_interaction(data, n_bins, diagnostic_dir)
