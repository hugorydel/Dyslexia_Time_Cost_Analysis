"""
Diagnostic tool for investigating zipf-length interaction effects
APA-7 formatted, grouped Control vs Dyslexic bars per length bin
(Plot error bars are 95% CIs = 1.96 * SE; JSON still stores SEs.)
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
    and generate appropriate labels based on the actual length ranges.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = data["length"].quantile(quantiles).values
    bin_edges = np.unique(bin_edges)

    bin_labels = []
    for i in range(len(bin_edges) - 1):
        start = int(np.ceil(bin_edges[i]))
        end = int(np.floor(bin_edges[i + 1]))

        if i == 0:
            start = int(np.floor(bin_edges[i]))

        if i == len(bin_edges) - 2:
            end = int(np.ceil(bin_edges[i + 1]))
            if end > start:
                bin_labels.append(f"{start}-{end}")
            else:
                bin_labels.append(f"{start}+")
        elif start == end:
            bin_labels.append(f"{start}")
        else:
            bin_labels.append(f"{start}-{end}")

    return bin_edges, bin_labels


def _apply_apa7_style(ax):
    """Minimal APA-7 styling (no in-figure title; grayscale-ready)."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    # No gridlines per APA-7 exemplars
    ax.set_axisbelow(True)
    # Ticks
    ax.tick_params(length=3, width=0.8, direction="out")


def diagnose_zipf_length_interaction(
    data: pd.DataFrame,
    n_bins: int = 10,
    output_dir: Path = None,
):
    """
    Diagnostic for zipf effects within length bins - observed data only.

    Creates:
    1) Grouped bar chart comparing Dyslexic vs Control zipf effects across length bins
       (error bars = 95% CI, grayscale, legend to right)
    2) JSON export of all diagnostic data (schema unchanged; SEs stored)
    """
    if output_dir is None:
        output_dir = Path("./diagnostics")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # === Binning ===
    bin_edges, bin_labels = create_length_bins(data, n_bins)

    data = data.copy()
    data["length_bin"] = pd.cut(
        data["length"], bins=bin_edges, labels=bin_labels, include_lowest=True
    )

    diagnostic_results = {
        "bin_edges": bin_edges.tolist(),
        "bin_labels": bin_labels,
        "bins": {},
    }

    # Data for grouped plot
    contrast_data = {
        "bin_labels": [],
        "control_effects": [],
        "dyslexic_effects": [],
        "control_se": [],
        "dyslexic_se": [],
    }

    # === Per-bin summaries ===
    for bin_idx, bin_label in enumerate(bin_labels):
        bin_start = bin_edges[bin_idx]
        bin_end = bin_edges[bin_idx + 1]
        bin_data = data[data["length_bin"] == bin_label]

        if len(bin_data) < 10:
            logger.info(f"  Skipping {bin_label} (only {len(bin_data)} observations)")
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

            # Zipf quantiles
            zipf_q1 = group_data["zipf"].quantile(0.25)
            zipf_q3 = group_data["zipf"].quantile(0.75)
            zipf_iqr = zipf_q3 - zipf_q1

            # Observed quintiles for ERT-by-Zipf (descriptive)
            g = group_data.copy()
            g["zipf_quintile"] = pd.qcut(
                g["zipf"], q=5, labels=False, duplicates="drop"
            )

            observed_ert_by_zipf = []
            zipf_quintile_means = []
            zipf_quintile_se = []

            for q in range(5):
                qd = g[g["zipf_quintile"] == q]
                if len(qd) > 0:
                    observed_ert_by_zipf.append(float(qd["ERT"].mean()))
                    zipf_quintile_means.append(float(qd["zipf"].mean()))
                    zipf_quintile_se.append(float(qd["ERT"].sem()))

            if len(observed_ert_by_zipf) >= 2:
                # SE for difference between top and bottom quintiles
                observed_effect = float(
                    observed_ert_by_zipf[-1] - observed_ert_by_zipf[0]
                )
                observed_se = float(
                    np.sqrt(zipf_quintile_se[0] ** 2 + zipf_quintile_se[-1] ** 2)
                )
            else:
                observed_effect, observed_se = np.nan, np.nan

            if observed_effect > 0:
                logger.info(
                    f"    ⚠️  {bin_label} {group}: REVERSAL (higher freq → more time)"
                )

            bin_results["groups"][group] = {
                "n": int(len(group_data)),
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
                "zipf_quintile_se": [
                    float(x) for x in zipf_quintile_se
                ],  # SEs kept for JSON
                "observed_effect": (
                    observed_effect if not np.isnan(observed_effect) else None
                ),
                "observed_effect_se": (
                    observed_se if not np.isnan(observed_se) else None
                ),
            }

            if group == "control":
                bin_control_effect, bin_control_se = observed_effect, observed_se
            else:
                bin_dyslexic_effect, bin_dyslexic_se = observed_effect, observed_se

        diagnostic_results["bins"][bin_label] = bin_results

        if (
            bin_control_effect is not None
            and not np.isnan(bin_control_effect)
            and bin_dyslexic_effect is not None
            and not np.isnan(bin_dyslexic_effect)
        ):
            contrast_data["bin_labels"].append(bin_label)
            contrast_data["control_effects"].append(bin_control_effect)
            contrast_data["dyslexic_effects"].append(bin_dyslexic_effect)
            contrast_data["control_se"].append(bin_control_se)
            contrast_data["dyslexic_se"].append(bin_dyslexic_se)

    # === Grouped APA-7 figure (grayscale; legend right; 95% CI) ===
    logger.info("\n" + "=" * 80)
    logger.info("Creating grouped contrast plot (grayscale, 95% CI, legend right)...")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    fig.set_facecolor("white")
    _apply_apa7_style(ax)

    x = np.arange(len(contrast_data["bin_labels"]))
    width = 0.38
    ci_mult = 1.96  # 95% CI under normal approximation

    yerr_ctrl = np.array(contrast_data["control_se"]) * ci_mult
    yerr_dys = np.array(contrast_data["dyslexic_se"]) * ci_mult

    # Grayscale fills; thin black edges; no transparency
    bars_ctrl = ax.bar(
        x - width / 2,
        contrast_data["control_effects"],
        yerr=yerr_ctrl,
        width=width,
        label="Control",
        color="0.35",
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
    )
    bars_dys = ax.bar(
        x + width / 2,
        contrast_data["dyslexic_effects"],
        yerr=yerr_dys,
        width=width,
        label="Dyslexic",
        color="0.75",
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
    )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Word length (characters)")
    ax.set_ylabel("Zipf effect on ERT (ms)\n(High freq – Low freq)")
    ax.set_xticks(x)
    ax.set_xticklabels(contrast_data["bin_labels"])

    # Legend to the right, minimal footprint
    leg = ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        handlelength=1.5,
        handletextpad=0.6,
    )

    plt.tight_layout(pad=0.6)

    # Save with legacy and new names (higher dpi for grayscale line art)
    grouped_path_legacy = output_dir / "zipf_diagnostic_contrast.png"
    grouped_path_new = output_dir / "zipf_diagnostic_grouped.png"
    plt.savefig(grouped_path_legacy, dpi=600, bbox_inches="tight")
    plt.savefig(grouped_path_new, dpi=600, bbox_inches="tight")
    plt.close()

    # === JSON (schema unchanged; still stores SEs) ===
    with open(output_dir / "zipf_diagnostic_data.json", "w") as f:
        json.dump(diagnostic_results, f, indent=2)

    logger.info(f"✅ Diagnostic complete. Saved to {output_dir}/")
    logger.info("  - zipf_diagnostic_contrast.png (grouped, APA-7 grayscale, 95% CI)")
    logger.info("  - zipf_diagnostic_grouped.png (same image, clearer name)")
    logger.info("  - zipf_diagnostic_data.json (SEs)")

    return diagnostic_results


def run_zipf_diagnostic(data, n_bins=10, results_dir="./results"):
    """Run diagnostic and save to results/diagnostics/"""
    diagnostic_dir = Path(results_dir) / "diagnostics"
    return diagnose_zipf_length_interaction(data, n_bins, diagnostic_dir)
