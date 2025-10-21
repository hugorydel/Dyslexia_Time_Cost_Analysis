"""
Visualization Utilities - REVISED
Key changes:
1. Wider range for smooths (99% vs 95%)
2. Confidence bands added to smooths
3. Unified y-axes per row
4. Added ERT row (3x3 grid: skip, duration, ERT)
5. Removed Figure 2 and Figure 3
6. Model stats now exported to JSON instead of Figure S2
"""

import json
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.titlesize"] = 14


def set_row_ylim(axes_row):
    """Set consistent y-limits across all axes in a row"""
    ymins, ymaxs = [], []
    for ax in axes_row:
        ys = []
        # Get data from filled bands
        for coll in ax.collections:
            if hasattr(coll, "get_paths") and len(coll.get_paths()) > 0:
                try:
                    vertices = coll.get_paths()[0].vertices
                    ys.extend(vertices[:, 1])
                except:
                    pass
        # Get data from lines
        for line in ax.lines:
            ys.extend(line.get_ydata())
        if ys:
            ymins.append(np.nanmin(ys))
            ymaxs.append(np.nanmax(ys))

    if ymins and ymaxs:
        lo, hi = float(np.min(ymins)), float(np.max(ymaxs))
        pad = 0.05 * (hi - lo)
        for ax in axes_row:
            ax.set_ylim(lo - pad, hi + pad)


def create_figure_1_overall_effects(
    h1_results: Dict, h2_results: Dict, output_path: Path
):
    """
    Figure 1: Overall Effects Overview (unchanged)
    """
    logger.info("Creating Figure 1: Overall Effects Overview...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # === Panel A: Delta ERT ===
    features = ["length", "zipf", "surprisal"]
    x = np.arange(len(features))
    width = 0.35

    ctrl_amies = []
    dys_amies = []

    for feat in features:
        feat_data = h1_results["features"][feat]
        ctrl_amie = feat_data["amie_control"].get("amie_ms", 0)
        dys_amie = feat_data["amie_dyslexic"].get("amie_ms", 0)
        ctrl_amies.append(ctrl_amie)
        dys_amies.append(dys_amie)

    ax1.bar(
        x - width / 2, ctrl_amies, width, label="Control", color="steelblue", alpha=0.8
    )
    ax1.bar(x + width / 2, dys_amies, width, label="Dyslexic", color="coral", alpha=0.8)

    ax1.set_xlabel("Feature")
    ax1.set_ylabel("Delta ERT (ms, Q1->Q3)")
    ax1.set_title("Panel A: Feature Effects on Expected Reading Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Length", "Zipf Frequency", "Surprisal"])
    ax1.legend()
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1.grid(axis="y", alpha=0.3)

    # === Panel B: Slope Ratios ===
    pathways = ["skip", "duration", "ert"]
    pathway_labels = ["Skip", "Duration", "ERT"]
    colors_pathway = ["lightblue", "lightcoral", "lightgreen"]

    sr_data = []

    for feat in features:
        if feat in h2_results["slope_ratios"]:
            feat_srs = h2_results["slope_ratios"][feat]

            for pathway in pathways:
                sr = feat_srs.get(f"sr_{pathway}", np.nan)
                ci_low = feat_srs.get(f"sr_{pathway}_ci_low", np.nan)
                ci_high = feat_srs.get(f"sr_{pathway}_ci_high", np.nan)

                sr_data.append(
                    {
                        "feature": feat,
                        "pathway": pathway,
                        "sr": sr,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )

    df_sr = pd.DataFrame(sr_data)

    # Forest plot
    y_pos = 0
    yticks = []
    ytick_labels = []

    for i, feat in enumerate(features):
        feat_data = df_sr[df_sr["feature"] == feat]

        for j, (pathway, label, color) in enumerate(
            zip(pathways, pathway_labels, colors_pathway)
        ):
            pathway_data = feat_data[feat_data["pathway"] == pathway]

            if len(pathway_data) > 0:
                row = pathway_data.iloc[0]
                sr = row["sr"]
                ci_low = row["ci_low"]
                ci_high = row["ci_high"]

                if not np.isnan(sr):
                    ax2.plot(sr, y_pos, "o", color=color, markersize=8)

                    if not np.isnan(ci_low) and not np.isnan(ci_high):
                        ax2.plot(
                            [ci_low, ci_high], [y_pos, y_pos], color=color, linewidth=2
                        )

                yticks.append(y_pos)
                ytick_labels.append(f"{feat.capitalize()}\n{label}")
                y_pos += 1

        y_pos += 0.5

    ax2.axvline(
        x=1.0,
        color="black",
        linestyle="--",
        linewidth=1,
        label="SR = 1.0 (no amplification)",
    )
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ytick_labels, fontsize=9)
    ax2.set_xlabel("Slope Ratio (Dyslexic / Control)")
    ax2.set_title("Panel B: Dyslexic Amplification by Pathway")
    ax2.set_xlim(0, max(3, df_sr["sr"].max() * 1.1))
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved: {output_path}")


def create_figure_s1_gam_smooths(
    ert_predictor, data: pd.DataFrame, gam_models, output_path: Path
):
    """
    Figure S1: GAM Smooth Effects - REVISED
    Now 3x3 grid: Skip, Duration, ERT
    With confidence bands and unified y-axes per row
    """
    logger.info("Creating Figure S1: GAM Smooth Effects (3x3 with confidence bands)...")

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    features = ["length", "zipf", "surprisal"]
    feature_labels = ["Word Length", "Zipf Frequency", "Surprisal"]

    # === Row 1: Skip pathway ===
    for i, (feat, label) in enumerate(zip(features, feature_labels)):
        ax = axes[0, i]

        # Use 99th percentile range
        feat_range = np.linspace(
            data[feat].quantile(0.01), data[feat].quantile(0.99), 100
        )

        other_feats = [f for f in features if f != feat]
        means = {f: data[f].mean() for f in other_feats}

        grid = pd.DataFrame({feat: feat_range, **means})
        X_grid = grid[["length", "zipf", "surprisal"]].values

        for group, color in [("control", "steelblue"), ("dyslexic", "coral")]:
            # Get model
            model = gam_models.skip_model[group]

            # Predictions
            p_skip = model.predict_proba(X_grid)

            # Confidence intervals
            try:
                ci = model.confidence_intervals(X_grid, width=0.95)
                ci_low = 1 / (1 + np.exp(-ci[:, 0]))  # Transform to probability scale
                ci_high = 1 / (1 + np.exp(-ci[:, 1]))

                # Plot line
                ax.plot(
                    feat_range,
                    p_skip,
                    label=group.capitalize(),
                    linewidth=2.5,
                    color=color,
                )

                # Plot confidence band
                ax.fill_between(feat_range, ci_low, ci_high, color=color, alpha=0.2)
            except:
                # Fallback if confidence intervals fail
                ax.plot(
                    feat_range,
                    p_skip,
                    label=group.capitalize(),
                    linewidth=2.5,
                    color=color,
                )

        # Mark quartiles
        q1, q3 = data[feat].quantile([0.25, 0.75])
        ax.axvline(q1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(q3, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel(label)
        ax.set_ylabel("P(skip)")
        if i == 0:
            ax.set_title("Skip Pathway", fontweight="bold", fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

    # Set uniform y-axis for skip row
    set_row_ylim(axes[0, :])

    # === Row 2: Duration pathway ===
    for i, (feat, label) in enumerate(zip(features, feature_labels)):
        ax = axes[1, i]

        feat_range = np.linspace(
            data[feat].quantile(0.01), data[feat].quantile(0.99), 100
        )

        other_feats = [f for f in features if f != feat]
        means = {f: data[f].mean() for f in other_feats}

        grid = pd.DataFrame({feat: feat_range, **means})
        X_grid = grid[["length", "zipf", "surprisal"]].values

        for group, color in [("control", "steelblue"), ("dyslexic", "coral")]:
            model = gam_models.duration_model[group]

            # Predictions (on log scale, then transform back)
            y_log = model.predict(X_grid)
            trt = np.exp(y_log) * gam_models.smearing_factors[group]

            # Confidence intervals
            try:
                ci = model.confidence_intervals(X_grid, width=0.95)
                ci_low = np.exp(ci[:, 0]) * gam_models.smearing_factors[group]
                ci_high = np.exp(ci[:, 1]) * gam_models.smearing_factors[group]

                ax.plot(
                    feat_range,
                    trt,
                    label=group.capitalize(),
                    linewidth=2.5,
                    color=color,
                )
                ax.fill_between(feat_range, ci_low, ci_high, color=color, alpha=0.2)
            except:
                ax.plot(
                    feat_range,
                    trt,
                    label=group.capitalize(),
                    linewidth=2.5,
                    color=color,
                )

        q1, q3 = data[feat].quantile([0.25, 0.75])
        ax.axvline(q1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(q3, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel(label)
        ax.set_ylabel("TRT | fixated (ms)")
        if i == 0:
            ax.set_title("Duration Pathway", fontweight="bold", fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

    # Set uniform y-axis for duration row
    set_row_ylim(axes[1, :])

    # === Row 3: ERT (Combined) ===
    for i, (feat, label) in enumerate(zip(features, feature_labels)):
        ax = axes[2, i]

        feat_range = np.linspace(
            data[feat].quantile(0.01), data[feat].quantile(0.99), 100
        )

        other_feats = [f for f in features if f != feat]
        means = {f: data[f].mean() for f in other_feats}

        grid = pd.DataFrame({feat: feat_range, **means})

        for group, color in [("control", "steelblue"), ("dyslexic", "coral")]:
            # Predict ERT = (1 - P(skip)) × TRT
            ert = ert_predictor.predict_ert(grid, group)

            ax.plot(
                feat_range, ert, label=group.capitalize(), linewidth=2.5, color=color
            )

        q1, q3 = data[feat].quantile([0.25, 0.75])
        ax.axvline(q1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(q3, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel(label)
        ax.set_ylabel("ERT (ms)")
        if i == 0:
            ax.set_title("Combined ERT", fontweight="bold", fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

    # Set uniform y-axis for ERT row
    set_row_ylim(axes[2, :])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved: {output_path}")


def export_model_stats_json(
    gam_models, skip_metadata: Dict, duration_metadata: Dict, output_path: Path
):
    """
    Export model statistics to JSON (replaces Figure S2)
    """
    logger.info("Exporting model statistics to JSON...")

    stats = {
        "skip_model": {
            "control": {
                "auc_training": skip_metadata["auc_control"],
                "auc_cv": skip_metadata["cv_auc_control"],
                "n_obs": skip_metadata["n_obs_control"],
                "n_splines": skip_metadata["n_splines_control"],
                "lambda": skip_metadata.get("lambda_control", None),
                "edf": skip_metadata.get("edf_control", None),
            },
            "dyslexic": {
                "auc_training": skip_metadata["auc_dyslexic"],
                "auc_cv": skip_metadata["cv_auc_dyslexic"],
                "n_obs": skip_metadata["n_obs_dyslexic"],
                "n_splines": skip_metadata["n_splines_dyslexic"],
                "lambda": skip_metadata.get("lambda_dyslexic", None),
                "edf": skip_metadata.get("edf_dyslexic", None),
            },
        },
        "duration_model": {
            "control": {
                "r2_training": duration_metadata["r2_control"],
                "rmse_cv": duration_metadata["cv_rmse_control"],
                "n_obs": duration_metadata["n_obs_control"],
                "n_splines": duration_metadata["n_splines_control"],
                "lambda": duration_metadata.get("lambda_control", None),
                "edf": duration_metadata.get("edf_control", None),
            },
            "dyslexic": {
                "r2_training": duration_metadata["r2_dyslexic"],
                "rmse_cv": duration_metadata["cv_rmse_dyslexic"],
                "n_obs": duration_metadata["n_obs_dyslexic"],
                "n_splines": duration_metadata["n_splines_dyslexic"],
                "lambda": duration_metadata.get("lambda_dyslexic", None),
                "edf": duration_metadata.get("edf_dyslexic", None),
            },
        },
        "model_notes": {
            "family": {
                "skip": skip_metadata["family"],
                "duration": duration_metadata["family"],
            },
            "method": skip_metadata["method"],
            "has_tensor_product": skip_metadata["has_tensor_product"],
        },
    }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"  Saved: {output_path}")


def generate_supplementary_figures(
    ert_predictor,
    data: pd.DataFrame,
    gam_models,
    skip_metadata: Dict,
    duration_metadata: Dict,
    output_dir: Path,
):
    """Generate supplementary figures - REVISED"""
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING SUPPLEMENTARY FIGURES & OUTPUTS")
    logger.info("=" * 60)

    supp_dir = output_dir / "supplementary"
    supp_dir.mkdir(exist_ok=True)

    # Figure S1: GAM Smooths (3x3 with confidence bands)
    create_figure_s1_gam_smooths(
        ert_predictor, data, gam_models, supp_dir / "figure_s1_gam_smooths.png"
    )

    # Model statistics as JSON (replaces Figure S2)
    export_model_stats_json(
        gam_models,
        skip_metadata,
        duration_metadata,
        supp_dir / "model_statistics.json",
    )

    logger.info("Supplementary outputs complete!")


def generate_all_figures(
    ert_predictor,
    data: pd.DataFrame,
    h1_results: Dict,
    h2_results: Dict,
    h3_results: Dict,
    quartiles: Dict,
    gam_models,
    skip_metadata: Dict,
    duration_metadata: Dict,
    output_dir: Path,
):
    """
    Generate all figures - REVISED
    Removed Figure 2 (zipf pathways) and Figure 3 (gap decomposition)
    """
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING PUBLICATION FIGURES")
    logger.info("=" * 60)

    output_dir.mkdir(exist_ok=True, parents=True)

    # Main Figure: Overall Effects
    create_figure_1_overall_effects(
        h1_results, h2_results, output_dir / "figure_1_overall_effects.png"
    )

    logger.info("\nMain figure complete!")

    # Supplementary Figures
    generate_supplementary_figures(
        ert_predictor,
        data,
        gam_models,
        skip_metadata,
        duration_metadata,
        output_dir,
    )

    logger.info("\nAll figures generated successfully!")
    logger.info(f"Figures saved to: {output_dir}")
