"""
Visualization Utilities for Analysis Plan Figures
Generates publication-quality figures as specified in the analysis plan
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

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


def create_figure_1_overall_effects(
    h1_results: Dict, h2_results: Dict, output_path: Path
):
    """
    Figure 1: Overall Effects Overview

    Layout: 1 row × 2 columns
    Panel A: ΔERT (Q1→Q3) by feature and group
    Panel B: Slope Ratios by pathway and feature
    """
    logger.info("Creating Figure 1: Overall Effects Overview...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # === Panel A: ΔERT ===
    features = ["length", "zipf", "surprisal"]
    x = np.arange(len(features))
    width = 0.35

    # Extract AMIEs
    ctrl_amies = []
    dys_amies = []

    for feat in features:
        feat_data = h1_results["features"][feat]
        ctrl_amie = feat_data["amie_control"].get("amie_ms", 0)
        dys_amie = feat_data["amie_dyslexic"].get("amie_ms", 0)
        ctrl_amies.append(ctrl_amie)
        dys_amies.append(dys_amie)

    # Plot bars
    ax1.bar(
        x - width / 2, ctrl_amies, width, label="Control", color="steelblue", alpha=0.8
    )
    ax1.bar(x + width / 2, dys_amies, width, label="Dyslexic", color="coral", alpha=0.8)

    ax1.set_xlabel("Feature")
    ax1.set_ylabel("ΔERT (ms, Q1→Q3)")
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

    # Extract SRs with CIs
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
                    # Plot point and CI
                    ax2.plot(sr, y_pos, "o", color=color, markersize=8)

                    if not np.isnan(ci_low) and not np.isnan(ci_high):
                        ax2.plot(
                            [ci_low, ci_high], [y_pos, y_pos], color=color, linewidth=2
                        )

                    # Mark unstable SRs
                    if feat == "zipf" and pathway == "skip":
                        ax2.text(
                            sr, y_pos + 0.2, "†", ha="center", fontsize=12, color="red"
                        )

                yticks.append(y_pos)
                ytick_labels.append(f"{feat.capitalize()}\n{label}")
                y_pos += 1

        y_pos += 0.5  # Space between features

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

    # Add note about unstable SRs
    fig.text(
        0.99,
        0.01,
        "† Flagged: unstable denominator (control Δp ≈ 0)",
        ha="right",
        fontsize=8,
        style="italic",
        color="red",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved: {output_path}")


def create_figure_2_zipf_pathways(
    ert_predictor,
    data: pd.DataFrame,
    quartiles: Dict,
    h2_results: Dict,
    output_path: Path,
):
    """
    Figure 2: Zipf Pathway Decomposition (FIXED - smooth curves)
    """
    logger.info("Creating Figure 2: Zipf Pathway Decomposition...")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # FIXED: Use SMOOTH prediction grid (200 points, no bin conditioning for plotting)
    zipf_q1 = quartiles["zipf"]["q1"]
    zipf_q3 = quartiles["zipf"]["q3"]

    # Extend range slightly for visualization
    zipf_min = data["zipf"].quantile(0.05)
    zipf_max = data["zipf"].quantile(0.95)
    zipf_range = np.linspace(zipf_min, zipf_max, 200)  # High resolution

    # Fix other features at POOLED medians (not means, more robust)
    length_median = data["length"].median()
    surprisal_median = data["surprisal"].median()

    grid = pd.DataFrame(
        {"length": length_median, "zipf": zipf_range, "surprisal": surprisal_median}
    )

    # Get predictions for both groups
    results = {"control": {}, "dyslexic": {}}

    for group in ["control", "dyslexic"]:
        ert, p_skip, trt = ert_predictor.predict_ert(
            grid, group, return_components=True
        )
        results[group]["p_skip"] = p_skip
        results[group]["trt"] = trt
        results[group]["ert"] = ert

    # Colors
    colors = {"control": "steelblue", "dyslexic": "coral"}

    # === Panel A: Skip Pathway ===
    for group in ["control", "dyslexic"]:
        ax1.plot(
            zipf_range,
            results[group]["p_skip"],
            label=group.capitalize(),
            color=colors[group],
            linewidth=2.5,
            alpha=0.9,
        )

    # Mark Q1 and Q3
    ax1.axvline(zipf_q1, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ax1.axvline(zipf_q3, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ax1.text(zipf_q1, 0.02, "Q1", ha="center", fontsize=9, color="gray")
    ax1.text(zipf_q3, 0.02, "Q3", ha="center", fontsize=9, color="gray")

    ax1.set_xlabel("Zipf Frequency (rare → frequent)", fontsize=11)
    ax1.set_ylabel("P(skip)", fontsize=11)
    ax1.set_title("Panel A: Skip Pathway", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(alpha=0.3)

    # Add SR annotation
    if "zipf" in h2_results["slope_ratios"]:
        sr_skip = h2_results["slope_ratios"]["zipf"].get("sr_skip", np.nan)
        if not np.isnan(sr_skip):
            ax1.text(
                0.95,
                0.95,
                f"SR(skip) = {sr_skip:.2f}†",
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            )

    # === Panel B: Duration Pathway ===
    for group in ["control", "dyslexic"]:
        ax2.plot(
            zipf_range,
            results[group]["trt"],
            label=group.capitalize(),
            color=colors[group],
            linewidth=2.5,
            alpha=0.9,
        )

    ax2.axvline(zipf_q1, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ax2.axvline(zipf_q3, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)

    # Position labels at bottom
    ylim = ax2.get_ylim()
    ax2.text(
        zipf_q1,
        ylim[0] + (ylim[1] - ylim[0]) * 0.02,
        "Q1",
        ha="center",
        fontsize=9,
        color="gray",
    )
    ax2.text(
        zipf_q3,
        ylim[0] + (ylim[1] - ylim[0]) * 0.02,
        "Q3",
        ha="center",
        fontsize=9,
        color="gray",
    )

    ax2.set_xlabel("Zipf Frequency (rare → frequent)", fontsize=11)
    ax2.set_ylabel("TRT | fixated (ms)", fontsize=11)
    ax2.set_title(
        "Panel B: Duration Pathway\n(conditioned within length bins)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(alpha=0.3)

    # Add SR annotation
    if "zipf" in h2_results["slope_ratios"]:
        sr_dur = h2_results["slope_ratios"]["zipf"].get("sr_duration", np.nan)
        ci_low = h2_results["slope_ratios"]["zipf"].get("sr_duration_ci_low", np.nan)
        ci_high = h2_results["slope_ratios"]["zipf"].get("sr_duration_ci_high", np.nan)
        if not np.isnan(sr_dur):
            ax2.text(
                0.05,
                0.95,
                f"SR(duration) = {sr_dur:.2f}\n[{ci_low:.2f}, {ci_high:.2f}]",
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
            )

    # === Panel C: Combined ERT ===
    for group in ["control", "dyslexic"]:
        ax3.plot(
            zipf_range,
            results[group]["ert"],
            label=group.capitalize(),
            color=colors[group],
            linewidth=2.5,
            alpha=0.9,
        )

    ax3.axvline(zipf_q1, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ax3.axvline(zipf_q3, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)

    ylim = ax3.get_ylim()
    ax3.text(
        zipf_q1,
        ylim[0] + (ylim[1] - ylim[0]) * 0.02,
        "Q1",
        ha="center",
        fontsize=9,
        color="gray",
    )
    ax3.text(
        zipf_q3,
        ylim[0] + (ylim[1] - ylim[0]) * 0.02,
        "Q3",
        ha="center",
        fontsize=9,
        color="gray",
    )

    ax3.set_xlabel("Zipf Frequency (rare → frequent)", fontsize=11)
    ax3.set_ylabel("ERT (ms)", fontsize=11)
    ax3.set_title("Panel C: Combined ERT", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(alpha=0.3)

    # Add SR annotation
    if "zipf" in h2_results["slope_ratios"]:
        sr_ert = h2_results["slope_ratios"]["zipf"].get("sr_ert", np.nan)
        if not np.isnan(sr_ert):
            ax3.text(
                0.95,
                0.95,
                f"SR(ERT) = {sr_ert:.2f}",
                transform=ax3.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
            )

    # Caption
    fig.text(
        0.5,
        0.02,
        "Frequency operates via opposing pathways: controls skip more frequent words (Panel A), while dyslexics show amplified\n"
        "duration costs for rare words (Panel B). These opposing effects partially cancel in combined ERT (Panel C).",
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.text(
        0.99, 0.01, "† Unstable: control Δp ≈ 0", ha="right", fontsize=8, color="red"
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved: {output_path}")


def create_figure_3_gap_decomposition(h3_results: Dict, output_path: Path):
    """
    Figure 3: Gap Decomposition Waterfall
    """
    logger.info("Creating Figure 3: Gap Decomposition Waterfall...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data
    shapley = h3_results["shapley_decomposition"]
    equal_ease = h3_results["equal_ease_counterfactual"]

    total_gap = shapley["total_gap"]
    skip_contrib = shapley["skip_contribution"]
    duration_contrib = shapley["duration_contribution"]

    # Waterfall data
    categories = [
        "Total Gap",
        "Skip\nContribution",
        "Duration\nContribution",
        "Baseline\n(Zero)",
    ]
    values = [total_gap, -skip_contrib, -duration_contrib, 0]
    cumulative = [0, total_gap, total_gap - skip_contrib, 0]

    # Colors
    colors = ["coral", "steelblue", "lightcoral", "gray"]

    # Plot waterfall
    for i, (cat, val, cum, color) in enumerate(
        zip(categories, values, cumulative, colors)
    ):
        if i == 0:
            ax.bar(i, val, color=color, alpha=0.7, width=0.6)
        elif i < len(categories) - 1:
            ax.bar(i, val, bottom=cum, color=color, alpha=0.7, width=0.6)
            if i > 0:
                ax.plot([i - 0.5, i - 0.3], [cum, cum], "k--", linewidth=1)
        else:
            ax.axhline(0, color="black", linewidth=2)

    # Annotations
    ax.text(
        0,
        total_gap + 5,
        f"{total_gap:.1f} ms\n(100%)",
        ha="center",
        fontsize=10,
        weight="bold",
    )
    ax.text(
        1,
        cumulative[1] - skip_contrib / 2,
        f'-{skip_contrib:.1f} ms\n({shapley["skip_pct"]:.0f}%)',
        ha="center",
        fontsize=9,
        color="white",
        weight="bold",
    )
    ax.text(
        2,
        cumulative[2] - duration_contrib / 2,
        f'-{duration_contrib:.1f} ms\n({shapley["duration_pct"]:.0f}%)',
        ha="center",
        fontsize=9,
        color="white",
        weight="bold",
    )

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Reading Time Gap (ms)", fontsize=11)
    ax.set_title(
        "Gap Decomposition: Skip vs Duration Contributions", fontsize=12, weight="bold"
    )
    ax.grid(axis="y", alpha=0.3)

    # FIXED: Add equal-ease inset with NUMERIC x positions
    inset = ax.inset_axes([0.65, 0.6, 0.3, 0.35])

    gap_shrink_pct = equal_ease["gap_shrink_pct"]
    baseline_gap = equal_ease["baseline_gap"]
    cf_gap = equal_ease["counterfactual_gap"]

    labels = ["Baseline", "Equal-Ease"]
    x_pos = np.array([0, 1])  # NUMERIC positions

    inset.bar(
        x_pos,
        [baseline_gap, cf_gap],
        color=["coral", "lightgreen"],
        alpha=0.7,
        width=0.6,
    )

    # Set ticks explicitly
    inset.set_xticks(x_pos)
    inset.set_xticklabels(labels, fontsize=9)
    inset.set_ylabel("Gap (ms)", fontsize=9)
    inset.set_title(f"Equal-Ease: {gap_shrink_pct:.0f}% Reduction", fontsize=9)
    inset.grid(axis="y", alpha=0.3)

    # Add shrink annotation
    inset.annotate(
        "",
        xy=(x_pos[1], cf_gap),
        xytext=(x_pos[1], baseline_gap),
        arrowprops=dict(arrowstyle="<->", color="red", lw=2),
    )
    inset.text(
        x_pos[1] + 0.15,
        (baseline_gap + cf_gap) / 2,
        f'{equal_ease["gap_shrink_ms"]:.0f} ms\nsaved',
        fontsize=8,
        color="red",
        weight="bold",
        va="center",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved: {output_path}")


def generate_supplementary_figures(
    ert_predictor,
    data: pd.DataFrame,
    gam_models,
    skip_metadata: Dict,
    duration_metadata: Dict,
    output_dir: Path,
):
    """Generate all supplementary figures"""
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING SUPPLEMENTARY FIGURES")
    logger.info("=" * 60)

    supp_dir = output_dir / "supplementary"
    supp_dir.mkdir(exist_ok=True)

    # Figure S1: GAM Smooth Effects
    create_figure_s1_gam_smooths(
        ert_predictor, data, supp_dir / "figure_s1_gam_smooths.png"
    )

    # Figure S2: Model Diagnostics
    create_figure_s2_diagnostics(
        gam_models,
        skip_metadata,
        duration_metadata,
        supp_dir / "figure_s2_diagnostics.png",
    )

    logger.info("✅ Supplementary figures complete!")


def create_figure_s1_gam_smooths(ert_predictor, data: pd.DataFrame, output_path: Path):
    """Figure S1: GAM Smooth Effects (2×3 panel)"""
    logger.info("Creating Figure S1: GAM Smooth Effects...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    features = ["length", "zipf", "surprisal"]
    feature_labels = ["Word Length", "Zipf Frequency", "Surprisal"]

    # Row 1: Skip pathway
    for i, (feat, label) in enumerate(zip(features, feature_labels)):
        ax = axes[0, i]

        feat_range = np.linspace(
            data[feat].quantile(0.05), data[feat].quantile(0.95), 100
        )

        other_feats = [f for f in features if f != feat]
        means = {f: data[f].mean() for f in other_feats}

        grid = pd.DataFrame({feat: feat_range, **means})

        for group, color in [("control", "steelblue"), ("dyslexic", "coral")]:
            p_skip = ert_predictor.predict_skip_probability(grid, group)
            ax.plot(
                feat_range, p_skip, label=group.capitalize(), linewidth=2.5, color=color
            )

        q1, q3 = data[feat].quantile([0.25, 0.75])
        ax.axvline(q1, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(q3, color="gray", linestyle="--", alpha=0.5)

        ax.set_xlabel(label)
        ax.set_ylabel("P(skip)")
        if i == 0:
            ax.set_title("Skip Pathway", fontweight="bold", fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

    # Row 2: Duration pathway
    for i, (feat, label) in enumerate(zip(features, feature_labels)):
        ax = axes[1, i]

        feat_range = np.linspace(
            data[feat].quantile(0.05), data[feat].quantile(0.95), 100
        )

        other_feats = [f for f in features if f != feat]
        means = {f: data[f].mean() for f in other_feats}

        grid = pd.DataFrame({feat: feat_range, **means})

        for group, color in [("control", "steelblue"), ("dyslexic", "coral")]:
            trt = ert_predictor.predict_trt_given_fixation(grid, group)
            ax.plot(
                feat_range, trt, label=group.capitalize(), linewidth=2.5, color=color
            )

        q1, q3 = data[feat].quantile([0.25, 0.75])
        ax.axvline(q1, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(q3, color="gray", linestyle="--", alpha=0.5)

        ax.set_xlabel(label)
        ax.set_ylabel("TRT | fixated (ms)")
        if i == 0:
            ax.set_title("Duration Pathway", fontweight="bold", fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved: {output_path}")


def create_figure_s2_diagnostics(
    gam_models, skip_metadata: Dict, duration_metadata: Dict, output_path: Path
):
    """Figure S2: Model Diagnostics (2×2 panel)"""
    logger.info("Creating Figure S2: Model Diagnostics...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Skip Model AUC
    ax = axes[0, 0]
    groups = ["Control", "Dyslexic"]
    aucs = [skip_metadata["auc_control"], skip_metadata["auc_dyslexic"]]
    cv_aucs = [skip_metadata["cv_auc_control"], skip_metadata["cv_auc_dyslexic"]]

    x = np.arange(len(groups))
    width = 0.35

    ax.bar(
        x - width / 2, aucs, width, label="Training AUC", alpha=0.8, color="steelblue"
    )
    ax.bar(x + width / 2, cv_aucs, width, label="CV AUC", alpha=0.8, color="coral")

    ax.set_ylabel("AUC", fontsize=11)
    ax.set_title("Skip Model Performance", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0.5, 0.8])
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.3, label="Chance")

    # Panel B: Duration Model R²
    ax = axes[0, 1]
    r2s = [duration_metadata["r2_control"], duration_metadata["r2_dyslexic"]]

    ax.bar(groups, r2s, color=["steelblue", "coral"], alpha=0.8)
    ax.set_ylabel("Pseudo-R² (explained deviance)", fontsize=11)
    ax.set_title("Duration Model Performance", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 0.15])

    # Add value labels on bars
    for i, (group, r2) in enumerate(zip(groups, r2s)):
        ax.text(i, r2 + 0.002, f"{r2:.3f}", ha="center", va="bottom", fontsize=10)

    # Panel C: Sample Sizes
    ax = axes[1, 0]
    skip_n = [skip_metadata["n_obs_control"], skip_metadata["n_obs_dyslexic"]]
    dur_n = [duration_metadata["n_obs_control"], duration_metadata["n_obs_dyslexic"]]

    x = np.arange(len(groups))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        skip_n,
        width,
        label="Skip (all words)",
        alpha=0.8,
        color="lightblue",
    )
    bars2 = ax.bar(
        x + width / 2,
        dur_n,
        width,
        label="Duration (fixated)",
        alpha=0.8,
        color="lightcoral",
    )

    ax.set_ylabel("N observations", fontsize=11)
    ax.set_title("Model Training Sample Sizes", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    # Panel D: Model Complexity
    ax = axes[1, 1]
    n_splines_data = {
        "Control": [
            skip_metadata["n_splines_control"],
            duration_metadata["n_splines_control"],
        ],
        "Dyslexic": [
            skip_metadata["n_splines_dyslexic"],
            duration_metadata["n_splines_dyslexic"],
        ],
    }

    models = ["Skip", "Duration"]
    x = np.arange(len(models))
    width = 0.35

    ax.bar(
        x - width / 2,
        [n_splines_data["Control"][0], n_splines_data["Control"][1]],
        width,
        label="Control",
        alpha=0.8,
        color="steelblue",
    )
    ax.bar(
        x + width / 2,
        [n_splines_data["Dyslexic"][0], n_splines_data["Dyslexic"][1]],
        width,
        label="Dyslexic",
        alpha=0.8,
        color="coral",
    )

    ax.set_ylabel("N splines (selected)", fontsize=11)
    ax.set_title("Model Complexity", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 15])

    # Add note
    fig.text(
        0.5,
        0.02,
        "Note: Models selected via 1-SE rule in nested cross-validation. "
        "All models include tensor product te(length, zipf) interaction.",
        ha="center",
        fontsize=9,
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved: {output_path}")


def generate_all_figures(
    ert_predictor,
    data: pd.DataFrame,
    h1_results: Dict,
    h2_results: Dict,
    h3_results: Dict,
    quartiles: Dict,
    gam_models,  # NEW parameter
    skip_metadata: Dict,  # NEW parameter
    duration_metadata: Dict,  # NEW parameter
    output_dir: Path,
):
    """Generate all main AND supplementary figures"""
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING PUBLICATION FIGURES")
    logger.info("=" * 60)

    output_dir.mkdir(exist_ok=True, parents=True)

    # Main Figures
    create_figure_1_overall_effects(
        h1_results, h2_results, output_dir / "figure_1_overall_effects.png"
    )

    create_figure_2_zipf_pathways(
        ert_predictor,
        data,
        quartiles,
        h2_results,
        output_dir / "figure_2_zipf_pathways.png",
    )

    create_figure_3_gap_decomposition(
        h3_results, output_dir / "figure_3_gap_decomposition.png"
    )

    logger.info("\n✅ Main figures complete!")

    # Supplementary Figures
    generate_supplementary_figures(
        ert_predictor,
        data,
        gam_models,
        skip_metadata,
        duration_metadata,
        output_dir,
    )

    logger.info("\n✅ All figures generated successfully!")
    logger.info(f"Figures saved to: {output_dir}")
