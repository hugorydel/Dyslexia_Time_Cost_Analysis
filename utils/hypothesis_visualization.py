# utils/hypothesis_visualization.py
"""
Visualization utilities for hypothesis testing results
Creates publication-quality figures
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import make_interp_spline

logger = logging.getLogger(__name__)


def create_hypothesis_testing_visualizations(
    data: pd.DataFrame,
    quartile_results: dict,
    continuous_results: dict,
    gap_results: dict,
    output_dir: Path,
) -> None:
    """
    Create all visualizations for hypothesis testing
    """
    logger.info("=" * 60)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. Continuous effect curves with Q1/Q3 overlays
    logger.info("\n1. Continuous effect curves...")
    plot_continuous_effects_with_quartiles(
        data,
        quartile_results,
        continuous_results,
        output_dir / "continuous_effects_with_quartiles.png",
    )

    # 2. Gap decomposition bar chart
    logger.info("2. Gap decomposition...")
    plot_gap_decomposition(gap_results, output_dir / "gap_decomposition.png")

    # 3. Slope ratio visualization
    logger.info("3. Slope ratios...")
    plot_slope_ratios(continuous_results, output_dir / "slope_ratios.png")

    # 4. Model comparison (R²)
    logger.info("4. Model comparison...")
    plot_model_comparison(gap_results, output_dir / "model_comparison.png")

    # 5. Quartile effects by feature
    logger.info("5. Quartile effects...")
    plot_quartile_effects_grid(quartile_results, output_dir / "quartile_effects.png")

    # 6. Predicted vs observed ERT
    logger.info("6. Predicted vs observed...")
    plot_predicted_vs_observed(
        continuous_results, output_dir / "predicted_vs_observed.png"
    )

    logger.info(f"\nAll visualizations saved to {output_dir}")


def plot_continuous_effects_with_quartiles(
    data: pd.DataFrame,
    quartile_results: dict,
    continuous_results: dict,
    output_path: Path,
) -> None:
    """
    Plot continuous effect curves with Q1/Q3 quartile points overlaid
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    features = ["word_length", "word_frequency_zipf", "surprisal"]
    feature_labels = [
        "Word Length (characters)",
        "Word Frequency (Zipf)",
        "Surprisal (bits)",
    ]
    colors = {"control": "#2E7D32", "dyslexic": "#D84315"}

    for idx, (feature, label) in enumerate(zip(features, feature_labels)):
        ax = axes[idx]

        if feature not in data.columns:
            ax.text(
                0.5,
                0.5,
                f"{feature} not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # Get participant means from quartile results
        if feature in quartile_results:
            participant_df = quartile_results[feature].get("participant_means")

            if participant_df is not None and not participant_df.empty:
                # Plot continuous smooth curves using binned means
                for dyslexic, group_name in [(False, "control"), (True, "dyslexic")]:
                    group_data = data[data["dyslexic"] == dyslexic]

                    # Create bins for smooth curve
                    n_bins = 20
                    try:
                        group_data["bin"] = pd.qcut(
                            group_data[feature],
                            q=n_bins,
                            labels=False,
                            duplicates="drop",
                        )

                        # Compute mean ERT per bin
                        bin_means = (
                            group_data.groupby("bin")
                            .agg({feature: "mean", "ERT": "mean"})
                            .reset_index()
                        )

                        # Sort by feature value
                        bin_means = bin_means.sort_values(feature)

                        # Plot smooth curve
                        if len(bin_means) > 3:
                            x = bin_means[feature].values
                            y = bin_means["ERT"].values

                            # Smooth with spline
                            try:
                                x_smooth = np.linspace(x.min(), x.max(), 100)
                                spl = make_interp_spline(x, y, k=min(3, len(x) - 1))
                                y_smooth = spl(x_smooth)

                                ax.plot(
                                    x_smooth,
                                    y_smooth,
                                    label=group_name.capitalize(),
                                    color=colors[group_name],
                                    linewidth=2.5,
                                    alpha=0.8,
                                )
                            except:
                                # Fall back to line plot
                                ax.plot(
                                    x,
                                    y,
                                    label=group_name.capitalize(),
                                    color=colors[group_name],
                                    linewidth=2.5,
                                    alpha=0.8,
                                    marker="o",
                                )
                    except:
                        pass

                # Overlay Q1/Q3 points
                for dyslexic, group_name in [(False, "control"), (True, "dyslexic")]:
                    group_part = participant_df[participant_df["dyslexic"] == dyslexic]

                    if len(group_part) > 0:
                        q1_mean = group_part["Q1_mean"].mean()
                        q3_mean = group_part["Q3_mean"].mean()

                        # Get Q1/Q3 values
                        q1_val = data[feature].quantile(0.25)
                        q3_val = data[feature].quantile(0.75)

                        ax.scatter(
                            [q1_val, q3_val],
                            [q1_mean, q3_mean],
                            color=colors[group_name],
                            s=150,
                            zorder=5,
                            edgecolors="black",
                            linewidths=2,
                            marker="D",
                            alpha=0.9,
                        )

        ax.set_xlabel(label, fontsize=12, fontweight="bold")
        ax.set_ylabel("Expected Reading Time (ms)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(
        "Feature Effects on Reading Time by Group\n(with Q1/Q3 markers)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved to {output_path}")


def plot_gap_decomposition(gap_results: dict, output_path: Path) -> None:
    """
    Bar chart showing reduction in group gap across models
    """
    gaps = gap_results.get("gaps", {})
    pct_explained = gap_results.get("percent_explained", {})

    if not gaps:
        logger.warning("No gap results available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    models = ["M0", "M1", "M2", "M3"]
    labels = ["M0\n(Baseline)", "M1\n(+Length)", "M2\n(+Frequency)", "M3\n(+Surprisal)"]
    gap_values = [gaps.get(m, np.nan) for m in models]
    colors = ["#78909C", "#64B5F6", "#42A5F5", "#1976D2"]

    bars = ax.bar(labels, gap_values, color=colors, edgecolor="black", linewidth=1.5)

    # Add percentage explained labels
    for i, (bar, model) in enumerate(zip(bars, models)):
        if model in pct_explained and i > 0:
            pct = pct_explained[model]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                f"{pct:.1f}%\nexplained",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.8,
                ),
            )

    ax.set_ylabel("Group Gap (ms)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Reduction in Dyslexic-Control Gap Across Models",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add text box with final gap
    final_gap = gap_values[-1]
    if not np.isnan(final_gap):
        textstr = f"Residual Gap: {final_gap:.1f}ms"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.98,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved to {output_path}")


def plot_slope_ratios(continuous_results: dict, output_path: Path) -> None:
    """
    Horizontal bar chart of slope ratios with confidence intervals
    """
    slope_ratios = continuous_results.get("slope_ratios", {})
    slope_ratio_cis = continuous_results.get("slope_ratio_cis", {})

    if not slope_ratios:
        logger.warning("No slope ratio results available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    features = ["Length", "Frequency", "Surprisal"]
    srs = [slope_ratios.get(f, {}).get("slope_ratio", np.nan) for f in features]

    # Get CIs
    ci_lows = []
    ci_highs = []
    for f in features:
        if f in slope_ratio_cis:
            ci_lows.append(slope_ratio_cis[f].get("ci_low", np.nan))
            ci_highs.append(slope_ratio_cis[f].get("ci_high", np.nan))
        else:
            ci_lows.append(np.nan)
            ci_highs.append(np.nan)

    # Compute error bars
    errors_low = [
        sr - ci_low if not np.isnan(ci_low) else 0 for sr, ci_low in zip(srs, ci_lows)
    ]
    errors_high = [
        ci_high - sr if not np.isnan(ci_high) else 0
        for sr, ci_high in zip(srs, ci_highs)
    ]

    # Colors: green if SR > 1 (amplification), red if < 1
    colors = ["#D32F2F" if sr > 1 else "#1976D2" for sr in srs]

    y_pos = np.arange(len(features))
    bars = ax.barh(
        y_pos,
        srs,
        xerr=[errors_low, errors_high],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        capsize=5,
        error_kw={"linewidth": 2},
    )

    # Add reference line at 1.0
    ax.axvline(
        x=1,
        color="black",
        linestyle="--",
        linewidth=2,
        label="No amplification (SR=1)",
        zorder=0,
    )

    # Add SR values as text
    for i, (sr, bar) in enumerate(zip(srs, bars)):
        if not np.isnan(sr):
            ax.text(
                sr + 0.02,
                i,
                f"{sr:.3f}",
                va="center",
                ha="left",
                fontsize=11,
                fontweight="bold",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=12, fontweight="bold")
    ax.set_xlabel("Slope Ratio (Dyslexic / Control)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Dyslexic Amplification of Feature Effects\n(Slope Ratios with 95% CI)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add shaded region for amplification (SR > 1)
    ax.axvspan(
        1, ax.get_xlim()[1], alpha=0.1, color="red", label="Amplification region"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved to {output_path}")


def plot_model_comparison(gap_results: dict, output_path: Path) -> None:
    """
    Line plot showing R² improvement across models
    """
    r2_values = gap_results.get("r2_values", {})

    if not r2_values:
        logger.warning("No R² results available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    models = ["M0", "M1", "M2", "M3"]
    r2s = [r2_values.get(m, np.nan) for m in models]

    # Plot R² progression
    ax.plot(
        models,
        r2s,
        marker="o",
        markersize=12,
        linewidth=3,
        color="#1976D2",
        label="Marginal R²",
    )

    # Fill area under curve
    ax.fill_between(range(len(models)), 0, r2s, alpha=0.2, color="#1976D2")

    # Add R² values as text
    for i, (model, r2) in enumerate(zip(models, r2s)):
        if not np.isnan(r2):
            ax.text(
                i,
                r2 + 0.01,
                f"{r2:.4f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"
                ),
            )

    # Compute and show ΔR²
    for i in range(1, len(models)):
        if not np.isnan(r2s[i]) and not np.isnan(r2s[i - 1]):
            delta_r2 = r2s[i] - r2s[i - 1]
            mid_x = i - 0.5
            mid_y = (r2s[i] + r2s[i - 1]) / 2
            ax.annotate(
                f"ΔR²={delta_r2:.4f}",
                xy=(mid_x, mid_y),
                xytext=(mid_x, mid_y - 0.03),
                ha="center",
                fontsize=9,
                style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
            )

    ax.set_xlabel("Model", fontsize=13, fontweight="bold")
    ax.set_ylabel("Marginal R² (Fixed Effects)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Model Comparison: R² Improvement with Additional Predictors",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add model descriptions
    descriptions = [
        "Baseline\n(Group only)",
        "Add Length\n(+interaction)",
        "Add Frequency\n(+interaction)",
        "Add Surprisal\n(+interaction)",
    ]
    for i, desc in enumerate(descriptions):
        ax.text(
            i,
            -0.05,
            desc,
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
            color="gray",
            transform=ax.get_xaxis_transform(),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved to {output_path}")


def plot_quartile_effects_grid(quartile_results: dict, output_path: Path) -> None:
    """
    Grid of quartile effects (Q1 vs Q3) for each feature
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    features = ["word_length", "word_frequency_zipf", "surprisal"]
    feature_labels = ["Word Length", "Word Frequency", "Surprisal"]
    colors = {"Control": "#2E7D32", "Dyslexic": "#D84315"}

    for idx, (feature, label) in enumerate(zip(features, feature_labels)):
        ax = axes[idx]

        if feature not in quartile_results:
            ax.text(
                0.5,
                0.5,
                f"{feature} not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        participant_df = quartile_results[feature].get("participant_means")

        if participant_df is None or participant_df.empty:
            continue

        # Prepare data for plotting
        plot_data = []
        for dyslexic in [False, True]:
            group_name = "Dyslexic" if dyslexic else "Control"
            group_df = participant_df[participant_df["dyslexic"] == dyslexic]

            q1_mean = group_df["Q1_mean"].mean()
            q1_se = group_df["Q1_mean"].sem()

            q3_mean = group_df["Q3_mean"].mean()
            q3_se = group_df["Q3_mean"].sem()

            plot_data.append(
                {
                    "Group": group_name,
                    "Q1": q1_mean,
                    "Q1_se": q1_se,
                    "Q3": q3_mean,
                    "Q3_se": q3_se,
                }
            )

        plot_df = pd.DataFrame(plot_data)

        # Plot bars
        x = np.arange(2)
        width = 0.35

        for i, group in enumerate(["Control", "Dyslexic"]):
            group_row = plot_df[plot_df["Group"] == group].iloc[0]

            q1_val = group_row["Q1"]
            q3_val = group_row["Q3"]
            q1_err = group_row["Q1_se"]
            q3_err = group_row["Q3_se"]

            offset = -width / 2 if group == "Control" else width / 2

            ax.bar(
                x[0] + offset,
                q1_val,
                width,
                yerr=q1_err,
                capsize=5,
                label=group if idx == 0 else "",
                color=colors[group],
                edgecolor="black",
                linewidth=1.5,
                alpha=0.8,
            )
            ax.bar(
                x[1] + offset,
                q3_val,
                width,
                yerr=q3_err,
                capsize=5,
                color=colors[group],
                edgecolor="black",
                linewidth=1.5,
                alpha=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(["Q1\n(Low)", "Q3\n(High)"], fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Reading Time (ms)", fontsize=12, fontweight="bold")
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if idx == 0:
            ax.legend(fontsize=11, loc="upper left")

    plt.suptitle(
        "Quartile Effects: Q1 vs Q3 by Feature and Group",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved to {output_path}")


def plot_predicted_vs_observed(continuous_results: dict, output_path: Path) -> None:
    """
    Scatter plot of predicted vs observed ERT
    """
    predictions = continuous_results.get("predictions")

    # Check for predictions column
    if predictions is None or "ERT_expected" not in predictions.columns:
        logger.warning("No predictions available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, dyslexic in enumerate([False, True]):
        ax = axes[idx]
        group_name = "Dyslexic" if dyslexic else "Control"

        group_data = predictions[predictions["dyslexic"] == dyslexic].copy()

        # CRITICAL FIX: Drop rows with NaN in either observed or predicted
        group_data = group_data.dropna(subset=["ERT", "ERT_expected"])

        if len(group_data) == 0:
            logger.warning(f"No valid data for {group_name} group")
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # Sample if too many points
        if len(group_data) > 10000:
            group_data = group_data.sample(10000, random_state=42)

        observed = group_data["ERT"].values  # Convert to numpy
        predicted = group_data["ERT_expected"].values  # Convert to numpy

        # Scatter plot with transparency
        ax.scatter(
            observed,
            predicted,
            alpha=0.3,
            s=10,
            color="#1976D2" if not dyslexic else "#D84315",
        )

        # Add diagonal line (perfect prediction)
        max_val = max(observed.max(), predicted.max())
        ax.plot(
            [0, max_val], [0, max_val], "k--", linewidth=2, label="Perfect prediction"
        )

        # Compute R²
        from sklearn.metrics import r2_score

        r2 = r2_score(observed, predicted)

        # Add text with R²
        textstr = f"R² = {r2:.4f}\nn = {len(group_data):,}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=props,
        )

        ax.set_xlabel("Observed ERT (ms)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Predicted ERT (ms)", fontsize=12, fontweight="bold")
        ax.set_title(f"{group_name} Group", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Set equal aspect ratio
        ax.set_aspect("equal", adjustable="box")

    plt.suptitle(
        "Model Fit: Predicted vs Observed Reading Time",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved to {output_path}")
