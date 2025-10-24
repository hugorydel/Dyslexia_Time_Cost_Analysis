"""
Visualization Utilities
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

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.titlesize"] = 14


def set_row_ylim(axes_row):
    """
    Set consistent, snug y-limits across all axes in a row.

    - Ignores vertical lines (e.g., Q1/Q3 axvline markers)
    - Uses robust 1–99% quantiles with a small pad
    - Ensures true extrema are included with a tiny headroom to avoid clipping
    - Clamps probability rows to [0, 1] and enforces minimum visual spans
    """
    chunks = []

    for ax in axes_row:
        # Lines (skip verticals)
        for line in ax.lines:
            try:
                xd = np.asarray(line.get_xdata(orig=False), dtype=float)
                yd = np.asarray(line.get_ydata(orig=False), dtype=float)
            except Exception:
                continue
            if xd.size == 0 or yd.size == 0:
                continue
            # vertical line => all x nearly equal -> ignore
            if np.isfinite(xd).sum() <= 1 or (np.nanmax(xd) - np.nanmin(xd)) < 1e-9:
                continue
            yd = yd[np.isfinite(yd)]
            if yd.size:
                chunks.append(yd)

        # Confidence bands / fill_between (PolyCollections)
        for coll in getattr(ax, "collections", []):
            try:
                paths = coll.get_paths()
                if not paths:
                    continue
                ys = np.concatenate([p.vertices[:, 1] for p in paths])
                ys = ys[np.isfinite(ys)]
                if ys.size:
                    chunks.append(ys)
            except Exception:
                pass

    if not chunks:
        return

    y = np.concatenate(chunks)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return

    # Robust central range
    q_lo, q_hi = np.percentile(y, [1.0, 99.0])
    if not np.isfinite(q_lo) or not np.isfinite(q_hi) or q_hi <= q_lo:
        return

    span = max(q_hi - q_lo, 1e-9)
    pad = 0.06 * span
    lo = q_lo - pad
    hi = q_hi + pad

    # ---- NEW: make sure the true extrema are inside the range with tiny headroom ----
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    edge_pad = 0.01 * span  # just a hair to prevent touching the frame
    lo = min(lo, y_min - edge_pad)
    hi = max(hi, y_max + edge_pad)
    # -------------------------------------------------------------------------------

    # Probability-like vs ms rows
    is_prob_like = (np.nanmin(y) >= -0.05) and (np.nanmax(y) <= 1.05)
    if is_prob_like:
        lo = max(0.0, lo)
        hi = min(1.0, hi)
        if hi - lo < 0.10:
            mid = 0.5 * (hi + lo)
            lo = max(0.0, mid - 0.05)
            hi = min(1.0, mid + 0.05)
    else:
        lo = max(0.0, lo)
        if hi - lo < 40.0:
            mid = 0.5 * (hi + lo)
            lo = max(0.0, mid - 20.0)
            hi = mid + 20.0

    for ax in axes_row:
        ax.set_ylim(float(lo), float(hi))


def create_figure_1_overall_effects(
    h1_results: Dict, h2_results: Dict, output_path: Path
):
    """Figure 1: Overall Effects Overview with JSON export"""
    logger.info("Creating Figure 1: Overall Effects Overview...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # === Panel A: Delta ERT ===
    features = ["length", "zipf", "surprisal"]
    x = np.arange(len(features))
    width = 0.35

    ctrl_amies, dys_amies = [], []
    ctrl_err_lo, ctrl_err_hi, dys_err_lo, dys_err_hi = [], [], [], []

    for feat in features:
        fd = h1_results["features"][feat]
        c = fd["amie_control"]
        d = fd["amie_dyslexic"]

        ctrl_amies.append(c.get("amie_ms", np.nan))
        dys_amies.append(d.get("amie_ms", np.nan))

        # asymmetric yerr around the point estimate
        ctrl_err_lo.append(max(0.0, c["amie_ms"] - c["ci_low"]))
        ctrl_err_hi.append(max(0.0, c["ci_high"] - c["amie_ms"]))
        dys_err_lo.append(max(0.0, d["amie_ms"] - d["ci_low"]))
        dys_err_hi.append(max(0.0, d["ci_high"] - d["amie_ms"]))

    ctrl_yerr = np.vstack([ctrl_err_lo, ctrl_err_hi])
    dys_yerr = np.vstack([dys_err_lo, dys_err_hi])

    ax1.bar(
        x - width / 2,
        ctrl_amies,
        width,
        label="Control",
        color="steelblue",
        alpha=0.85,
        yerr=ctrl_yerr,
        capsize=4,
        ecolor="black",
        error_kw=dict(linewidth=1),
    )
    ax1.bar(
        x + width / 2,
        dys_amies,
        width,
        label="Dyslexic",
        color="coral",
        alpha=0.85,
        yerr=dys_yerr,
        capsize=4,
        ecolor="black",
        error_kw=dict(linewidth=1),
    )

    ax1.set_xlabel("Feature")
    ax1.set_ylabel("Delta ERT (ms, Q1->Q3)")
    ax1.set_title("Feature Effects on Expected Reading Time")
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
    ax2.set_title("Dyslexic Amplification by Pathway")
    ax2.set_xlim(0, max(3, df_sr["sr"].max() * 1.1))
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # === EXPORT JSON DATA ===
    figure_data = {
        "panel_a_bar_chart": {
            "features": ["Length", "Zipf Frequency", "Surprisal"],
            "control_amies": [float(a) for a in ctrl_amies],
            "dyslexic_amies": [float(a) for a in dys_amies],
        },
        "panel_b_forest_plot": df_sr.to_dict("records"),
    }

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(figure_data, f, indent=2)

    logger.info(f"  Saved: {output_path}")
    logger.info(f"  Saved data: {json_path}")


def create_figure_s1_gam_smooths(
    ert_predictor, data: pd.DataFrame, gam_models, output_path: Path
):
    """Figure S1: GAM Smooth Effects (3x3) with JSON export"""
    logger.info("Creating Figure S1: GAM Smooth Effects (3x3 with confidence bands)...")

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    features = ["length", "zipf", "surprisal"]
    feature_labels = ["Word Length", "Zipf Frequency", "Surprisal"]

    # Data for JSON export
    json_data = {
        "skip_pathway": {},
        "duration_pathway": {},
        "ert_pathway": {},
    }

    # === Row 1: Skip pathway ===
    for i, (feat, label) in enumerate(zip(features, feature_labels)):
        ax = axes[0, i]

        feat_range = np.linspace(
            data[feat].quantile(0.01), data[feat].quantile(0.99), 100
        )

        other_feats = [f for f in features if f != feat]
        means = {f: data[f].mean() for f in other_feats}

        grid = pd.DataFrame({feat: feat_range, **means})
        X_grid = grid[["length", "zipf", "surprisal"]].values

        json_data["skip_pathway"][feat] = {"feature_values": feat_range.tolist()}

        for group, color in [("control", "steelblue"), ("dyslexic", "coral")]:
            model = gam_models.skip_model[group]
            p_skip = model.predict_proba(X_grid)

            try:
                ci = model.confidence_intervals(X_grid, width=0.95)
                ci_low, ci_high = ci[:, 0], ci[:, 1]

                # Clip confidence intervals to fit within bounds.
                ci_low = np.clip(ci_low, 0.0, 1.0)
                ci_high = np.clip(ci_high, 0.0, 1.0)

                ax.plot(
                    feat_range,
                    p_skip,
                    label=group.capitalize(),
                    linewidth=2.5,
                    color=color,
                )
                ax.fill_between(feat_range, ci_low, ci_high, color=color, alpha=0.2)

                json_data["skip_pathway"][feat][group] = {
                    "predictions": p_skip.tolist(),
                    "ci_low": ci_low.tolist(),
                    "ci_high": ci_high.tolist(),
                }
            except:
                ax.plot(
                    feat_range,
                    p_skip,
                    label=group.capitalize(),
                    linewidth=2.5,
                    color=color,
                )
                json_data["skip_pathway"][feat][group] = {
                    "predictions": p_skip.tolist(),
                }

        q1, q3 = data[feat].quantile([0.25, 0.75])
        ax.axvline(q1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(q3, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel(label)
        ax.set_ylabel("P(skip)")
        if i == 0:
            ax.set_title("Skip Pathway", fontweight="bold", fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

        # --- Remove x-axis gaps and prevent clipping at ends ---
        ax.set_xlim(feat_range[0], feat_range[-1])
        ax.margins(x=0)

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

        json_data["duration_pathway"][feat] = {"feature_values": feat_range.tolist()}

        for group, color in [("control", "steelblue"), ("dyslexic", "coral")]:
            model = gam_models.duration_model[group]

            y_log = model.predict(X_grid)
            trt = np.exp(y_log) * gam_models.smearing_factors[group]

            try:
                ci = model.confidence_intervals(X_grid, width=0.95)
                ci_low = np.exp(ci[:, 0]) * gam_models.smearing_factors[group]
                ci_high = np.exp(ci[:, 1]) * gam_models.smearing_factors[group]

                # Clip confidence intervals to fit within bounds.
                ci_low, ci_high = np.minimum(ci_low, ci_high), np.maximum(
                    ci_low, ci_high
                )

                ax.plot(
                    feat_range,
                    trt,
                    label=group.capitalize(),
                    linewidth=2.5,
                    color=color,
                )
                ax.fill_between(feat_range, ci_low, ci_high, color=color, alpha=0.2)

                json_data["duration_pathway"][feat][group] = {
                    "predictions": trt.tolist(),
                    "ci_low": ci_low.tolist(),
                    "ci_high": ci_high.tolist(),
                }
            except:
                ax.plot(
                    feat_range,
                    trt,
                    label=group.capitalize(),
                    linewidth=2.5,
                    color=color,
                )
                json_data["duration_pathway"][feat][group] = {
                    "predictions": trt.tolist(),
                }

        q1, q3 = data[feat].quantile([0.25, 0.75])
        ax.axvline(q1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(q3, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel(label)
        ax.set_ylabel("TRT | fixated (ms)")
        if i == 0:
            ax.set_title("Duration Pathway", fontweight="bold", fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

        # --- Remove x-axis gaps and prevent clipping at ends ---
        ax.set_xlim(feat_range[0], feat_range[-1])
        ax.margins(x=0)

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

        json_data["ert_pathway"][feat] = {"feature_values": feat_range.tolist()}

        for group, color in [("control", "steelblue"), ("dyslexic", "coral")]:
            ert = ert_predictor.predict_ert(grid, group)

            # Calculate confidence intervals by propagating uncertainty
            try:
                # Get skip probability and CI
                skip_model = gam_models.skip_model[group]
                X_grid = grid[["length", "zipf", "surprisal"]].values
                p_skip = skip_model.predict_proba(X_grid)
                skip_ci = skip_model.confidence_intervals(X_grid, width=0.95)
                p_skip_low = np.clip(skip_ci[:, 0], 0.0, 1.0)
                p_skip_high = np.clip(skip_ci[:, 1], 0.0, 1.0)

                # Get duration and CI
                dur_model = gam_models.duration_model[group]
                y_log = dur_model.predict(X_grid)
                trt = np.exp(y_log) * gam_models.smearing_factors[group]
                dur_ci = dur_model.confidence_intervals(X_grid, width=0.95)
                trt_low = np.exp(dur_ci[:, 0]) * gam_models.smearing_factors[group]
                trt_high = np.exp(dur_ci[:, 1]) * gam_models.smearing_factors[group]

                # Propagate uncertainty: ERT = (1 - p_skip) * trt
                # Use conservative approach: combine extremes
                ert_low = (1 - p_skip_high) * trt_low
                ert_high = (1 - p_skip_low) * trt_high

                ax.plot(
                    feat_range,
                    ert,
                    label=group.capitalize(),
                    linewidth=2.5,
                    color=color,
                )
                ax.fill_between(feat_range, ert_low, ert_high, color=color, alpha=0.2)

                json_data["ert_pathway"][feat][group] = {
                    "predictions": ert.tolist(),
                    "ci_low": ert_low.tolist(),
                    "ci_high": ert_high.tolist(),
                }
            except:
                ax.plot(
                    feat_range,
                    ert,
                    label=group.capitalize(),
                    linewidth=2.5,
                    color=color,
                )

                json_data["ert_pathway"][feat][group] = {
                    "predictions": ert.tolist(),
                }

        q1, q3 = data[feat].quantile([0.25, 0.75])
        ax.axvline(q1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(q3, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel(label)
        ax.set_ylabel("ERT (ms)")
        if i == 0:
            ax.set_title("Combined ERT", fontweight="bold", fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

        # --- Remove x-axis gaps and prevent clipping at ends ---
        ax.set_xlim(feat_range[0], feat_range[-1])
        ax.margins(x=0)

    set_row_ylim(axes[2, :])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # === EXPORT JSON DATA ===
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"  Saved: {output_path}")
    logger.info(f"  Saved data: {json_path}")


def export_model_stats_json(
    gam_models, skip_metadata: Dict, duration_metadata: Dict, output_path: Path
):
    """Export model statistics to JSON"""
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
    Merged supplementary figures into main function
    """
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING ALL FIGURES")
    logger.info("=" * 60)

    output_dir.mkdir(exist_ok=True, parents=True)

    # === MAIN FIGURES ===
    logger.info("\nMain figures:")

    # Figure 1: Overall Effects
    create_figure_1_overall_effects(
        h1_results, h2_results, output_dir / "figure_1_overall_effects.png"
    )

    # Figure 2: GAM Smooths
    create_figure_s1_gam_smooths(
        ert_predictor,
        data,
        gam_models,
        output_dir / "figure_2_gam_smooths.png",
    )

    # Model Statistics (JSON)
    export_model_stats_json(
        gam_models,
        skip_metadata,
        duration_metadata,
        output_dir / "model_statistics.json",
    )

    logger.info("\nAll figures generated successfully!")
    logger.info(f"Figures saved to: {output_dir}")
    logger.info("  Each figure has accompanying .json data file")
