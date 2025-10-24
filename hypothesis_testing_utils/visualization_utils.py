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

# APA-7 friendly base style: minimal gridlines, consistent typography
sns.set_style("white")
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

    # Ensure the true extrema are inside the range with tiny headroom
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    edge_pad = 0.01 * span
    lo = min(lo, y_min - edge_pad)
    hi = max(hi, y_max + edge_pad)

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
    """Figure 1: Overall Effects Overview with JSON export (APA-7 style, legend fix, tiny-CI visibility)"""
    logger.info("Creating Figure 1: Overall Effects Overview (APA-7)...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # -------------------------------
    # Panel A: Delta ERT (bars + CI)
    # -------------------------------
    features = ["length", "zipf", "surprisal"]
    feat_labels = ["Length", "Zipf Frequency", "Surprisal"]
    x = np.arange(len(features))
    width = 0.36

    ctrl_vals, dys_vals = [], []
    ctrl_yerr, dys_yerr = [], []

    def _center_and_yerr(rec):
        ci_lo = rec.get("ci_low", np.nan)
        ci_hi = rec.get("ci_high", np.nan)
        amie = rec.get("amie_ms", np.nan)
        if np.isfinite(ci_lo) and np.isfinite(ci_hi):
            center = 0.5 * (ci_lo + ci_hi)
            err_low = max(0.0, center - ci_lo)
            err_high = max(0.0, ci_hi - center)
        else:
            center, err_low, err_high = amie, 0.0, 0.0
        return center, (err_low, err_high)

    for feat in features:
        fd = h1_results["features"][feat]
        c = fd["amie_control"]
        d = fd["amie_dyslexic"]

        c_center, (c_lo, c_hi) = _center_and_yerr(c)
        d_center, (d_lo, d_hi) = _center_and_yerr(d)

        ctrl_vals.append(c_center)
        dys_vals.append(d_center)
        ctrl_yerr.append([c_lo, c_hi])
        dys_yerr.append([d_lo, d_hi])

    ctrl_yerr = np.array(ctrl_yerr).T
    dys_yerr = np.array(dys_yerr).T

    # APA grayscale bars with black outlines & error bars
    ax1.bar(
        x - width / 2,
        ctrl_vals,
        width,
        label="Control",
        color="#C7C7C7",
        edgecolor="black",
        linewidth=1.0,
        yerr=ctrl_yerr,
        ecolor="black",
        capsize=4,
        error_kw=dict(linewidth=1.2),
    )
    ax1.bar(
        x + width / 2,
        dys_vals,
        width,
        label="Dyslexic",
        color="#6E6E6E",
        edgecolor="black",
        linewidth=1.0,
        yerr=dys_yerr,
        ecolor="black",
        capsize=4,
        error_kw=dict(linewidth=1.2),
    )

    ax1.set_xlabel("Feature")
    ax1.set_ylabel("Delta ERT (ms, Q1→Q3)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(feat_labels)
    ax1.axhline(0, color="black", linewidth=1)

    # Legend ABOVE the panel to avoid overlap with bars
    ax1.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.6,
        columnspacing=1.4,
    )

    ax1.grid(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # -------------------------------------------------
    # Panel B: Dyslexic amplification (dot + whiskers)
    # -------------------------------------------------
    pathways = ["skip", "duration", "ert"]
    pathway_labels = ["Skip", "Duration", "ERT"]

    sr_rows = []
    for feat in features:
        if feat in h2_results["slope_ratios"]:
            srs = h2_results["slope_ratios"][feat]
            for pw in pathways:
                sr_rows.append(
                    {
                        "feature": feat,
                        "pathway": pw,
                        "sr": srs.get(f"sr_{pw}", np.nan),
                        "ci_low": srs.get(f"sr_{pw}_ci_low", np.nan),
                        "ci_high": srs.get(f"sr_{pw}_ci_high", np.nan),
                    }
                )
    df_sr = pd.DataFrame(sr_rows)

    y_pos = 0.0
    yticks, ylabels = [], []
    points_for_xlim = []

    MIN_VIS_WHISK = 0.01  # SR units; purely for visibility when CI≈0

    for feat, feat_lab in zip(features, feat_labels):
        sub = df_sr[df_sr["feature"] == feat]
        for pw, pw_lab in zip(pathways, pathway_labels):
            row = sub[sub["pathway"] == pw]
            if len(row) == 1:
                r = row.iloc[0]

                if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]):
                    m = 0.5 * (r["ci_low"] + r["ci_high"])
                    xl = max(0.0, m - r["ci_low"])
                    xh = max(0.0, r["ci_high"] - m)
                    if (xl + xh) <= 1e-9:
                        # draw a tiny whisker so ultra-small CIs are still visible
                        xl = xh = MIN_VIS_WHISK
                else:
                    m = r.get("sr", np.nan)
                    xl = xh = MIN_VIS_WHISK if np.isfinite(m) else 0.0

                # errorbar with caps (APA style)
                ax2.errorbar(
                    m,
                    y_pos,
                    xerr=np.array([[xl], [xh]]),
                    fmt="o",
                    color="black",
                    elinewidth=1.5,
                    capsize=3,
                    markersize=5,
                )

                # track for x-limits
                points_for_xlim.extend([m - xl, m + xh])

                yticks.append(y_pos)
                ylabels.append(f"{feat_lab} — {pw_lab}")
                y_pos += 1.0
        y_pos += 0.6  # spacing between feature blocks

    ax2.axvline(1.0, linestyle="--", color="black", linewidth=1)
    ax2.set_xlabel("Slope ratio (Dyslexic / Control), 95% CI")
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ylabels, fontsize=9)

    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    if points_for_xlim:
        xmin = max(0.6, float(np.nanmin(points_for_xlim)) * 0.98)
        xmax = float(np.nanmax(points_for_xlim)) * 1.02
        ax2.set_xlim(xmin, max(1.4, xmax))

    # Panel letters
    fig.text(0.01, 0.98, "A", fontsize=12, fontweight="bold", va="top")
    fig.text(0.51, 0.98, "B", fontsize=12, fontweight="bold", va="top")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # JSON export (unchanged)
    figure_data = {
        "panel_a_bar_chart": {
            "features": feat_labels,
            "control_amies": [float(a) for a in ctrl_vals],
            "dyslexic_amies": [float(a) for a in dys_vals],
        },
        "panel_b_forest_plot": df_sr.to_dict("records"),
    }
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(figure_data, f, indent=2)

    logger.info(f"  Saved: {output_path}")
    logger.info(f"  Saved data: {json_path}")


def create_figure_2_gam_smooths(
    ert_predictor, data: pd.DataFrame, gam_models, output_path: Path
):
    """Figure 2: GAM Smooth Effects (3x3) with JSON export (APA-7 compliant styling)"""
    logger.info("Creating Figure 2: GAM Smooth Effects (3x3, APA-7)...")

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    features = ["length", "zipf", "surprisal"]
    # Sentence-case x-axis labels; Zipf is a proper noun
    feature_labels = ["Word length", "Zipf frequency", "Surprisal"]

    # Styles: grayscale + distinct line styles for print / color-blind safety
    STYLE = {
        "control": {"color": "0.1", "linestyle": "-", "linewidth": 2.5},
        "dyslexic": {"color": "0.55", "linestyle": "--", "linewidth": 2.5},
    }
    CI_FACE = {"control": "0.3", "dyslexic": "0.75"}
    CI_ALPHA = 0.22

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

        for group in ("control", "dyslexic"):
            model = gam_models.skip_model[group]
            p_skip = model.predict_proba(X_grid)

            try:
                ci = model.confidence_intervals(X_grid, width=0.95)
                ci_low, ci_high = ci[:, 0], ci[:, 1]

                # Clip confidence intervals to [0, 1]
                ci_low = np.clip(ci_low, 0.0, 1.0)
                ci_high = np.clip(ci_high, 0.0, 1.0)

                ax.plot(
                    feat_range,
                    p_skip,
                    label=group.capitalize(),
                    **STYLE[group],
                )
                ax.fill_between(
                    feat_range, ci_low, ci_high, color=CI_FACE[group], alpha=CI_ALPHA
                )

                json_data["skip_pathway"][feat][group] = {
                    "predictions": p_skip.tolist(),
                    "ci_low": ci_low.tolist(),
                    "ci_high": ci_high.tolist(),
                }
            except Exception:
                ax.plot(
                    feat_range,
                    p_skip,
                    label=group.capitalize(),
                    **STYLE[group],
                )
                json_data["skip_pathway"][feat][group] = {
                    "predictions": p_skip.tolist(),
                }

        q1, q3 = data[feat].quantile([0.25, 0.75])
        for q in (q1, q3):
            if feat_range[0] <= q <= feat_range[-1]:
                ax.axvline(q, color="0.6", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel(label)
        ax.set_ylabel("P(skip)")
        ax.grid(False)
        # Prevent x-end clipping
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

        for group in ("control", "dyslexic"):
            model = gam_models.duration_model[group]

            y_log = model.predict(X_grid)
            trt = np.exp(y_log) * gam_models.smearing_factors[group]

            try:
                ci = model.confidence_intervals(X_grid, width=0.95)
                ci_low = np.exp(ci[:, 0]) * gam_models.smearing_factors[group]
                ci_high = np.exp(ci[:, 1]) * gam_models.smearing_factors[group]

                # Ensure ordering for fill_between
                ci_low, ci_high = np.minimum(ci_low, ci_high), np.maximum(
                    ci_low, ci_high
                )

                ax.plot(
                    feat_range,
                    trt,
                    label=group.capitalize(),
                    **STYLE[group],
                )
                ax.fill_between(
                    feat_range, ci_low, ci_high, color=CI_FACE[group], alpha=CI_ALPHA
                )

                json_data["duration_pathway"][feat][group] = {
                    "predictions": trt.tolist(),
                    "ci_low": ci_low.tolist(),
                    "ci_high": ci_high.tolist(),
                }
            except Exception:
                ax.plot(
                    feat_range,
                    trt,
                    label=group.capitalize(),
                    **STYLE[group],
                )
                json_data["duration_pathway"][feat][group] = {
                    "predictions": trt.tolist(),
                }

        q1, q3 = data[feat].quantile([0.25, 0.75])
        for q in (q1, q3):
            if feat_range[0] <= q <= feat_range[-1]:
                ax.axvline(q, color="0.6", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel(label)
        ax.set_ylabel("TRT (ms)")
        ax.grid(False)
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

        for group in ("control", "dyslexic"):
            ert = ert_predictor.predict_ert(grid, group)

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
                # Conservative extremes
                ert_low = (1 - p_skip_high) * trt_low
                ert_high = (1 - p_skip_low) * trt_high

                ax.plot(
                    feat_range,
                    ert,
                    label=group.capitalize(),
                    **STYLE[group],
                )
                ax.fill_between(
                    feat_range, ert_low, ert_high, color=CI_FACE[group], alpha=CI_ALPHA
                )

                json_data["ert_pathway"][feat][group] = {
                    "predictions": ert.tolist(),
                    "ci_low": ert_low.tolist(),
                    "ci_high": ert_high.tolist(),
                }
            except Exception:
                ax.plot(
                    feat_range,
                    ert,
                    label=group.capitalize(),
                    **STYLE[group],
                )
                json_data["ert_pathway"][feat][group] = {
                    "predictions": ert.tolist(),
                }

        q1, q3 = data[feat].quantile([0.25, 0.75])
        for q in (q1, q3):
            if feat_range[0] <= q <= feat_range[-1]:
                ax.axvline(q, color="0.6", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel(label)
        ax.set_ylabel("ERT (ms)")
        ax.grid(False)
        ax.set_xlim(feat_range[0], feat_range[-1])
        ax.margins(x=0)

    set_row_ylim(axes[2, :])

    # Add panel letters (A–I) in the top-left of each subplot
    letters = list("ABCDEFGHI")
    for ax, L in zip(axes.flatten(), letters):
        ax.text(
            0.02,
            0.98,
            L,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    # Shared legend (single, for the whole figure)
    legend_lines = [
        plt.Line2D([0], [0], **STYLE["control"], label="Control"),
        plt.Line2D([0], [0], **STYLE["dyslexic"], label="Dyslexic"),
    ]
    # Make some space at the bottom for the legend
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.legend(
        handles=legend_lines,
        labels=["Control", "Dyslexic"],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        handlelength=2.2,
        columnspacing=1.6,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # === EXPORT JSON DATA ===
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"  Saved: {output_path}")
    logger.info(f"  Saved data: {json_path}")


def create_figure_3_gap_decomposition(h3_results: Dict, output_path: Path):
    """
    Figure 3 (APA-7): Dyslexic–Control ERT gap decomposition (ms/word), 3 panels:
      A) Observed vs equal-ease gap (95% CI)
      B) Shapley gap contributions: Skip vs Duration (95% CI)
      C) Equal-ease feature contributions: Length, Zipf, Surprisal (95% CI)

    Exports a single PNG and a .json sidecar of the plotted values.
    """
    logger.info("Creating Figure 3 (APA-7): Gap Decomposition")

    # ---------- APA-7 visual defaults ----------
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    # neutral grayscale palette
    GRAY_LIGHT = (0.85, 0.85, 0.85)
    GRAY_MED = (0.65, 0.65, 0.65)
    GRAY_DARK = (0.40, 0.40, 0.40)
    EDGE_COLOR = "black"

    def _apa_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.tick_params(axis="both", which="both", length=3, width=0.8)
        ax.tick_params(top=False, right=False)
        ax.set_ylim(bottom=0)  # zero-baseline
        return ax

    def _safe_center_yerr(mean_val, ci_low, ci_high):
        # guard odd CI ordering or missing CI
        if np.isfinite(ci_low) and np.isfinite(ci_high):
            lo, hi = (ci_low, ci_high) if ci_low <= ci_high else (ci_high, ci_low)
            c = 0.5 * (lo + hi)
            return c, (max(0.0, c - lo), max(0.0, hi - c))
        return mean_val, (0.0, 0.0)

    # ---------- Pull stats from H3 outputs ----------
    g0_stats = h3_results.get("canonical_gap_stats", {})
    g0_mean = float(g0_stats.get("mean", h3_results.get("canonical_gap_G0", np.nan)))
    g0_ci_low = float(g0_stats.get("ci_low", np.nan))
    g0_ci_high = float(g0_stats.get("ci_high", np.nan))

    ee = h3_results.get("equal_ease_counterfactual", {})
    cf_stats = ee.get("counterfactual_gap_stats", {})
    cf_mean = float(cf_stats.get("mean", ee.get("counterfactual_gap", np.nan)))
    cf_ci_low = float(cf_stats.get("ci_low", np.nan))
    cf_ci_high = float(cf_stats.get("ci_high", np.nan))

    sh = h3_results.get("shapley_decomposition", {})
    skip_s = sh.get("skip_contribution_stats", {})
    dur_s = sh.get("duration_contribution_stats", {})

    fc = h3_results.get("equal_ease_feature_contributions", {})
    fstats = fc.get("feature_contributions_stats", {})

    # ---------- Layout ----------
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.25))
    # tighter margins, balanced inter-panel spacing
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.19, top=0.84, wspace=0.72)
    axA, axB, axC = axes

    # ============== Panel A: Observed vs Equal-ease ==================
    A_labels = ["Observed\ngap", "Equal-ease\ngap"]
    A_vals, A_errs = [], []

    c, (lo, hi) = _safe_center_yerr(g0_mean, g0_ci_low, g0_ci_high)
    A_vals.append(c)
    A_errs.append([lo, hi])

    c, (lo, hi) = _safe_center_yerr(cf_mean, cf_ci_low, cf_ci_high)
    A_vals.append(c)
    A_errs.append([lo, hi])

    A_errs = np.array(A_errs, float).T
    axA = _apa_axes(axA)

    xA = np.array([0.0, 1.1])
    axA.bar(
        xA,
        A_vals,
        width=0.66,
        color=[GRAY_LIGHT, GRAY_DARK],
        edgecolor=EDGE_COLOR,
        linewidth=1.0,
        yerr=A_errs,
        ecolor="black",
        capsize=3,
        error_kw=dict(linewidth=1.0),
    )
    axA.set_xticks(xA, A_labels)
    axA.set_xlim(xA[0] - 0.55, xA[-1] + 0.55)
    axA.set_ylabel("ERT gap (ms/word)", labelpad=4)

    A_max = np.nanmax(np.array(A_vals) + A_errs[1])
    axA.set_ylim(top=A_max * 1.18)

    # panel letter
    axA.text(
        -0.18,
        1.03,
        "A",
        transform=axA.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )

    # ============== Panel B: Shapley skip vs duration =================
    def _center_err_from(stats):
        m = float(stats.get("mean", np.nan))
        l = float(stats.get("ci_low", np.nan))
        h = float(stats.get("ci_high", np.nan))
        return _safe_center_yerr(m, l, h)

    skip_c, (skip_lo, skip_hi) = _center_err_from(skip_s)
    dur_c, (dur_lo, dur_hi) = _center_err_from(dur_s)

    B_vals = [skip_c, dur_c]
    B_errs = np.array([[skip_lo, dur_lo], [skip_hi, dur_hi]], float)

    axB = _apa_axes(axB)

    xB = np.array([0.0, 1.1])
    axB.bar(
        xB,
        B_vals,
        width=0.66,
        color=[GRAY_LIGHT, GRAY_DARK],
        edgecolor=EDGE_COLOR,
        linewidth=1.0,
        yerr=B_errs,
        ecolor="black",
        capsize=3,
        error_kw=dict(linewidth=1.0),
    )
    axB.set_xticks(xB, ["Skip", "Duration"])
    axB.set_xlim(xB[0] - 0.55, xB[-1] + 0.55)
    axB.set_ylabel("Contribution to gap (ms/word)", labelpad=4)

    B_max = np.nanmax(np.array(B_vals) + B_errs[1])
    axB.set_ylim(top=B_max * 1.18)

    axB.text(
        -0.18,
        1.03,
        "B",
        transform=axB.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )

    # ============== Panel C: Equal-ease feature contributions =========
    feats = ["length", "zipf", "surprisal"]
    feat_labels = ["Length", "Zipf\nfrequency", "Surprisal"]

    C_vals, C_errlist = [], []
    for key in feats:
        st = fstats.get(key, {})
        m = float(st.get("mean", np.nan))
        lo = float(st.get("ci_low", np.nan))
        hi = float(st.get("ci_high", np.nan))
        c, (e_lo, e_hi) = _safe_center_yerr(m, lo, hi)
        C_vals.append(c)
        C_errlist.append([e_lo, e_hi])

    C_errs = np.array(C_errlist, float).T

    axC = _apa_axes(axC)

    xC = np.array([0.0, 1.6, 3.2])
    axC.bar(
        xC,
        C_vals,
        width=0.72,
        color=[GRAY_LIGHT, GRAY_MED, GRAY_DARK],
        edgecolor=EDGE_COLOR,
        linewidth=1.0,
        yerr=C_errs,
        ecolor="black",
        capsize=3,
        error_kw=dict(linewidth=1.0),
    )
    axC.set_xticks(xC, feat_labels)
    axC.set_xlim(xC[0] - 0.60, xC[-1] + 0.60)
    axC.set_ylabel("Gap reduction (ms/word)", labelpad=4)

    C_max = np.nanmax(np.array(C_vals) + C_errs[1])
    axC.set_ylim(top=max(1e-6, C_max) * 1.20)

    axC.text(
        -0.18,
        1.03,
        "C",
        transform=axC.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )

    # ---------- Save (PNG only) ----------
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    # ---------- JSON sidecar of plotted values ----------
    out = {
        "panel_a": {
            "observed_gap": {
                "mean": g0_mean,
                "ci_low": g0_ci_low,
                "ci_high": g0_ci_high,
            },
            "equal_ease_gap": {
                "mean": cf_mean,
                "ci_low": cf_ci_low,
                "ci_high": cf_ci_high,
            },
        },
        "panel_b_shapley": {
            "skip": {
                "mean": float(skip_s.get("mean", np.nan)),
                "ci_low": float(skip_s.get("ci_low", np.nan)),
                "ci_high": float(skip_s.get("ci_high", np.nan)),
            },
            "duration": {
                "mean": float(dur_s.get("mean", np.nan)),
                "ci_low": float(dur_s.get("ci_low", np.nan)),
                "ci_high": float(dur_s.get("ci_high", np.nan)),
            },
            "g0_mean": g0_mean,
        },
        "panel_c_feature_contribs": fstats,
    }
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(out, f, indent=2)

    logger.info(f"  Saved: {output_path}")
    logger.info(f"  Saved data: {output_path.with_suffix('.json')}")


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

    # Figure 2: GAM Smooths (APA-7 formatted)
    create_figure_2_gam_smooths(
        ert_predictor,
        data,
        gam_models,
        output_dir / "figure_2_gam_smooths.png",
    )

    # Figure 3: Gap Decomposition (APA-7)
    create_figure_3_gap_decomposition(
        h3_results,
        output_dir / "figure_3_gap_decomposition.png",
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
