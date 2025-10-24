"""
Hypothesis 3: Gap Decomposition
Single source of truth (S & G0), consistent equal-ease policy
"""

import itertools
import logging
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from hypothesis_testing_utils.stats_utils import compute_two_tailed_pvalue_corrected

logger = logging.getLogger(__name__)

ANALYSIS_SEED = 42
BASELINE_TOLERANCE = 1e-6


def predict_components_batch(predictor, df, group):
    """
    Batch helper for predictions with components.
    Returns 1D arrays of ert, p_skip, trt.
    """
    X = df[["length", "zipf", "surprisal"]]
    ert, p_skip, trt = predictor.predict_ert(X, group, return_components=True)
    return np.asarray(ert).ravel(), np.asarray(p_skip).ravel(), np.asarray(trt).ravel()


def predict_gap_on_sample(ert_predictor, sample: pd.DataFrame) -> float:
    """
    Compute gap using model predictions on the given sample.
    Subject-balanced: mean of subject means within group (reduces variance).
    """
    group_means = {}
    for group_name in ["control", "dyslexic"]:
        gdf = sample[sample["group"] == group_name]
        if len(gdf) == 0:
            continue
        X = gdf[["length", "zipf", "surprisal"]]
        try:
            ert = np.asarray(ert_predictor.predict_ert(X, group_name)).ravel()
        except Exception:
            continue
        subj_mean = (
            pd.DataFrame({"subject_id": gdf["subject_id"].values, "ert": ert})
            .groupby("subject_id")["ert"]
            .mean()
        )
        group_means[group_name] = float(subj_mean.mean())

    if "control" in group_means and "dyslexic" in group_means:
        return group_means["dyslexic"] - group_means["control"]
    return np.nan


def bootstrap_gap_component(
    observed_value: float,
    data: pd.DataFrame,
    computation_fn,
    n_bootstrap: int = 2000,
    **kwargs,
) -> Dict:
    """Generic bootstrap with subject resampling (stratified by group)."""
    ctrl_subjects = data.loc[data["group"] == "control", "subject_id"].unique()
    dys_subjects = data.loc[data["group"] == "dyslexic", "subject_id"].unique()

    bootstrap_values = []

    print(f"      Bootstrap (n={n_bootstrap})...", flush=True)

    for i in tqdm(range(n_bootstrap), desc="      Bootstrap", leave=False):
        rng = np.random.RandomState(3000 + i)
        boot_ctrl = rng.choice(ctrl_subjects, size=len(ctrl_subjects), replace=True)
        boot_dys = rng.choice(dys_subjects, size=len(dys_subjects), replace=True)

        parts = []
        for j, s in enumerate(boot_ctrl):
            df = data[data["subject_id"] == s].copy()
            df["subject_id"] = f"ctrl_{s}_{j}"
            parts.append(df)
        for j, s in enumerate(boot_dys):
            df = data[data["subject_id"] == s].copy()
            df["subject_id"] = f"dys_{s}_{j}"
            parts.append(df)

        boot_data = pd.concat(parts, ignore_index=True)

        try:
            result = computation_fn(boot_data, **kwargs)
            if np.isfinite(result):
                bootstrap_values.append(result)
        except Exception:
            pass

    if len(bootstrap_values) > 0:
        arr = np.array(bootstrap_values, dtype=float)
        ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
        p_value = compute_two_tailed_pvalue_corrected(observed_value, arr)
    else:
        ci_low = ci_high = p_value = np.nan

    return {
        "mean": float(observed_value),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "n_bootstrap": int(n_bootstrap),
        "n_effective_bootstrap": int(len(bootstrap_values)),
    }


def _apply_equal_ease_subset(df, subset, quartiles, bin_edges, q3_zipf_by_bin):
    out = df.copy()
    if "length" in subset:
        out["length"] = out["length"].clip(upper=quartiles["length"]["q1"])
    if "surprisal" in subset:
        out["surprisal"] = out["surprisal"].clip(upper=quartiles["surprisal"]["q1"])
    if "zipf" in subset:
        out["zipf"] = _clamp_zipf_conditional(out, bin_edges, q3_zipf_by_bin)
    return out


def shapley_decomposition(
    ert_predictor, S: pd.DataFrame, G0: float, n_bootstrap: int = 2000
) -> Dict:
    """
    Shapley decomposition - skip vs duration contributions.
    Uses the SAME sample S and baseline G0 as rest of H3.
    """
    logger.info("  Computing Shapley decomposition...")
    print("  Shapley decomposition...", flush=True)

    dys_data = S[S["group"] == "dyslexic"]
    if len(dys_data) == 0:
        return {"error": "No dyslexic data"}

    # Batched predictions (point estimates)
    ert_dys, p_skip_dys, trt_dys = predict_components_batch(
        ert_predictor, dys_data, "dyslexic"
    )
    ert_ctrl, p_skip_ctrl, trt_ctrl = predict_components_batch(
        ert_predictor, dys_data, "control"
    )

    mean_ert_dys = ert_dys.mean()
    mean_ert_ctrl = ert_ctrl.mean()
    computed_gap = mean_ert_dys - mean_ert_ctrl

    # Path 1: equalize skip first
    ert_1 = (1 - p_skip_ctrl) * trt_dys
    delta_skip_path1 = mean_ert_dys - ert_1.mean()
    delta_duration_path1 = ert_1.mean() - mean_ert_ctrl

    # Path 2: equalize duration first
    ert_3 = (1 - p_skip_dys) * trt_ctrl
    delta_duration_path2 = mean_ert_dys - ert_3.mean()
    delta_skip_path2 = ert_3.mean() - mean_ert_ctrl

    # Average Shapley values
    delta_skip = (delta_skip_path1 + delta_skip_path2) / 2
    delta_duration = (delta_duration_path1 + delta_duration_path2) / 2

    logger.info(
        f"    Computed gap (from predictions on dys subset): {computed_gap:.2f} ms"
    )
    logger.info(f"    Canonical gap G0 (single source of truth): {G0:.2f} ms")

    # Anchor to G0
    if abs(computed_gap) > 1e-9 and abs(computed_gap - G0) > BASELINE_TOLERANCE:
        scale = G0 / computed_gap
        logger.info(f"    Applying scale factor: {scale:.4f}")
    else:
        scale = 1.0

    # ---------- Single-pass bootstrap for BOTH Shapley components ----------
    print("    Shapley bootstrap (single pass)...", flush=True)
    ctrl_subjects = S.loc[S["group"] == "control", "subject_id"].unique()
    dys_subjects = S.loc[S["group"] == "dyslexic", "subject_id"].unique()

    skip_samples, dur_samples = [], []

    for i in tqdm(range(n_bootstrap), desc="      Bootstrap", leave=False):
        rng = np.random.RandomState(3000 + i)
        boot_ctrl = rng.choice(ctrl_subjects, size=len(ctrl_subjects), replace=True)
        boot_dys = rng.choice(dys_subjects, size=len(dys_subjects), replace=True)

        parts = []
        for j, s in enumerate(boot_ctrl):
            df = S[S["subject_id"] == s].copy()
            df["subject_id"] = f"ctrl_{s}_{j}"
            parts.append(df)
        for j, s in enumerate(boot_dys):
            df = S[S["subject_id"] == s].copy()
            df["subject_id"] = f"dys_{s}_{j}"
            parts.append(df)
        boot = pd.concat(parts, ignore_index=True)

        dys_boot = boot[boot["group"] == "dyslexic"]
        if len(dys_boot) < 10:
            continue

        ed, psd, td = predict_components_batch(ert_predictor, dys_boot, "dyslexic")
        ec, psc, tc = predict_components_batch(ert_predictor, dys_boot, "control")
        mean_dys_b = ed.mean()
        mean_ctrl_b = ec.mean()

        # Path 1 / Path 2 on the boot sample
        ert_1_b = (1 - psc) * td  # skip equalized first
        ert_3_b = (1 - psd) * tc  # duration equalized first

        # Skip contribution (avg of paths)
        skip_1 = mean_dys_b - ert_1_b.mean()
        skip_2 = ert_3_b.mean() - mean_ctrl_b
        skip_b = 0.5 * (skip_1 + skip_2)

        # Duration contribution (avg of paths)
        dur_1 = mean_dys_b - ert_3_b.mean()
        dur_2 = ert_1_b.mean() - mean_ctrl_b
        dur_b = 0.5 * (dur_1 + dur_2)

        skip_samples.append(skip_b)
        dur_samples.append(dur_b)

    skip_arr = np.array(skip_samples, dtype=float)
    dur_arr = np.array(dur_samples, dtype=float)

    # Two-tailed p-values (+1 correction inside util). p-values are
    # computed on the unscaled distribution; scaling doesn't affect them.
    if skip_arr.size > 0:
        skip_ci_low, skip_ci_high = np.percentile(skip_arr, [2.5, 97.5])
        skip_p = compute_two_tailed_pvalue_corrected(delta_skip, skip_arr)
    else:
        skip_ci_low = skip_ci_high = skip_p = np.nan

    if dur_arr.size > 0:
        dur_ci_low, dur_ci_high = np.percentile(dur_arr, [2.5, 97.5])
        dur_p = compute_two_tailed_pvalue_corrected(delta_duration, dur_arr)
    else:
        dur_ci_low = dur_ci_high = dur_p = np.nan

    # Scale means/CIs to G0
    stats_skip_scaled = {
        "mean": float(delta_skip * scale),
        "ci_low": float(skip_ci_low * scale),
        "ci_high": float(skip_ci_high * scale),
        "p_value": float(skip_p),
        "n_bootstrap": int(skip_arr.size),
    }
    stats_duration_scaled = {
        "mean": float(delta_duration * scale),
        "ci_low": float(dur_ci_low * scale),
        "ci_high": float(dur_ci_high * scale),
        "p_value": float(dur_p),
        "n_bootstrap": int(dur_arr.size),
    }

    skip_contribution_ms = stats_skip_scaled["mean"]
    duration_contribution_ms = stats_duration_scaled["mean"]

    pct_skip = (skip_contribution_ms / G0 * 100) if G0 != 0 else 0
    pct_duration = (duration_contribution_ms / G0 * 100) if G0 != 0 else 0

    logger.info(
        f"    Skip contribution: {skip_contribution_ms:.2f} ms ({pct_skip:.1f}%)"
    )
    logger.info(
        f"    Duration contribution: {duration_contribution_ms:.2f} ms ({pct_duration:.1f}%)"
    )
    logger.info(f"    Sum check: {pct_skip + pct_duration:.1f}% (should be 100%)")

    assert abs((skip_contribution_ms + duration_contribution_ms) - G0) < max(
        0.1 * abs(G0), 1.0
    ), "Skip + Duration must sum to G0"

    return {
        "skip_contribution": float(skip_contribution_ms),
        "skip_contribution_stats": stats_skip_scaled,
        "duration_contribution": float(duration_contribution_ms),
        "duration_contribution_stats": stats_duration_scaled,
        "skip_pct": float(pct_skip),
        "duration_pct": float(pct_duration),
    }


# ---------------------- NEW HELPERS (conditional Zipf policy) ----------------------


def _assign_length_bin(df: pd.DataFrame, bin_edges: np.ndarray) -> pd.Series:
    return pd.cut(df["length"], bins=bin_edges, labels=False, include_lowest=True)


def _zipf_q3_by_length_bin(S: pd.DataFrame, bin_edges: np.ndarray) -> pd.Series:
    bins_S = _assign_length_bin(S, bin_edges)
    tmp = S.copy()
    tmp["length_bin"] = bins_S
    return tmp.groupby("length_bin")["zipf"].quantile(0.75)


def _clamp_zipf_conditional(
    df: pd.DataFrame, bin_edges: np.ndarray, q3_by_bin: pd.Series
) -> pd.Series:
    bins_df = _assign_length_bin(df, bin_edges)
    thr = bins_df.map(q3_by_bin)
    z = df["zipf"].values
    t = thr.values.astype(float)
    t = np.where(np.isnan(t), z, t)  # no-op if bin unseen
    return pd.Series(np.maximum(z, t), index=df.index)


# -------------------------------------------------------------------------------


def equal_ease_counterfactual(
    ert_predictor,
    S: pd.DataFrame,
    quartiles: Dict,
    G0: float,
    n_bootstrap: int = 2000,
    bin_edges: np.ndarray = None,
) -> Dict:
    """
    Equal-ease counterfactual using Q1/Q3 clamp policy.
    Uses the SAME sample S and baseline G0.
    Zipf clamp is conditional on length: per-bin Q3 computed once on S.
    """
    logger.info("  Computing equal-ease counterfactual...")
    print("  Equal-ease counterfactual...", flush=True)

    Q1_len = quartiles["length"]["q1"]
    Q1_surpr = quartiles["surprisal"]["q1"]

    if bin_edges is None:
        raise ValueError("bin_edges must be provided for conditional Zipf equal-ease.")
    q3_zipf_by_bin = _zipf_q3_by_length_bin(S, bin_edges)

    # Point estimate on S
    baseline_gap = predict_gap_on_sample(ert_predictor, S)
    if abs(baseline_gap - G0) > BASELINE_TOLERANCE:
        logger.warning(f"    Baseline gap {baseline_gap:.2f} differs from G0 {G0:.2f}")

    S_easy = S.copy()
    S_easy["length"] = S_easy["length"].clip(upper=Q1_len)
    S_easy["surprisal"] = S_easy["surprisal"].clip(upper=Q1_surpr)
    S_easy["zipf"] = _clamp_zipf_conditional(S_easy, bin_edges, q3_zipf_by_bin)

    counterfactual_gap = predict_gap_on_sample(ert_predictor, S_easy)
    gap_shrink_ms = G0 - counterfactual_gap
    gap_shrink_pct = (gap_shrink_ms / G0 * 100) if G0 != 0 else 0

    # --------- Single-pass bootstrap: collect shrink and counterfactual gap samples ----------
    print("    Gap shrink & CF-gap bootstrap (single pass)...", flush=True)

    ctrl_subjects = S.loc[S["group"] == "control", "subject_id"].unique()
    dys_subjects = S.loc[S["group"] == "dyslexic", "subject_id"].unique()

    shrink_samples, cf_gap_samples = [], []

    for i in tqdm(range(n_bootstrap), desc="      Bootstrap", leave=False):
        rng = np.random.RandomState(3000 + i)
        boot_ctrl = rng.choice(ctrl_subjects, size=len(ctrl_subjects), replace=True)
        boot_dys = rng.choice(dys_subjects, size=len(dys_subjects), replace=True)

        parts = []
        for j, s in enumerate(boot_ctrl):
            df = S[S["subject_id"] == s].copy()
            df["subject_id"] = f"ctrl_{s}_{j}"
            parts.append(df)
        for j, s in enumerate(boot_dys):
            df = S[S["subject_id"] == s].copy()
            df["subject_id"] = f"dys_{s}_{j}"
            parts.append(df)
        boot = pd.concat(parts, ignore_index=True)

        base_b = predict_gap_on_sample(ert_predictor, boot)
        if not np.isfinite(base_b):
            continue

        boot_easy = boot.copy()
        boot_easy["length"] = boot_easy["length"].clip(upper=Q1_len)
        boot_easy["surprisal"] = boot_easy["surprisal"].clip(upper=Q1_surpr)
        boot_easy["zipf"] = _clamp_zipf_conditional(
            boot_easy, bin_edges, q3_zipf_by_bin
        )

        cf_b = predict_gap_on_sample(ert_predictor, boot_easy)
        if not np.isfinite(cf_b):
            continue

        cf_gap_samples.append(cf_b)
        shrink_samples.append(base_b - cf_b)

    shrink_arr = np.array(shrink_samples, dtype=float)
    cf_arr = np.array(cf_gap_samples, dtype=float)

    # Stats (two-tailed, +1 correction)
    if shrink_arr.size > 0:
        shrink_ci_low, shrink_ci_high = np.percentile(shrink_arr, [2.5, 97.5])
        shrink_p = compute_two_tailed_pvalue_corrected(gap_shrink_ms, shrink_arr)
    else:
        shrink_ci_low = shrink_ci_high = shrink_p = np.nan

    if cf_arr.size > 0:
        cf_ci_low, cf_ci_high = np.percentile(cf_arr, [2.5, 97.5])
        cf_p = compute_two_tailed_pvalue_corrected(counterfactual_gap, cf_arr)
    else:
        cf_ci_low = cf_ci_high = cf_p = np.nan

    stats_gap_shrink = {
        "mean": float(gap_shrink_ms),
        "ci_low": float(shrink_ci_low),
        "ci_high": float(shrink_ci_high),
        "p_value": float(shrink_p),
        "n_bootstrap": int(shrink_arr.size),
    }

    stats_cf_gap = {
        "mean": float(counterfactual_gap),
        "ci_low": float(cf_ci_low),
        "ci_high": float(cf_ci_high),
        "p_value": float(cf_p),
        "n_bootstrap": int(cf_arr.size),
    }
    # -----------------------------------------------------------------------------------------

    logger.info(f"    Baseline gap G0: {G0:.2f} ms")
    logger.info(f"    Counterfactual gap: {counterfactual_gap:.2f} ms")
    logger.info(f"    Gap shrink: {gap_shrink_ms:.2f} ms ({gap_shrink_pct:.1f}%)")

    return {
        "baseline_gap": float(G0),
        "counterfactual_gap": float(counterfactual_gap),
        "counterfactual_gap_stats": stats_cf_gap,
        "gap_shrink_stats": stats_gap_shrink,
        "gap_shrink_ms": float(gap_shrink_ms),
        "gap_shrink_pct": float(gap_shrink_pct),
    }


def equal_ease_feature_contributions(
    ert_predictor,
    S: pd.DataFrame,
    quartiles: Dict,
    gap_shrink_ms: float,
    n_bootstrap: int = 2000,
    bin_edges: np.ndarray = None,
) -> Dict:
    """
    Shapley feature contributions via exact enumeration of 3! orders.
    Uncertainty via subject bootstrap; policy thresholds fixed from S.
    """
    if bin_edges is None:
        raise ValueError("bin_edges must be provided for conditional Zipf equal-ease.")
    q3_zipf_by_bin = _zipf_q3_by_length_bin(S, bin_edges)
    features = ["length", "zipf", "surprisal"]
    orders = list(itertools.permutations(features))  # 6 orders

    def v(df):
        return predict_gap_on_sample(ert_predictor, df)

    def apply_subset(df, subset):
        return _apply_equal_ease_subset(
            df, subset, quartiles, bin_edges, q3_zipf_by_bin
        )

    # Point estimate on S by enumerating all 6 orders
    baseline_gap = v(S)
    per_order_contribs = {f: [] for f in features}
    for order in orders:
        applied = set()
        gap_prev = baseline_gap
        for feat in order:
            applied.add(feat)
            S_curr = apply_subset(S, applied)
            gap_new = v(S_curr)
            if np.isnan(gap_new):
                continue
            per_order_contribs[feat].append(gap_prev - gap_new)
            gap_prev = gap_new

    contrib_point = {
        f: float(np.mean(per_order_contribs[f])) if per_order_contribs[f] else 0.0
        for f in features
    }

    # Bootstrap for CIs/p-values: re-enumerate 6 orders each draw
    contrib_samples = {f: [] for f in features}
    if n_bootstrap and n_bootstrap > 0:
        ctrl_subj = S.loc[S["group"] == "control", "subject_id"].unique()
        dys_subj = S.loc[S["group"] == "dyslexic", "subject_id"].unique()

        for i in tqdm(
            range(n_bootstrap), desc="      Feature Shapley bootstrap", leave=False
        ):
            rng = np.random.RandomState(7000 + i)
            boot_ctrl = rng.choice(ctrl_subj, size=len(ctrl_subj), replace=True)
            boot_dys = rng.choice(dys_subj, size=len(dys_subj), replace=True)

            parts = []
            for j, s in enumerate(boot_ctrl):
                df = S[S["subject_id"] == s].copy()
                df["subject_id"] = f"ctrl_{s}_{j}"
                parts.append(df)
            for j, s in enumerate(boot_dys):
                df = S[S["subject_id"] == s].copy()
                df["subject_id"] = f"dys_{s}_{j}"
                parts.append(df)
            boot = pd.concat(parts, ignore_index=True)

            base = v(boot)
            if np.isnan(base):
                continue

            sums = {f: 0.0 for f in features}
            counts = {f: 0 for f in features}
            for order in orders:
                applied = set()
                gap_prev = base
                for feat in order:
                    applied.add(feat)
                    boot_curr = apply_subset(boot, applied)
                    gap_new = v(boot_curr)
                    if np.isnan(gap_new):
                        continue
                    delta = gap_prev - gap_new
                    sums[feat] += delta
                    counts[feat] += 1
                    gap_prev = gap_new

            for f in features:
                if counts[f] > 0:
                    contrib_samples[f].append(sums[f] / counts[f])

    # Assemble stats
    stats = {}
    for f in features:
        if contrib_samples[f]:
            arr = np.array(contrib_samples[f], float)
            ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
            p = compute_two_tailed_pvalue_corrected(contrib_point[f], arr)
            stats[f] = {
                "mean": float(contrib_point[f]),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_value": float(p),
                "n_samples": int(arr.size),
            }
        else:
            stats[f] = {
                "mean": float(contrib_point[f]),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "n_samples": 0,
            }

    total = sum(contrib_point.values())
    # Optional renormalization to gap_shrink_ms if small drift:
    if not np.isnan(gap_shrink_ms) and abs(total - gap_shrink_ms) > max(
        1.0, 0.1 * abs(gap_shrink_ms)
    ):
        scale = gap_shrink_ms / total if abs(total) > 1e-9 else 1.0
        for f in features:
            stats[f]["mean"] *= scale
            if contrib_samples[f]:
                stats[f]["ci_low"] *= scale
                stats[f]["ci_high"] *= scale
        total = gap_shrink_ms

    return {
        "feature_contributions_ms": {f: stats[f]["mean"] for f in features},
        "feature_contributions_stats": stats,
        "total_ms": float(total),
        "n_orders": len(orders),
        "n_bootstrap": int(n_bootstrap),
    }


def test_hypothesis_3(
    ert_predictor,
    data: pd.DataFrame,
    quartiles: Dict[str, Dict[str, float]],
    n_bootstrap: int = 2000,
    bin_edges: np.ndarray = None,
) -> Dict:
    """
    Test Hypothesis 3: Gap decomposition -
    Single source of truth: one sample S, one baseline G0.
    All analyses use the same S, G0, and equal-ease policy.
    """
    logger.info("=" * 60)
    logger.info("HYPOTHESIS 3: GAP DECOMPOSITION")
    logger.info("=" * 60)
    logger.info(f"Using {n_bootstrap} bootstrap iterations")
    print(f"\n=== H3: Gap Decomposition (n_bootstrap={n_bootstrap}) ===\n", flush=True)

    if bin_edges is None:
        raise ValueError(
            "H3 requires bin_edges for conditional Zipf equal-ease (match H1)."
        )

    # Single analysis sample S
    np.random.seed(ANALYSIS_SEED)
    S = data.copy()

    logger.info(f"\nAnalysis sample S: {len(S):,} observations")
    logger.info(f"Random seed: {ANALYSIS_SEED}")

    # Canonical gap G0
    G0 = predict_gap_on_sample(ert_predictor, S)
    logger.info(f"\nCanonical gap G0: {G0:.2f} ms")

    # Bootstrap G0
    def compute_gap_bootstrap(boot_data):
        return predict_gap_on_sample(ert_predictor, boot_data)

    logger.info("  Computing statistics for G0...")
    print("  Computing G0 statistics...", flush=True)
    gap_stats = bootstrap_gap_component(
        G0, S, compute_gap_bootstrap, n_bootstrap=n_bootstrap
    )

    # Run analyses on the same S / G0
    shapley = shapley_decomposition(ert_predictor, S, G0, n_bootstrap)
    equal_ease = equal_ease_counterfactual(
        ert_predictor, S, quartiles, G0, n_bootstrap, bin_edges=bin_edges
    )

    feature_contributions = equal_ease_feature_contributions(
        ert_predictor,
        S,
        quartiles,
        equal_ease["gap_shrink_ms"],
        n_bootstrap=n_bootstrap,
        bin_edges=bin_edges,
    )

    text_difficulty_explained = equal_ease.get("gap_shrink_pct", 0) >= 20
    status = "CONFIRMED" if text_difficulty_explained else "NOT CONFIRMED"
    summary = (
        "text difficulty (simplification reduces gap)"
        if text_difficulty_explained
        else "features explain minimal portion of gap"
    )

    logger.info(f"\nHYPOTHESIS 3: {status}")
    logger.info(f"  {summary}")

    return {
        "status": status,
        "canonical_gap_G0": float(G0),
        "canonical_gap_stats": gap_stats,
        "sample_size": len(S),
        "random_seed": ANALYSIS_SEED,
        "shapley_decomposition": shapley,
        "equal_ease_counterfactual": equal_ease,
        "equal_ease_feature_contributions": feature_contributions,
        "summary": summary,
    }
