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


# ========================= NEW FAST HELPERS (for bootstraps) =========================


def _precompute_shapley_subject_means(ert_predictor, S: pd.DataFrame):
    """
    Precompute per-dyslexic-subject means needed for skip-vs-duration Shapley.
    Returns:
      subj_dys, mean_ed, mean_ec, mean_ert1, mean_ert3
      where:
        ed   = ERT(dys model on dys items)
        ec   = ERT(ctrl model on dys items)
        ert1 = (1 - p_ctrl) * TRT_dys
        ert3 = (1 - p_dys) * TRT_ctrl
    """
    dys = S[S["group"] == "dyslexic"].copy()
    if dys.empty:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # Predict components ONCE on dys items, with both models
    ed, psd, td = predict_components_batch(ert_predictor, dys, "dyslexic")
    ec, psc, tc = predict_components_batch(ert_predictor, dys, "control")

    ert1 = (1.0 - psc) * td  # skip equalized first
    ert3 = (1.0 - psd) * tc  # duration equalized first

    subj = np.sort(dys["subject_id"].unique())

    def _by_subj_mean(vals):
        df = (
            pd.DataFrame({"subject_id": dys["subject_id"].values, "v": vals})
            .groupby("subject_id")["v"]
            .mean()
        )
        return df.reindex(subj).to_numpy(dtype=float)

    mean_ed = _by_subj_mean(ed)
    mean_ec = _by_subj_mean(ec)
    mean_ert1 = _by_subj_mean(ert1)
    mean_ert3 = _by_subj_mean(ert3)

    return subj, mean_ed, mean_ec, mean_ert1, mean_ert3


def _precompute_counterfactual_subject_means(ert_predictor, S, quartiles, bin_edges):
    """
    Precompute per-subject mean ERT for baseline and equal-ease configs
    for each group. Returns:
      subj_ctrl, subj_dys, base_ctrl, base_dys, easy_ctrl, easy_dys
    """
    # Baseline features
    X_base = S[["length", "zipf", "surprisal"]].copy()

    # Equal-ease features: clamp length & surprisal; zipf conditional on (clipped) length
    Q1_len = float(quartiles["length"]["q1"])
    Q1_surpr = float(quartiles["surprisal"]["q1"])
    X_easy = X_base.copy()
    X_easy["length"] = np.minimum(X_easy["length"].to_numpy(), Q1_len)
    X_easy["surprisal"] = np.minimum(X_easy["surprisal"].to_numpy(), Q1_surpr)

    q3_zipf_by_bin = _zipf_q3_by_length_bin(S, bin_edges)
    X_easy["zipf"] = _clamp_zipf_conditional(
        X_easy, bin_edges, q3_zipf_by_bin
    ).to_numpy()

    m_ctrl = (S["group"] == "control").to_numpy()
    m_dys = (S["group"] == "dyslexic").to_numpy()
    subj_ctrl = np.sort(S.loc[m_ctrl, "subject_id"].unique())
    subj_dys = np.sort(S.loc[m_dys, "subject_id"].unique())

    def _per_subject_means(X, mask, group, subj_sorted):
        y = np.asarray(
            ert_predictor.predict_ert(
                X.loc[mask, ["length", "zipf", "surprisal"]], group
            )
        ).ravel()
        df = (
            pd.DataFrame({"subject_id": S.loc[mask, "subject_id"].values, "ert": y})
            .groupby("subject_id")["ert"]
            .mean()
        )
        return df.reindex(subj_sorted).to_numpy(dtype=float)

    base_ctrl = (
        _per_subject_means(X_base, m_ctrl, "control", subj_ctrl)
        if subj_ctrl.size
        else np.array([])
    )
    base_dys = (
        _per_subject_means(X_base, m_dys, "dyslexic", subj_dys)
        if subj_dys.size
        else np.array([])
    )
    easy_ctrl = (
        _per_subject_means(X_easy, m_ctrl, "control", subj_ctrl)
        if subj_ctrl.size
        else np.array([])
    )
    easy_dys = (
        _per_subject_means(X_easy, m_dys, "dyslexic", subj_dys)
        if subj_dys.size
        else np.array([])
    )

    return subj_ctrl, subj_dys, base_ctrl, base_dys, easy_ctrl, easy_dys


# ===================================================================================


def shapley_decomposition(
    ert_predictor, S: pd.DataFrame, G0: float, n_bootstrap: int = 2000
) -> Dict:
    """
    Shapley decomposition - skip vs duration contributions.
    Uses the SAME sample S and baseline G0 as rest of H3.

    FAST: precompute per-dys-subject means once, then vectorized resampling
    of dys subject IDs (no per-draw predictions).
    """
    logger.info("  Computing Shapley decomposition...")
    print("  Shapley decomposition...", flush=True)

    # Precompute subject-level means needed
    (subj_dys, mean_ed, mean_ec, mean_ert1, mean_ert3) = (
        _precompute_shapley_subject_means(ert_predictor, S)
    )
    if subj_dys.size == 0:
        return {"error": "No dyslexic data"}

    # Point estimates on S (dys items)
    mean_ert_dys = float(mean_ed.mean())
    mean_ert_ctrl = float(mean_ec.mean())
    ert_1_mean = float(mean_ert1.mean())
    ert_3_mean = float(mean_ert3.mean())
    computed_gap = mean_ert_dys - mean_ert_ctrl

    # Path contributions
    delta_skip_path1 = mean_ert_dys - ert_1_mean
    delta_duration_path1 = ert_1_mean - mean_ert_ctrl
    delta_duration_path2 = mean_ert_dys - ert_3_mean
    delta_skip_path2 = ert_3_mean - mean_ert_ctrl

    # Average Shapley values
    delta_skip = 0.5 * (delta_skip_path1 + delta_skip_path2)
    delta_duration = 0.5 * (delta_duration_path1 + delta_duration_path2)

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

    # ---------- Vectorized bootstrap over dys subjects only ----------
    print("    Shapley bootstrap (single pass)...", flush=True)

    n_dys = subj_dys.size
    if n_bootstrap and n_dys > 0:
        rng = np.random.default_rng(3000)
        dys_idx = rng.integers(0, n_dys, size=(n_bootstrap, n_dys))

        mean_dys_b = mean_ed[dys_idx].mean(axis=1)
        mean_ctrl_b = mean_ec[dys_idx].mean(axis=1)
        ert_1_b = mean_ert1[dys_idx].mean(axis=1)
        ert_3_b = mean_ert3[dys_idx].mean(axis=1)

        skip_1 = mean_dys_b - ert_1_b
        skip_2 = ert_3_b - mean_ctrl_b
        dur_1 = mean_dys_b - ert_3_b
        dur_2 = ert_1_b - mean_ctrl_b

        skip_arr = 0.5 * (skip_1 + skip_2)
        dur_arr = 0.5 * (dur_1 + dur_2)
    else:
        skip_arr = np.array([])
        dur_arr = np.array([])

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

    FAST: precompute per-subject baseline & equal-ease means once,
    then vectorized resampling of subject IDs (no model calls in bootstrap).
    """
    logger.info("  Computing equal-ease counterfactual...")
    print("  Equal-ease counterfactual...", flush=True)

    if bin_edges is None:
        raise ValueError("bin_edges must be provided for conditional Zipf equal-ease.")

    # Precompute per-subject means (baseline & easy) for each group
    (
        subj_ctrl,
        subj_dys,
        base_ctrl,
        base_dys,
        easy_ctrl,
        easy_dys,
    ) = _precompute_counterfactual_subject_means(ert_predictor, S, quartiles, bin_edges)

    # Point estimate on S, anchored to G0
    baseline_gap = (
        float(base_dys.mean() - base_ctrl.mean())
        if (base_dys.size and base_ctrl.size)
        else np.nan
    )
    if np.isfinite(baseline_gap) and abs(baseline_gap - G0) > BASELINE_TOLERANCE:
        logger.warning(f"    Baseline gap {baseline_gap:.2f} differs from G0 {G0:.2f}")

    counterfactual_gap = (
        float(easy_dys.mean() - easy_ctrl.mean())
        if (easy_dys.size and easy_ctrl.size)
        else np.nan
    )
    gap_shrink_ms = (
        float(G0 - counterfactual_gap) if np.isfinite(counterfactual_gap) else np.nan
    )
    gap_shrink_pct = (
        (gap_shrink_ms / G0 * 100) if (G0 != 0 and np.isfinite(gap_shrink_ms)) else 0
    )

    # --------- Vectorized bootstrap on subject IDs ----------
    print("    Gap shrink & CF-gap bootstrap (single pass)...", flush=True)

    shrink_arr = np.array([])
    cf_arr = np.array([])

    n_ctrl = subj_ctrl.size
    n_dys = subj_dys.size

    if n_bootstrap and n_ctrl > 0 and n_dys > 0:
        rng = np.random.default_rng(3000)
        ctrl_idx = rng.integers(0, n_ctrl, size=(n_bootstrap, n_ctrl))
        dys_idx = rng.integers(0, n_dys, size=(n_bootstrap, n_dys))

        base_ctrl_boot = base_ctrl[ctrl_idx].mean(axis=1)
        base_dys_boot = base_dys[dys_idx].mean(axis=1)
        easy_ctrl_boot = easy_ctrl[ctrl_idx].mean(axis=1)
        easy_dys_boot = easy_dys[dys_idx].mean(axis=1)

        cf_arr = easy_dys_boot - easy_ctrl_boot
        shrink_arr = (base_dys_boot - base_ctrl_boot) - cf_arr

    # Stats (two-tailed, +1 correction)
    if shrink_arr.size > 0 and np.isfinite(gap_shrink_ms):
        shrink_ci_low, shrink_ci_high = np.percentile(shrink_arr, [2.5, 97.5])
        shrink_p = compute_two_tailed_pvalue_corrected(gap_shrink_ms, shrink_arr)
    else:
        shrink_ci_low = shrink_ci_high = shrink_p = np.nan

    if cf_arr.size > 0 and np.isfinite(counterfactual_gap):
        cf_ci_low, cf_ci_high = np.percentile(cf_arr, [2.5, 97.5])
        cf_p = compute_two_tailed_pvalue_corrected(counterfactual_gap, cf_arr)
    else:
        cf_ci_low = cf_ci_high = cf_p = np.nan

    stats_gap_shrink = {
        "mean": float(gap_shrink_ms),
        "ci_low": float(shrink_ci_low),
        "ci_high": float(shrink_ci_high),
        "p_value": float(shrink_p),
        "n_bootstrap": int(cf_arr.size),
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


# ---------------------- NEW: ultra-fast Shapley feature contributions ----------------------


def _precompute_config_subject_means(
    ert_predictor,
    S: pd.DataFrame,
    quartiles: Dict,
    bin_edges: np.ndarray,
):
    """
    Precompute per-subject mean ERT for each group and each of the 8 equal-ease
    configurations (bitmask over {length, zipf, surprisal} = {1,2,4}).

    Returns:
      subjects_ctrl: np.ndarray of unique control subject_ids (sorted)
      subjects_dys:  np.ndarray of unique dyslexic subject_ids (sorted)
      means_ctrl:    np.ndarray shape (8, n_ctrl_subjects) per-config subject means
      means_dys:     np.ndarray shape (8, n_dys_subjects)  per-config subject means
    """
    # Original columns
    L = S["length"].to_numpy()
    Z = S["zipf"].to_numpy()
    R = S["surprisal"].to_numpy()

    # Clamp once
    L_clamp = np.minimum(L, float(quartiles["length"]["q1"]))
    R_clamp = np.minimum(R, float(quartiles["surprisal"]["q1"]))

    q3_zipf_by_bin = _zipf_q3_by_length_bin(S, bin_edges)
    Z_clamp = _clamp_zipf_conditional(S, bin_edges, q3_zipf_by_bin).to_numpy()

    # Group masks and subject lists
    m_ctrl = (S["group"] == "control").to_numpy()
    m_dys = (S["group"] == "dyslexic").to_numpy()

    subj_ctrl = np.sort(S.loc[m_ctrl, "subject_id"].unique())
    subj_dys = np.sort(S.loc[m_dys, "subject_id"].unique())

    n_ctrl = subj_ctrl.size
    n_dys = subj_dys.size

    means_ctrl = np.empty((8, n_ctrl), dtype=float)
    means_dys = np.empty((8, n_dys), dtype=float)

    def _X_for_cfg(mask_bits):
        L_use = L_clamp if (mask_bits & 1) else L
        Z_use = Z_clamp if (mask_bits & 2) else Z
        R_use = R_clamp if (mask_bits & 4) else R
        return pd.DataFrame(
            {"length": L_use, "zipf": Z_use, "surprisal": R_use},
            index=S.index,
        )

    for cfg in range(8):
        X_cfg = _X_for_cfg(cfg)

        if n_ctrl > 0:
            ert_ctrl = np.asarray(
                ert_predictor.predict_ert(
                    X_cfg.loc[m_ctrl, ["length", "zipf", "surprisal"]], "control"
                )
            ).ravel()
            dfc = (
                pd.DataFrame(
                    {"subject_id": S.loc[m_ctrl, "subject_id"].values, "ert": ert_ctrl}
                )
                .groupby("subject_id")["ert"]
                .mean()
            )
            means_ctrl[cfg, :] = dfc.reindex(subj_ctrl).to_numpy()

        if n_dys > 0:
            ert_dys = np.asarray(
                ert_predictor.predict_ert(
                    X_cfg.loc[m_dys, ["length", "zipf", "surprisal"]], "dyslexic"
                )
            ).ravel()
            dfd = (
                pd.DataFrame(
                    {"subject_id": S.loc[m_dys, "subject_id"].values, "ert": ert_dys}
                )
                .groupby("subject_id")["ert"]
                .mean()
            )
            means_dys[cfg, :] = dfd.reindex(subj_dys).to_numpy()

    return subj_ctrl, subj_dys, means_ctrl, means_dys


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

    FAST version: precompute per-subject ERT means for each of the 8 clamp configs,
    then bootstrap only over subject IDs; compute exact Shapley from the 8 v(S) values.
    """
    if bin_edges is None:
        raise ValueError("bin_edges must be provided for conditional Zipf equal-ease.")

    # 1) Precompute per-subject means for each config (0..7)
    subj_ctrl, subj_dys, means_ctrl, means_dys = _precompute_config_subject_means(
        ert_predictor, S, quartiles, bin_edges
    )
    n_ctrl = subj_ctrl.size
    n_dys = subj_dys.size

    # Config bitmasks
    C_NONE = 0
    C_L = 1
    C_Z = 2
    C_S = 4
    C_LZ = C_L | C_Z
    C_LS = C_L | C_S
    C_ZS = C_Z | C_S
    C_LZS = C_L | C_Z | C_S

    # 2) Point estimate via exact Shapley from 8 configs
    def v_cfg(cfg):
        return float(means_dys[cfg, :].mean() - means_ctrl[cfg, :].mean())

    v = {cfg: v_cfg(cfg) for cfg in range(8)}

    def shapley_from_v(v, i_name):
        if i_name == "length":
            i, j, k = C_L, C_Z, C_S
            ij, ik, jk, ijk = C_LZ, C_LS, C_ZS, C_LZS
        elif i_name == "zipf":
            i, j, k = C_Z, C_L, C_S
            ij, ik, jk, ijk = C_LZ, C_ZS, C_LS, C_LZS
        else:  # surprisal
            i, j, k = C_S, C_L, C_Z
            ij, ik, jk, ijk = C_LS, C_ZS, C_LZ, C_LZS
        return (
            (1 / 3) * (v[i] - v[C_NONE])
            + (1 / 6) * (v[ij] - v[j])
            + (1 / 6) * (v[ik] - v[k])
            + (1 / 3) * (v[ijk] - v[jk])
        )

    contrib_point = {
        "length": shapley_from_v(v, "length"),
        "zipf": shapley_from_v(v, "zipf"),
        "surprisal": shapley_from_v(v, "surprisal"),
    }

    # 3) Bootstrap on subject means only (no model calls)
    features = ["length", "zipf", "surprisal"]
    contrib_samples = {f: [] for f in features}

    if n_bootstrap and n_bootstrap > 0 and (n_ctrl > 0 and n_dys > 0):
        rng = np.random.default_rng(7000)
        ctrl_idx = rng.integers(0, n_ctrl, size=(n_bootstrap, n_ctrl))
        dys_idx = rng.integers(0, n_dys, size=(n_bootstrap, n_dys))

        def boot_gap(cfg):
            ctrl_means = means_ctrl[cfg, :][ctrl_idx].mean(axis=1)
            dys_means = means_dys[cfg, :][dys_idx].mean(axis=1)
            return dys_means - ctrl_means

        v_boot = {cfg: boot_gap(cfg) for cfg in range(8)}

        def shapley_boot(vb, i_name):
            if i_name == "length":
                i, j, k = C_L, C_Z, C_S
                ij, ik, jk, ijk = C_LZ, C_LS, C_ZS, C_LZS
            elif i_name == "zipf":
                i, j, k = C_Z, C_L, C_S
                ij, ik, jk, ijk = C_LZ, C_ZS, C_LS, C_LZS
            else:
                i, j, k = C_S, C_L, C_Z
                ij, ik, jk, ijk = C_LS, C_ZS, C_LZ, C_LZS
            return (
                (1 / 3) * (vb[i] - vb[C_NONE])
                + (1 / 6) * (vb[ij] - vb[j])
                + (1 / 6) * (vb[ik] - vb[k])
                + (1 / 3) * (vb[ijk] - vb[jk])
            )

        for f in features:
            contrib_samples[f] = shapley_boot(v_boot, f)

    # 4) Assemble stats and optional renormalization (kept identical)
    stats = {}
    for f in features:
        arr = contrib_samples.get(f, None)
        if arr is not None and len(arr) > 0:
            ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
            p = compute_two_tailed_pvalue_corrected(contrib_point[f], arr)
            stats[f] = {
                "mean": float(contrib_point[f]),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_value": float(p),
                "n_bootstrap": int(arr.size),
            }
        else:
            stats[f] = {
                "mean": float(contrib_point[f]),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "n_bootstrap": 0,
            }

    total = sum(s["mean"] for s in stats.values())

    if not np.isnan(gap_shrink_ms) and abs(total - gap_shrink_ms) > max(
        1.0, 0.1 * abs(gap_shrink_ms)
    ):
        scale = gap_shrink_ms / total if abs(total) > 1e-9 else 1.0
        for f in features:
            stats[f]["mean"] *= scale
            if contrib_samples.get(f, None) is not None and len(contrib_samples[f]) > 0:
                stats[f]["ci_low"] *= scale
                stats[f]["ci_high"] *= scale
        total = gap_shrink_ms

    return {
        "feature_contributions_ms": {f: stats[f]["mean"] for f in features},
        "feature_contributions_stats": stats,
        "total_ms": float(total),
        "n_orders": 6,
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
