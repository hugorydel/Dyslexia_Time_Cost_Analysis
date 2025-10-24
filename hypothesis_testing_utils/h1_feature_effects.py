"""
Hypothesis 1: Feature Effects with CONDITIONAL Zipf Evaluation
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hypothesis_testing_utils.stats_utils import (
    cohens_h,
    compute_cohens_d_from_data,
    compute_two_tailed_pvalue_corrected,
)

logger = logging.getLogger(__name__)


# ==========================================
# Subject-level delta helpers (core change)
# ==========================================


def _predict_components(
    ert_predictor, X: pd.DataFrame, group: str, pathway: str
) -> np.ndarray:
    """
    Predicts the requested pathway from ert_predictor for a batch of rows.
    pathway in {"skip", "duration", "ert"}.
    """
    if pathway == "ert":
        y = ert_predictor.predict_ert(X, group)  # shape (n,)
    else:
        _, p_skip, trt = ert_predictor.predict_ert(X, group, return_components=True)
        if pathway == "skip":
            y = p_skip  # shape (n,)
        elif pathway == "duration":
            y = trt  # shape (n,)
        else:
            raise ValueError(f"Unknown pathway: {pathway}")
    return np.asarray(y).reshape(-1)


def _compute_bin_zipf_quartiles(
    gdf: pd.DataFrame, bin_edges: np.ndarray
) -> List[Tuple[float, float]]:
    """
    Compute (zipf_q1, zipf_q3) per length bin from the group data (gdf).
    """
    gdf = gdf.copy()
    gdf["length_bin"] = pd.cut(
        gdf["length"], bins=bin_edges, labels=False, include_lowest=True
    )
    n_bins = len(bin_edges) - 1
    bin_qs = []
    for b in range(n_bins):
        bin_data = gdf[gdf["length_bin"] == b]
        if len(bin_data) == 0:
            bin_qs.append((np.nan, np.nan))
        else:
            bin_qs.append(
                (
                    float(bin_data["zipf"].quantile(0.25)),
                    float(bin_data["zipf"].quantile(0.75)),
                )
            )
    return bin_qs  # list of (q1_bin, q3_bin)


def _subject_level_deltas_zipf_conditional(
    ert_predictor,
    sdf: pd.DataFrame,
    group: str,
    pathway: str,
    bin_edges: np.ndarray,
    bin_qs: List[Tuple[float, float]],
    bin_weights: Optional[pd.Series] = None,
) -> Optional[float]:
    """
    Conditional Zipf: within length bins. For each bin present in the subject's data,
    set Zipf to that bin's (q1, q3) while leaving the subject's own length/surprisal values;
    compute per-bin mean delta, then weighted-average across bins.

    Returns the subject-level weighted delta, or None if no usable bins.
    """
    sdf = sdf.copy()
    sdf["length_bin"] = pd.cut(
        sdf["length"], bins=bin_edges, labels=False, include_lowest=True
    )

    n_bins = len(bin_qs)
    bin_effects = []
    bin_ws = []

    for b in range(n_bins):
        sub_bin = sdf[sdf["length_bin"] == b]
        if len(sub_bin) == 0:
            continue
        q1_bin, q3_bin = bin_qs[b]
        if np.isnan(q1_bin) or np.isnan(q3_bin):
            continue

        X = sub_bin[["length", "zipf", "surprisal"]].copy()
        X_q1 = X.copy()
        X_q3 = X.copy()
        X_q1["zipf"] = q1_bin
        X_q3["zipf"] = q3_bin

        y1 = _predict_components(ert_predictor, X_q1, group, pathway)
        y3 = _predict_components(ert_predictor, X_q3, group, pathway)
        delta_b = float(np.mean(y3 - y1))
        bin_effects.append(delta_b)

        if bin_weights is not None:
            bin_ws.append(float(bin_weights.iloc[b]))
        else:
            bin_ws.append(1.0)

    if len(bin_effects) == 0:
        return None

    ws = np.array(bin_ws, dtype=float)
    ws = ws / ws.sum()
    return float(np.average(np.array(bin_effects, dtype=float), weights=ws))


def _subject_level_deltas_unconditional(
    ert_predictor,
    sdf: pd.DataFrame,
    feature: str,
    group: str,
    q1: float,
    q3: float,
    pathway: str,
) -> float:
    """
    For one subject's data (sdf), compute the mean predicted difference (Q3 - Q1)
    using the subject's actual other features; only 'feature' is set to q1/q3.
    Returns the subject-level mean delta.
    """
    X = sdf[["length", "zipf", "surprisal"]].copy()
    X_q1 = X.copy()
    X_q3 = X.copy()
    X_q1[feature] = q1
    X_q3[feature] = q3

    y1 = _predict_components(ert_predictor, X_q1, group, pathway)
    y3 = _predict_components(ert_predictor, X_q3, group, pathway)
    return float(np.mean(y3 - y1))


def subject_level_deltas(
    ert_predictor,
    data: pd.DataFrame,
    feature: str,
    group: str,
    q1: float,
    q3: float,
    pathway: str,  # "skip", "duration", "ert"
    bin_edges: Optional[np.ndarray] = None,
    bin_weights: Optional[pd.Series] = None,
) -> np.ndarray:
    """
    Compute subject-level deltas (Q3-Q1) for the requested pathway.
    For Zipf + bins provided, uses the conditional-in-bins approach with bin weights.
    Otherwise uses the unconditional (set feature=q1/q3) approach.
    """
    gdf = data[data["group"] == group]
    deltas = []

    # Precompute bin-specific Zipf quartiles for conditional Zipf (group-level)
    bin_qs = None
    use_conditional_zipf = (feature == "zipf") and (bin_edges is not None)
    if use_conditional_zipf:
        bin_qs = _compute_bin_zipf_quartiles(gdf, bin_edges)

    for sid, sdf in gdf.groupby("subject_id"):
        if use_conditional_zipf:
            d = _subject_level_deltas_zipf_conditional(
                ert_predictor, sdf, group, pathway, bin_edges, bin_qs, bin_weights
            )
            if d is None:
                continue
        else:
            d = _subject_level_deltas_unconditional(
                ert_predictor, sdf, feature, group, q1, q3, pathway
            )
        deltas.append(d)

    return np.array(deltas, dtype=float)


# Vectorized helper for all pathways


def _batch_q1_q3_preds_all_paths(
    ert_predictor, X_q1: pd.DataFrame, X_q3: pd.DataFrame, group: str
):
    """Two predictor calls -> arrays for ert, p(skip), trt at Q1 and Q3 (vectorized)."""
    ert1, p1, trt1 = ert_predictor.predict_ert(X_q1, group, return_components=True)
    ert3, p3, trt3 = ert_predictor.predict_ert(X_q3, group, return_components=True)
    return {
        "ert1": np.asarray(ert1).ravel(),
        "ert3": np.asarray(ert3).ravel(),
        "p1": np.asarray(p1).ravel(),
        "p3": np.asarray(p3).ravel(),
        "trt1": np.asarray(trt1).ravel(),
        "trt3": np.asarray(trt3).ravel(),
    }


def subject_level_deltas_all_pathways_uncond(
    ert_predictor, data: pd.DataFrame, feature: str, group: str, q1: float, q3: float
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Vectorized subject-level deltas for length/surprisal (skip/duration/ert simultaneously).
    Returns:
      - deltas_df: index=subject_id, cols=['skip','duration','ert']
      - preds: raw arrays at Q1/Q3 for effect sizes (p/ert/trt)
    """
    g = data.loc[data["group"] == group].copy()
    X = g[["length", "zipf", "surprisal"]]
    X_q1, X_q3 = X.copy(), X.copy()
    X_q1[feature] = q1
    X_q3[feature] = q3

    preds = _batch_q1_q3_preds_all_paths(ert_predictor, X_q1, X_q3, group)
    d_skip = preds["p3"] - preds["p1"]
    d_trt = preds["trt3"] - preds["trt1"]
    d_ert = preds["ert3"] - preds["ert1"]

    df = pd.DataFrame(
        {
            "subject_id": g["subject_id"].values,
            "skip": d_skip,
            "duration": d_trt,
            "ert": d_ert,
        }
    )
    deltas_df = df.groupby("subject_id", sort=False).mean()
    return deltas_df, preds


def subject_level_deltas_all_pathways_zipf_cond(
    ert_predictor,
    data: pd.DataFrame,
    group: str,
    bin_edges: np.ndarray,
    bin_weights: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Vectorized subject-level deltas for Zipf (conditional within length bins).
    Two calls total; then per-subject, per-bin mean, then weighted avg across used bins.
    """
    g = data.loc[data["group"] == group].copy()
    g["length_bin"] = pd.cut(
        g["length"], bins=bin_edges, labels=False, include_lowest=True
    )

    q1_by_bin = g.groupby("length_bin")["zipf"].quantile(0.25)
    q3_by_bin = g.groupby("length_bin")["zipf"].quantile(0.75)

    z1 = g["length_bin"].map(q1_by_bin)
    z3 = g["length_bin"].map(q3_by_bin)
    mask = z1.notna() & z3.notna()
    g = g.loc[mask].copy()

    X = g[["length", "zipf", "surprisal"]]
    X_q1, X_q3 = X.copy(), X.copy()
    X_q1["zipf"] = z1.loc[mask].values
    X_q3["zipf"] = z3.loc[mask].values

    preds = _batch_q1_q3_preds_all_paths(ert_predictor, X_q1, X_q3, group)
    d_skip = preds["p3"] - preds["p1"]
    d_trt = preds["trt3"] - preds["trt1"]
    d_ert = preds["ert3"] - preds["ert1"]

    # per-subject, per-bin means
    wide = pd.DataFrame(
        {
            "subject_id": g["subject_id"].values,
            "length_bin": g["length_bin"].values,
            "skip": d_skip,
            "duration": d_trt,
            "ert": d_ert,
        }
    )
    per_sb = wide.groupby(["subject_id", "length_bin"], sort=False).mean()

    # attach weights and compute weighted avg over available bins for each subject
    w = bin_weights.rename("w")
    per_sb = per_sb.join(w, on="length_bin").dropna(subset=["w"])

    def _wavg(df):
        ws = df["w"].values.astype(float)
        out = {}
        for col in ["skip", "duration", "ert"]:
            out[col] = (
                np.average(df[col].values, weights=ws) if ws.sum() > 0 else np.nan
            )
        return pd.Series(out)

    deltas_df = per_sb.groupby(level=0).apply(_wavg)
    return deltas_df, preds


# ===========================
# AMIE CIs & P-Values
# ===========================


def bootstrap_amie(
    ert_predictor,
    data: pd.DataFrame,
    feature: str,
    group: str,
    quartiles: Dict,
    n_bootstrap: int = 2000,
    method: str = "standard",
    bin_edges: np.ndarray = None,
    bin_weights: pd.Series = None,
) -> Dict:
    """
    Bootstrap AMIE on subject-level deltas for the ERT pathway.
    Two-tailed p with +1 correction; 95% CI via percentiles.
    """
    # Vectorized subject-level deltas for ERT
    if feature == "zipf" and bin_edges is not None and bin_weights is not None:
        deltas_df, _ = subject_level_deltas_all_pathways_zipf_cond(
            ert_predictor, data, group, bin_edges, bin_weights
        )
    else:
        q1 = quartiles[feature]["q1"]
        q3 = quartiles[feature]["q3"]
        deltas_df, _ = subject_level_deltas_all_pathways_uncond(
            ert_predictor, data, feature, group, q1, q3
        )

    deltas = deltas_df["ert"].to_numpy()
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return {
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_bootstrap": n_bootstrap,
            "std_dev": np.nan,
            "method": method,
        }

    obs = float(np.mean(deltas))

    # bootstrap (resample subjects)
    rng = np.random.default_rng(123)
    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, deltas.size, deltas.size)
        boot_means.append(np.mean(deltas[idx]))
    boot_means = np.array(boot_means, dtype=float)

    # CI
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    # +1 correction, two-tailed
    p_value = compute_two_tailed_pvalue_corrected(obs, boot_means)

    return {
        "p_value": float(p_value),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_bootstrap": int(n_bootstrap),
        "std_dev": float(np.std(boot_means, ddof=1)),
        "method": method,
    }


def bootstrap_pathway_effect(
    ert_predictor,
    data: pd.DataFrame,
    feature: str,
    group: str,
    quartiles: Dict,
    pathway: str,  # "skip" | "duration" | "ert"
    n_bootstrap: int = 2000,
    bin_edges: np.ndarray = None,
    bin_weights: pd.Series = None,
) -> dict:
    """
    Inference: subject-level deltas (Q3 - Q1) for the chosen pathway.
    p-value: +1 corrected two-tailed via bootstrap.
    Effect size (descriptive):
      - skip -> Cohen's h (from predicted probabilities)
      - duration & ert -> Hedges-corrected d from predicted values
    """
    # Vectorized subject-level deltas for all pathways (+ raw Q1/Q3 preds for effect sizes)
    if feature == "zipf" and bin_edges is not None and bin_weights is not None:
        deltas_df, preds = subject_level_deltas_all_pathways_zipf_cond(
            ert_predictor, data, group, bin_edges, bin_weights
        )
    else:
        q1 = quartiles[feature]["q1"]
        q3 = quartiles[feature]["q3"]
        deltas_df, preds = subject_level_deltas_all_pathways_uncond(
            ert_predictor, data, feature, group, q1, q3
        )

    if pathway not in ["skip", "duration", "ert"]:
        raise ValueError("pathway must be one of {'skip','duration','ert'}")

    deltas = deltas_df[pathway].to_numpy()
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return {
            "p_value": np.nan,
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_bootstrap": n_bootstrap,
            "std_dev": np.nan,
            "cohens_d": np.nan,
            "cohens_d_ci_low": np.nan,
            "cohens_d_ci_high": np.nan,
            "skip_cohens_h": 0.0 if pathway == "skip" else np.nan,
        }

    # Bootstrap on subject means
    rng = np.random.default_rng(123)
    boot_means = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, deltas.size, deltas.size)
        boot_means[b] = np.mean(deltas[idx])

    obs = float(np.mean(deltas))
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    p_value = compute_two_tailed_pvalue_corrected(obs, boot_means)

    # Descriptive effect sizes reusing already-computed predictions (no extra model calls)
    if pathway == "skip":
        h = cohens_h(float(np.mean(preds["p1"])), float(np.mean(preds["p3"])))
        d_desc, d_lo, d_hi = np.nan, np.nan, np.nan
    elif pathway == "duration":
        d_desc = compute_cohens_d_from_data(preds["trt1"], preds["trt3"])
        d_lo, d_hi = np.nan, np.nan
        h = np.nan
    else:  # "ert"
        d_desc = compute_cohens_d_from_data(preds["ert1"], preds["ert3"])
        d_lo, d_hi = np.nan, np.nan
        h = np.nan

    return {
        "p_value": float(p_value),
        "mean": obs,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_bootstrap": int(n_bootstrap),
        "std_dev": float(np.std(boot_means, ddof=1)),
        "cohens_d": float(d_desc) if np.isfinite(d_desc) else np.nan,
        "cohens_d_ci_low": float(d_lo),
        "cohens_d_ci_high": float(d_hi),
        "skip_cohens_h": float(h) if pathway == "skip" else np.nan,
    }


# ==================================
# AMIE point estimators
# ==================================


def compute_amie_standard(
    ert_predictor, data: pd.DataFrame, feature: str, group: str, quartiles: Dict
) -> Dict:
    """
    Standard AMIE for length and surprisal
    Q1→Q3 shift with other features at mean (descriptive).
    """
    q1 = quartiles[feature]["q1"]
    q3 = quartiles[feature]["q3"]

    group_data = data[data["group"] == group].copy()
    other_features = [f for f in ["length", "zipf", "surprisal"] if f != feature]
    means = {f: group_data[f].mean() for f in other_features}

    grid_q1 = pd.DataFrame({feature: [q1], **means})
    grid_q3 = pd.DataFrame({feature: [q3], **means})

    ert_q1 = ert_predictor.predict_ert(grid_q1, group)[0]
    ert_q3 = ert_predictor.predict_ert(grid_q3, group)[0]

    amie = ert_q3 - ert_q1

    return {
        "feature": feature,
        "group": group,
        "q1": float(q1),
        "q3": float(q3),
        "ert_q1": float(ert_q1),
        "ert_q3": float(ert_q3),
        "amie_ms": float(amie),
        "method": "standard",
    }


def compute_amie_conditional_zipf(
    ert_predictor,
    data: pd.DataFrame,
    group: str,
    bin_edges: np.ndarray,
    bin_weights: pd.Series,
) -> Dict:
    """
    Conditional AMIE for zipf (within length bins), weighted by bin_weights (descriptive).
    """
    group_data = data[data["group"] == group].copy()

    if "length_bin" not in group_data.columns:
        group_data["length_bin"] = pd.cut(
            group_data["length"], bins=bin_edges, labels=False, include_lowest=True
        )

    bin_effects = []
    bin_indices_used = []

    n_bins = len(bin_weights)

    for bin_idx in range(n_bins):
        bin_data = group_data[group_data["length_bin"] == bin_idx]
        if len(bin_data) < 10:
            continue

        zipf_q1_bin = bin_data["zipf"].quantile(0.25)
        zipf_q3_bin = bin_data["zipf"].quantile(0.75)

        length_mean = bin_data["length"].mean()
        surprisal_mean = bin_data["surprisal"].mean()

        grid_q1 = pd.DataFrame(
            {
                "length": [length_mean],
                "zipf": [zipf_q1_bin],
                "surprisal": [surprisal_mean],
            }
        )
        grid_q3 = pd.DataFrame(
            {
                "length": [length_mean],
                "zipf": [zipf_q3_bin],
                "surprisal": [surprisal_mean],
            }
        )

        ert_q1 = ert_predictor.predict_ert(grid_q1, group)[0]
        ert_q3 = ert_predictor.predict_ert(grid_q3, group)[0]

        bin_effect = ert_q3 - ert_q1
        bin_effects.append(bin_effect)
        bin_indices_used.append(bin_idx)

    if not bin_effects:
        return {
            "feature": "zipf",
            "group": group,
            "amie_ms": np.nan,
            "method": "conditional_within_length_bins",
            "n_bins_used": 0,
            "bin_effects": [],
        }

    # Weighted average
    weights_used = bin_weights.iloc[bin_indices_used].values
    weights_used = weights_used / weights_used.sum()
    weighted_amie = float(np.average(bin_effects, weights=weights_used))

    return {
        "feature": "zipf",
        "group": group,
        "amie_ms": weighted_amie,
        "method": "conditional_within_length_bins",
        "n_bins_used": len(bin_effects),
        "bin_effects": bin_effects,
    }


# ======================================
# Pathway effects (descriptive helpers)
# ======================================


def compute_pathway_effects(
    ert_predictor, data: pd.DataFrame, feature: str, group: str, quartiles: Dict
) -> Dict:
    """
    Descriptive pathway effects for one feature and one group.
    Includes grid-based p_skip, TRT, ERT (Q1, Q3, delta).
    """
    q1 = quartiles[feature]["q1"]
    q3 = quartiles[feature]["q3"]

    group_data = data[data["group"] == group].copy()
    other_features = [f for f in ["length", "zipf", "surprisal"] if f != feature]
    means = {f: group_data[f].mean() for f in other_features}

    grid_q1 = pd.DataFrame({feature: [q1], **means})
    grid_q3 = pd.DataFrame({feature: [q3], **means})

    ert_q1, p_skip_q1, trt_q1 = ert_predictor.predict_ert(
        grid_q1, group, return_components=True
    )
    ert_q3, p_skip_q3, trt_q3 = ert_predictor.predict_ert(
        grid_q3, group, return_components=True
    )

    return {
        "feature": feature,
        "group": group,
        "p_skip_q1_grid": float(p_skip_q1[0]),
        "p_skip_q3_grid": float(p_skip_q3[0]),
        "delta_p_skip_grid": float(p_skip_q3[0] - p_skip_q1[0]),
        "trt_q1_grid": float(trt_q1[0]),
        "trt_q3_grid": float(trt_q3[0]),
        "delta_trt_ms_grid": float(trt_q3[0] - trt_q1[0]),
        "ert_q1_grid": float(ert_q1[0]),
        "ert_q3_grid": float(ert_q3[0]),
        "delta_ert_ms_grid": float(ert_q3[0] - ert_q1[0]),
    }


# =======================
# H1 main driver
# =======================


def test_hypothesis_1(
    ert_predictor,
    data: pd.DataFrame,
    quartiles: Dict,
    bin_edges: np.ndarray,
    bin_weights: pd.Series,
    n_bootstrap: int = 2000,
) -> Dict:
    """
    Test Hypothesis 1: Feature Effects
    Adds subject-level, bootstrap-based inference and correct Cohen's d using pooled SD.
    """
    logger.info("=" * 60)
    logger.info("HYPOTHESIS 1: FEATURE EFFECTS")
    logger.info("=" * 60)
    logger.info(f"Using {n_bootstrap} bootstraps for p-value estimation")
    print(f"\n=== H1: Feature Effects (n_bootstrap={n_bootstrap}) ===\n", flush=True)

    features = ["length", "zipf", "surprisal"]
    results = {}

    expected_directions = {"length": "+", "zipf": "-", "surprisal": "+"}

    for feature in features:
        logger.info(f"\nTesting {feature}...")
        logger.info(f"  Expected direction: {expected_directions[feature]}")

        # === 1) AMIE with statistics ===
        if feature == "zipf":
            amie_control = compute_amie_conditional_zipf(
                ert_predictor, data, "control", bin_edges, bin_weights
            )
            amie_dyslexic = compute_amie_conditional_zipf(
                ert_predictor, data, "dyslexic", bin_edges, bin_weights
            )

            logger.info(f"  Computing bootstrap tests for {feature} (conditional)...")
            print(f"  Testing {feature} (conditional)...", flush=True)
            stats_ctrl = bootstrap_amie(
                ert_predictor,
                data,
                feature,
                "control",
                quartiles,
                n_bootstrap=n_bootstrap,
                method="conditional",
                bin_edges=bin_edges,
                bin_weights=bin_weights,
            )
            stats_dys = bootstrap_amie(
                ert_predictor,
                data,
                feature,
                "dyslexic",
                quartiles,
                n_bootstrap=n_bootstrap,
                method="conditional",
                bin_edges=bin_edges,
                bin_weights=bin_weights,
            )
        else:
            amie_control = compute_amie_standard(
                ert_predictor, data, feature, "control", quartiles
            )
            amie_dyslexic = compute_amie_standard(
                ert_predictor, data, feature, "dyslexic", quartiles
            )

            logger.info(f"  Computing bootstrap tests for {feature} (standard)...")
            print(f"  Testing {feature} (standard)...", flush=True)
            stats_ctrl = bootstrap_amie(
                ert_predictor,
                data,
                feature,
                "control",
                quartiles,
                n_bootstrap=n_bootstrap,
                method="standard",
            )
            stats_dys = bootstrap_amie(
                ert_predictor,
                data,
                feature,
                "dyslexic",
                quartiles,
                n_bootstrap=n_bootstrap,
                method="standard",
            )

        # Merge AMIE stats into AMIE results
        amie_control.update(
            {
                "p_value": stats_ctrl["p_value"],
                "ci_low": stats_ctrl["ci_low"],
                "ci_high": stats_ctrl["ci_high"],
            }
        )
        amie_dyslexic.update(
            {
                "p_value": stats_dys["p_value"],
                "ci_low": stats_dys["ci_low"],
                "ci_high": stats_dys["ci_high"],
            }
        )

        # === 2) Pathway decomposition with subject-level bootstrap & proper d ===
        pathway_control = compute_pathway_effects(
            ert_predictor, data, feature, "control", quartiles
        )
        pathway_dyslexic = compute_pathway_effects(
            ert_predictor, data, feature, "dyslexic", quartiles
        )

        for pathway in ["skip", "duration", "ert"]:
            logger.info(f"    Testing {pathway} pathway...")

            stats_ctrl_path = bootstrap_pathway_effect(
                ert_predictor,
                data,
                feature,
                "control",
                quartiles,
                pathway,
                n_bootstrap,
                bin_edges=bin_edges if feature == "zipf" else None,
                bin_weights=bin_weights if feature == "zipf" else None,
            )
            stats_dys_path = bootstrap_pathway_effect(
                ert_predictor,
                data,
                feature,
                "dyslexic",
                quartiles,
                pathway,
                n_bootstrap,
                bin_edges=bin_edges if feature == "zipf" else None,
                bin_weights=bin_weights if feature == "zipf" else None,
            )

            # Inferential (subject-level) outputs:
            pathway_control[f"{pathway}_mean"] = stats_ctrl_path["mean"]
            pathway_control[f"{pathway}_p_value"] = stats_ctrl_path["p_value"]
            pathway_control[f"{pathway}_ci_low"] = stats_ctrl_path["ci_low"]
            pathway_control[f"{pathway}_ci_high"] = stats_ctrl_path["ci_high"]
            pathway_control[f"{pathway}_cohens_d"] = stats_ctrl_path.get(
                "cohens_d", np.nan
            )
            pathway_control[f"{pathway}_cohens_d_ci_low"] = stats_ctrl_path.get(
                "cohens_d_ci_low", np.nan
            )
            pathway_control[f"{pathway}_cohens_d_ci_high"] = stats_ctrl_path.get(
                "cohens_d_ci_high", np.nan
            )
            pathway_control[f"{pathway}_cohens_h"] = stats_ctrl_path.get(
                "skip_cohens_h", np.nan
            )

            pathway_dyslexic[f"{pathway}_mean"] = stats_dys_path["mean"]
            pathway_dyslexic[f"{pathway}_p_value"] = stats_dys_path["p_value"]
            pathway_dyslexic[f"{pathway}_ci_low"] = stats_dys_path["ci_low"]
            pathway_dyslexic[f"{pathway}_ci_high"] = stats_dys_path["ci_high"]
            pathway_dyslexic[f"{pathway}_cohens_d"] = stats_dys_path.get(
                "cohens_d", np.nan
            )
            pathway_dyslexic[f"{pathway}_cohens_d_ci_low"] = stats_dys_path.get(
                "cohens_d_ci_low", np.nan
            )
            pathway_dyslexic[f"{pathway}_cohens_d_ci_high"] = stats_dys_path.get(
                "cohens_d_ci_high", np.nan
            )
            pathway_dyslexic[f"{pathway}_cohens_h"] = stats_dys_path.get(
                "skip_cohens_h", np.nan
            )

        # === 3) Direction check (based on AMIE point estimate) ===
        expected = expected_directions[feature]
        ctrl_amie = amie_control.get("amie_ms", np.nan)
        dys_amie = amie_dyslexic.get("amie_ms", np.nan)

        ctrl_correct = (ctrl_amie > 0) if expected == "+" else (ctrl_amie < 0)
        dys_correct = (dys_amie > 0) if expected == "+" else (dys_amie < 0)
        status = "CONFIRMED" if (ctrl_correct and dys_correct) else "NOT CONFIRMED"

        note = ""
        if feature == "zipf":
            note = "Zipf uses conditional evaluation within length bins (AMIE + pathway inference)."

        results[feature] = {
            "amie_control": amie_control,
            "amie_dyslexic": amie_dyslexic,
            "pathway_control": pathway_control,
            "pathway_dyslexic": pathway_dyslexic,
            "expected_direction": expected,
            "correct_direction": ctrl_correct and dys_correct,
            "status": status,
            "note": note,
        }

        logger.info(
            f"  AMIE Control: {ctrl_amie:.2f} ms (p={stats_ctrl['p_value']:.5f})"
        )
        logger.info(
            f"  AMIE Dyslexic: {dys_amie:.2f} ms (p={stats_dys['p_value']:.5f})"
        )
        logger.info(f"  Status: {status}")

    all_confirmed = all(r["status"] == "CONFIRMED" for r in results.values())
    overall_status = "CONFIRMED" if all_confirmed else "PARTIALLY CONFIRMED"

    logger.info(f"\nHYPOTHESIS 1: {overall_status}")

    return {
        "status": overall_status,
        "features": results,
        "summary": f"{'All' if all_confirmed else 'Some'} features show expected effects",
    }
