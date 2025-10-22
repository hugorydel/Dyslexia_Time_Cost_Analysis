"""
Hypothesis 1: Feature Effects with CONDITIONAL Zipf Evaluation
FULLY REVISED: Added p-values and comprehensive statistics for all effects
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ======================
# Effect size utilities
# ======================


def cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h for two proportions.
    h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
    """
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def hedges_J(n: int) -> float:
    """Small-sample correction for d (Hedges' J)."""
    return 1 - 3 / (4 * n - 9) if n and n > 2 else 1.0


def cohens_d(mean_diff: float, std_dev: float) -> float:
    """
    DEPRECATED: Formerly divided the mean difference by the *bootstrap SD of effects* (≈ SE),
    which yields a t-/z-like statistic, not Cohen's d. Kept for backwards compatibility.
    """
    if std_dev == 0 or np.isnan(std_dev):
        return np.nan
    return mean_diff / std_dev


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


# ===========================
# AMIE (unchanged inference)
# ===========================


def permutation_test_amie(
    ert_predictor,
    data: pd.DataFrame,
    feature: str,
    group: str,
    quartiles: Dict,
    n_permutations: int = 1000,
    method: str = "standard",
    bin_edges: np.ndarray = None,
    bin_weights: pd.Series = None,
) -> Dict:
    """
    Bootstrap test for AMIE significance (resample subjects).
    Returns p-value, 95% CI, and observed mean AMIE.
    """
    # Observed AMIE
    if method == "conditional":
        observed_result = compute_amie_conditional_zipf(
            ert_predictor, data, group, quartiles, bin_edges, bin_weights
        )
        observed_amie = observed_result.get("amie_ms", np.nan)
    else:
        observed_result = compute_amie_standard(
            ert_predictor, data, feature, group, quartiles
        )
        observed_amie = observed_result.get("amie_ms", np.nan)

    if np.isnan(observed_amie):
        return {
            "p_value": np.nan,
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_bootstrap": 0,
        }

    subjects = data["subject_id"].unique()
    n_subjects = len(subjects)
    bootstrap_amies = []

    print(f"      Bootstrap AMIE ({feature}, {group}, {method})...", flush=True)

    for i in tqdm(
        range(n_permutations), desc=f"      AMIE {feature[:3]}-{group[:3]}", leave=False
    ):
        rng = np.random.RandomState(1000 + i)
        boot_subjects = rng.choice(subjects, size=n_subjects, replace=True)
        boot_data = pd.concat(
            [data[data["subject_id"] == s] for s in boot_subjects], ignore_index=True
        )
        if len(boot_data) < 500:
            continue

        boot_quartiles = {
            feat: {
                "q1": boot_data[feat].quantile(0.25),
                "q3": boot_data[feat].quantile(0.75),
                "iqr": boot_data[feat].quantile(0.75) - boot_data[feat].quantile(0.25),
            }
            for feat in ["length", "zipf", "surprisal"]
        }

        try:
            if method == "conditional":
                result = compute_amie_conditional_zipf(
                    ert_predictor,
                    boot_data,
                    group,
                    boot_quartiles,
                    bin_edges,
                    bin_weights,
                )
            else:
                result = compute_amie_standard(
                    ert_predictor, boot_data, feature, group, boot_quartiles
                )
            amie = result.get("amie_ms", np.nan)
            if not np.isnan(amie):
                bootstrap_amies.append(amie)
        except Exception:
            pass

    if len(bootstrap_amies) > 0:
        ci_low = float(np.percentile(bootstrap_amies, 2.5))
        ci_high = float(np.percentile(bootstrap_amies, 97.5))

        if observed_amie > 0:
            p_value = float(np.mean(np.array(bootstrap_amies) <= 0))
        else:
            p_value = float(np.mean(np.array(bootstrap_amies) >= 0))
        p_value = min(1.0, 2 * p_value)
    else:
        ci_low = ci_high = p_value = np.nan

    return {
        "p_value": float(p_value),
        "mean": float(observed_amie),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_bootstrap": len(bootstrap_amies),
    }


# =====================================
# Pathway effects (with corrected d_z)
# =====================================


def permutation_test_pathway_effect(
    ert_predictor,
    data: pd.DataFrame,
    feature: str,
    group: str,
    quartiles: Dict,
    pathway: str,  # "skip", "duration", or "ert"
    n_permutations: int = 1000,
    bin_edges: Optional[np.ndarray] = None,
    bin_weights: Optional[pd.Series] = None,
) -> Dict:
    """
    BOOTSTRAP (resample subjects) test for pathway effects.

    - Observed "mean" is the **subject-mean delta** (Q3-Q1) for the requested pathway.
    - Cohen's d is **within-subject d_z** with Hedges' J.
    - For Zipf, if bin_edges/bin_weights are provided, uses *conditional-in-bins*
      subject-level deltas weighted by bin_weights (for consistency with conditional AMIE).
    - For skip pathway: report CI/p for the subject-mean delta (probability difference).
      Cohen's h is computed separately in compute_pathway_effects (descriptive).
    """
    q1 = quartiles[feature]["q1"]
    q3 = quartiles[feature]["q3"]

    # ---------- Observed (subject-level) ----------
    deltas = subject_level_deltas(
        ert_predictor,
        data,
        feature,
        group,
        q1,
        q3,
        pathway,
        bin_edges=bin_edges if feature == "zipf" else None,
        bin_weights=bin_weights if feature == "zipf" else None,
    )
    deltas = deltas[~np.isnan(deltas)]
    if deltas.size < 2:
        return {
            "p_value": np.nan,
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_bootstrap": 0,
            "std_dev": np.nan,
            "cohens_d": np.nan,
            "cohens_d_ci_low": np.nan,
            "cohens_d_ci_high": np.nan,
        }

    observed_mean = float(np.mean(deltas))
    sd_delta = float(np.std(deltas, ddof=1))
    d_z = hedges_J(deltas.size) * (observed_mean / sd_delta)

    # ---------- Bootstrap (resample subjects) ----------
    subjects = data["subject_id"].unique()
    n_subjects = len(subjects)
    bootstrap_means = []
    bootstrap_ds = []

    print(f"      Bootstrap {pathway} ({feature}, {group})...", flush=True)

    for i in tqdm(
        range(n_permutations),
        desc=f"      {pathway[:3]} {feature[:3]}-{group[:3]}",
        leave=False,
    ):
        rng = np.random.RandomState(2000 + i)
        boot_sids = rng.choice(subjects, size=n_subjects, replace=True)
        boot_data = pd.concat(
            [data[data["subject_id"] == s] for s in boot_sids], ignore_index=True
        )
        if len(boot_data) < 500:
            continue

        # Recompute quartiles (and bin zipf quartiles implicitly through subject_level_deltas)
        boot_quartiles = {
            feat: {
                "q1": boot_data[feat].quantile(0.25),
                "q3": boot_data[feat].quantile(0.75),
                "iqr": boot_data[feat].quantile(0.75) - boot_data[feat].quantile(0.25),
            }
            for feat in ["length", "zipf", "surprisal"]
        }

        try:
            boot_deltas = subject_level_deltas(
                ert_predictor,
                boot_data,
                feature,
                group,
                boot_quartiles[feature]["q1"],
                boot_quartiles[feature]["q3"],
                pathway,
                bin_edges=bin_edges if feature == "zipf" else None,
                bin_weights=bin_weights if feature == "zipf" else None,
            )
            boot_deltas = boot_deltas[~np.isnan(boot_deltas)]
            if boot_deltas.size < 2:
                continue

            m = float(np.mean(boot_deltas))
            s = float(np.std(boot_deltas, ddof=1))
            d_boot = hedges_J(boot_deltas.size) * (m / s)

            bootstrap_means.append(m)
            bootstrap_ds.append(d_boot)
        except Exception:
            pass

    if len(bootstrap_means) > 0:
        ci_low = float(np.percentile(bootstrap_means, 2.5))
        ci_high = float(np.percentile(bootstrap_means, 97.5))
        # Two-tailed p for the mean delta:
        if observed_mean > 0:
            p_value = float(np.mean(np.array(bootstrap_means) <= 0))
        else:
            p_value = float(np.mean(np.array(bootstrap_means) >= 0))
        p_value = min(1.0, 2 * p_value)

        # CI for d_z (optional but useful)
        d_ci_low = (
            float(np.percentile(bootstrap_ds, 2.5)) if len(bootstrap_ds) > 0 else np.nan
        )
        d_ci_high = (
            float(np.percentile(bootstrap_ds, 97.5))
            if len(bootstrap_ds) > 0
            else np.nan
        )

        std_dev = float(np.std(bootstrap_means, ddof=1))
    else:
        ci_low = ci_high = p_value = std_dev = np.nan
        d_ci_low = d_ci_high = np.nan

    return {
        "p_value": float(p_value),
        "mean": float(observed_mean),  # subject-mean delta (matches d_z numerator)
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_bootstrap": len(bootstrap_means),
        "std_dev": (
            float(std_dev) if not np.isnan(std_dev) else np.nan
        ),  # SD across boot means
        "cohens_d": float(d_z) if np.isfinite(d_z) else np.nan,
        "cohens_d_ci_low": float(d_ci_low),
        "cohens_d_ci_high": float(d_ci_high),
    }


# ==================================
# AMIE point estimators (unchanged)
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

    other_features = [f for f in ["length", "zipf", "surprisal"] if f != feature]
    means = {f: data[f].mean() for f in other_features}

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
    quartiles: Dict,
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

        effect = ert_q3 - ert_q1
        bin_effects.append(effect)
        bin_indices_used.append(bin_idx)

    if len(bin_effects) == 0:
        return {"error": "No valid bins"}

    weights_used = bin_weights.iloc[bin_indices_used].values
    weights_normalized = weights_used / weights_used.sum()
    amie_conditional = np.average(bin_effects, weights=weights_normalized)

    return {
        "feature": "zipf",
        "group": group,
        "amie_ms": float(amie_conditional),
        "method": "conditional_within_length_bins",
        "n_bins_used": len(bin_effects),
        "bin_effects": [float(e) for e in bin_effects],
    }


# ==============================
# Pathway (descriptive snapshot)
# ==============================


def compute_pathway_effects(
    ert_predictor, data: pd.DataFrame, feature: str, group: str, quartiles: Dict
) -> Dict:
    """
    Descriptive snapshot at group means (grid). Inferential stats are added later
    from permutation_test_pathway_effect (which uses subject-level deltas).
    """
    q1 = quartiles[feature]["q1"]
    q3 = quartiles[feature]["q3"]

    other_features = [f for f in ["length", "zipf", "surprisal"] if f != feature]
    means = {f: data[f].mean() for f in other_features}

    grid_q1 = pd.DataFrame({feature: [q1], **means})
    grid_q3 = pd.DataFrame({feature: [q3], **means})

    ert_q1, p_skip_q1, trt_q1 = ert_predictor.predict_ert(
        grid_q1, group, return_components=True
    )
    ert_q3, p_skip_q3, trt_q3 = ert_predictor.predict_ert(
        grid_q3, group, return_components=True
    )

    h = cohens_h(p_skip_q1[0], p_skip_q3[0])  # descriptive

    return {
        "feature": feature,
        "group": group,
        # descriptive (grid) values:
        "p_skip_q1_grid": float(p_skip_q1[0]),
        "p_skip_q3_grid": float(p_skip_q3[0]),
        "delta_p_skip_grid": float(p_skip_q3[0] - p_skip_q1[0]),
        "cohens_h": float(h),
        "trt_q1_grid": float(trt_q1[0]),
        "trt_q3_grid": float(trt_q3[0]),
        "delta_trt_ms_grid": float(trt_q3[0] - trt_q1[0]),
        "ert_q1_grid": float(ert_q1[0]),
        "ert_q3_grid": float(ert_q3[0]),
        "delta_ert_ms_grid": float(ert_q3[0] - ert_q1[0]),
        # inferential fields to be populated later:
        "cohens_d_trt": np.nan,
        "cohens_d_ert": np.nan,
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
    n_permutations: int = 200,  # Reduced default for speed
) -> Dict:
    """
    Test Hypothesis 1: Feature Effects
    Adds subject-level, bootstrap-based inference and correct Cohen's d (d_z with J).
    """
    logger.info("=" * 60)
    logger.info("HYPOTHESIS 1: FEATURE EFFECTS")
    logger.info("=" * 60)
    logger.info(f"Using {n_permutations} permutations for p-value estimation")
    print(f"\n=== H1: Feature Effects (n_perm={n_permutations}) ===\n", flush=True)

    features = ["length", "zipf", "surprisal"]
    results = {}

    expected_directions = {"length": "+", "zipf": "-", "surprisal": "+"}

    for feature in features:
        logger.info(f"\nTesting {feature}...")
        logger.info(f"  Expected direction: {expected_directions[feature]}")

        # === 1) AMIE with statistics ===
        if feature == "zipf":
            amie_control = compute_amie_conditional_zipf(
                ert_predictor, data, "control", quartiles, bin_edges, bin_weights
            )
            amie_dyslexic = compute_amie_conditional_zipf(
                ert_predictor, data, "dyslexic", quartiles, bin_edges, bin_weights
            )

            logger.info(f"  Computing bootstrap tests for {feature} (conditional)...")
            print(f"  Testing {feature} (conditional)...", flush=True)
            stats_ctrl = permutation_test_amie(
                ert_predictor,
                data,
                feature,
                "control",
                quartiles,
                n_permutations=n_permutations,
                method="conditional",
                bin_edges=bin_edges,
                bin_weights=bin_weights,
            )
            stats_dys = permutation_test_amie(
                ert_predictor,
                data,
                feature,
                "dyslexic",
                quartiles,
                n_permutations=n_permutations,
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
            stats_ctrl = permutation_test_amie(
                ert_predictor,
                data,
                feature,
                "control",
                quartiles,
                n_permutations=n_permutations,
                method="standard",
            )
            stats_dys = permutation_test_amie(
                ert_predictor,
                data,
                feature,
                "dyslexic",
                quartiles,
                n_permutations=n_permutations,
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

            stats_ctrl_path = permutation_test_pathway_effect(
                ert_predictor,
                data,
                feature,
                "control",
                quartiles,
                pathway,
                n_permutations,
                bin_edges=bin_edges if feature == "zipf" else None,
                bin_weights=bin_weights if feature == "zipf" else None,
            )
            stats_dys_path = permutation_test_pathway_effect(
                ert_predictor,
                data,
                feature,
                "dyslexic",
                quartiles,
                pathway,
                n_permutations,
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

        # Convenience mirrors:
        pathway_control["cohens_d_trt"] = pathway_control.get(
            "duration_cohens_d", np.nan
        )
        pathway_control["cohens_d_ert"] = pathway_control.get("ert_cohens_d", np.nan)
        pathway_dyslexic["cohens_d_trt"] = pathway_dyslexic.get(
            "duration_cohens_d", np.nan
        )
        pathway_dyslexic["cohens_d_ert"] = pathway_dyslexic.get("ert_cohens_d", np.nan)

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
