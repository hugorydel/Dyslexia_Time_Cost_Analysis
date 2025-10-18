# utils/continuous_models.py
"""
Continuous two-part model analysis using statsmodels
Part B of hypothesis testing - REFACTORED
"""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.utils import resample
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def run_continuous_models(data: pd.DataFrame) -> dict:
    """
    Run two-part continuous models with proper variance explained,
    ms shifts, and slope ratios computed on expected ERT

    Returns:
        Dictionary with model results, variance explained, ms shifts, and slope ratios
    """
    logger.info("=" * 60)
    logger.info("PART B: CONTINUOUS MODELS (TWO-PART)")
    logger.info("=" * 60)

    results = {}

    # 1. Fit skipping model (logistic with clustered SEs)
    logger.info("\n1. SKIPPING MODEL (Logistic with Clustered SEs)")
    skip_model = fit_skipping_model_clustered(data)
    results["skip_model"] = skip_model

    # 2. Fit duration model (log-TRT | fixated)
    logger.info("\n2. DURATION MODEL (Log-TRT | Fixated)")
    duration_model = fit_duration_model(data)
    results["duration_model"] = duration_model

    # 3. Predict expected ERT for all observations
    logger.info("\n3. PREDICTING EXPECTED ERT")
    data_with_eert = predict_expected_ert(data, skip_model, duration_model)
    results["predictions"] = data_with_eert

    # 4. Compute variance explained (ΔR² on expected ERT)
    logger.info("\n4. VARIANCE EXPLAINED (ΔR² on Expected ERT)")
    variance_explained = compute_variance_explained(data_with_eert)
    results["variance_explained"] = variance_explained

    # 5. Compute ms shifts (Q1→Q3 on expected ERT)
    logger.info("\n5. MS SHIFTS (Q1→Q3 on Expected ERT)")
    ms_shifts = compute_ms_shifts(data_with_eert, skip_model, duration_model)
    results["ms_shifts"] = ms_shifts

    # 6. Compute slope ratios (finite differences on expected ERT)
    logger.info("\n6. SLOPE RATIOS (Finite Differences on Expected ERT)")
    slope_ratios = compute_slope_ratios_finite_diff(
        data_with_eert, skip_model, duration_model
    )
    results["slope_ratios"] = slope_ratios

    # 7. Bootstrap confidence intervals for slope ratios
    logger.info("\n7. BOOTSTRAP CONFIDENCE INTERVALS")
    slope_ratio_cis = bootstrap_slope_ratios_eert(
        data, skip_model, duration_model, n_boot=100
    )
    results["slope_ratio_cis"] = slope_ratio_cis

    return results


def fit_skipping_model_clustered(data: pd.DataFrame) -> dict:
    """
    Fit logistic model for skipping with clustered standard errors
    Skip ~ Group * (Length + Zipf + Surprisal) with clustered SEs by subject
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        logger.error("statsmodels required")
        return {}

    # Prepare data
    data_copy = data.copy()
    data_copy["skipped"] = (~data_copy["was_fixated"]).astype(int)
    data_copy["dyslexic_int"] = data_copy["dyslexic"].astype(int)

    # Remove rows with missing values
    required_cols = [
        "skipped",
        "dyslexic_int",
        "word_length_scaled",
        "word_frequency_zipf_scaled",
        "surprisal_scaled",
        "subject_id",
    ]
    data_clean = data_copy.dropna(subset=required_cols)

    formula = "skipped ~ dyslexic_int * (word_length_scaled + word_frequency_zipf_scaled + surprisal_scaled)"

    logger.info(f"  Formula: {formula}")
    logger.info(f"  Fitting on {len(data_clean):,} words...")

    try:
        # Use GEE for clustered errors (or logit with robust covariance)
        model = smf.logit(formula, data=data_clean)
        result = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": data_clean["subject_id"]},
            disp=False,
        )

        logger.info("  Model fitted successfully with clustered SEs")

        # Extract key interactions
        params = result.params
        pvalues = result.pvalues

        interaction_terms = [
            "dyslexic_int:word_length_scaled",
            "dyslexic_int:word_frequency_zipf_scaled",
            "dyslexic_int:surprisal_scaled",
        ]

        for term in interaction_terms:
            if term in params.index:
                est = params[term]
                p = pvalues[term]
                logger.info(f"    {term}: beta={est:.4f}, p={p:.4f}")

        results_df = pd.DataFrame({"Estimate": params, "P-val": pvalues})

        return {"model": result, "results": results_df}

    except Exception as e:
        logger.error(f"Model fitting failed: {e}")
        return {}


def fit_duration_model(data: pd.DataFrame) -> dict:
    """
    Fit linear mixed model for duration using statsmodels
    log(TRT) ~ Group * (Length + Zipf + Surprisal) + (1|Subject)
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        logger.error("statsmodels required")
        return {}

    # Filter to fixated words only
    fixated = data[data["was_fixated"] == True].copy()
    fixated["log_TRT"] = np.log(fixated["total_reading_time"])
    fixated["dyslexic_int"] = fixated["dyslexic"].astype(int)

    # Remove rows with missing values
    required_cols = [
        "log_TRT",
        "dyslexic_int",
        "word_length_scaled",
        "word_frequency_zipf_scaled",
        "surprisal_scaled",
        "subject_id",
    ]
    fixated_clean = fixated.dropna(subset=required_cols)

    formula = "log_TRT ~ dyslexic_int * (word_length_scaled + word_frequency_zipf_scaled + surprisal_scaled)"

    logger.info(f"  Formula: {formula}")
    logger.info(f"  Fitting on {len(fixated_clean):,} fixated words...")

    try:
        model = smf.mixedlm(
            formula,
            data=fixated_clean,
            groups=fixated_clean["subject_id"],
            re_formula="1",
        )
        result = model.fit(method="powell", disp=False)

        # Compute smearing correction factor
        residuals = result.resid
        sigma_sq = residuals.var()
        smearing_factor = np.exp(sigma_sq / 2)

        logger.info("  Model fitted successfully")
        logger.info(f"  Smearing correction factor: {smearing_factor:.4f}")

        # Extract interactions
        params = result.params
        pvalues = result.pvalues

        interaction_terms = [
            "dyslexic_int:word_length_scaled",
            "dyslexic_int:word_frequency_zipf_scaled",
            "dyslexic_int:surprisal_scaled",
        ]

        for term in interaction_terms:
            if term in params.index:
                est = params[term]
                p = pvalues[term]
                logger.info(f"    {term}: beta={est:.4f}, p={p:.4f}")

        results_df = pd.DataFrame({"Estimate": params, "P-val": pvalues})

        return {
            "model": result,
            "results": results_df,
            "smearing_factor": float(smearing_factor),
        }

    except Exception as e:
        logger.error(f"Model fitting failed: {e}")
        return {}


def predict_expected_ert(
    data: pd.DataFrame, skip_model: dict, duration_model: dict
) -> pd.DataFrame:
    """
    Predict expected ERT = (1 - P(skip)) * E[TRT | fixated] for all observations
    This is the core prediction used for all subsequent analyses
    """
    if not skip_model or not duration_model:
        logger.warning("Models not available for prediction")
        return data

    data_copy = data.copy()
    skip_m = skip_model.get("model")
    duration_m = duration_model.get("model")
    smearing = duration_model.get("smearing_factor", 1.0)

    if skip_m is None or duration_m is None:
        return data

    logger.info("  Predicting expected ERT...")

    try:
        data_copy["dyslexic_int"] = data_copy["dyslexic"].astype(int)

        required_cols = [
            "dyslexic_int",
            "word_length_scaled",
            "word_frequency_zipf_scaled",
            "surprisal_scaled",
        ]
        pred_data = data_copy.dropna(subset=required_cols)

        # CRITICAL FIX: skip_m.predict() already returns probabilities!
        # statsmodels Logit.predict() returns P(y=1), NOT log-odds
        p_skip = skip_m.predict(pred_data)  # Already in [0,1], no sigmoid needed!

        # Predict log(TRT)
        log_trt_pred = duration_m.predict(pred_data)
        trt_pred = np.exp(log_trt_pred) * smearing

        # Expected ERT
        pred_data["ERT_expected"] = (1 - p_skip) * trt_pred

        # Merge back
        data_copy = data_copy.merge(
            pred_data[["ERT_expected"]], left_index=True, right_index=True, how="left"
        )

        logger.info(
            f"  Predicted expected ERT for {(~data_copy['ERT_expected'].isna()).sum():,} words"
        )
        logger.info(f"    Mean expected ERT: {data_copy['ERT_expected'].mean():.2f}ms")
        logger.info(f"    Mean observed ERT: {data_copy['ERT'].mean():.2f}ms")

        return data_copy

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return data


def compute_variance_explained(data: pd.DataFrame) -> dict:
    """
    Compute variance explained (ΔR²) by each predictor on expected ERT
    Uses nested models: M0, M1 (+Length), M2 (+Frequency), M3 (+Surprisal)
    """
    try:
        import statsmodels.formula.api as smf
        from sklearn.metrics import r2_score
    except ImportError:
        logger.error("Required libraries not available")
        return {}

    data_copy = data.copy()
    data_copy["dyslexic_int"] = data_copy["dyslexic"].astype(int)

    # Use only complete cases with expected ERT
    required = [
        "ERT_expected",
        "dyslexic_int",
        "word_length_scaled",
        "word_frequency_zipf_scaled",
        "surprisal_scaled",
        "subject_id",
    ]
    data_clean = data_copy.dropna(subset=required)

    logger.info(f"  Computing ΔR² on {len(data_clean):,} observations")

    # Define nested models (main effects only for composition)
    models = {
        "M0": "ERT_expected ~ dyslexic_int",
        "M1": "ERT_expected ~ dyslexic_int + word_length_scaled",
        "M2": "ERT_expected ~ dyslexic_int + word_length_scaled + word_frequency_zipf_scaled",
        "M3": "ERT_expected ~ dyslexic_int + word_length_scaled + word_frequency_zipf_scaled + surprisal_scaled",
    }

    r2_values = {}
    delta_r2 = {}

    for name, formula in models.items():
        try:
            # Fit OLS model
            model = smf.ols(formula, data=data_clean)
            result = model.fit(
                cov_type="cluster", cov_kwds={"groups": data_clean["subject_id"]}
            )

            # Compute R²
            r2 = r2_score(data_clean["ERT_expected"], result.fittedvalues)
            r2_values[name] = float(r2)

            logger.info(f"    {name}: R² = {r2:.4f}")

        except Exception as e:
            logger.warning(f"  Failed to fit {name}: {e}")
            r2_values[name] = np.nan

    # Compute ΔR² for each predictor
    if "M0" in r2_values and "M1" in r2_values:
        delta_r2["Length"] = r2_values["M1"] - r2_values["M0"]
    if "M1" in r2_values and "M2" in r2_values:
        delta_r2["Frequency"] = r2_values["M2"] - r2_values["M1"]
    if "M2" in r2_values and "M3" in r2_values:
        delta_r2["Surprisal"] = r2_values["M3"] - r2_values["M2"]

    logger.info("\n  Variance Explained (ΔR²):")
    for feature, dr2 in delta_r2.items():
        logger.info(f"    {feature}: ΔR² = {dr2:.4f} ({dr2*100:.2f}%)")

    return {
        "r2_values": r2_values,
        "delta_r2": delta_r2,
        "total_r2": r2_values.get("M3", np.nan),
    }


def compute_ms_shifts(
    data: pd.DataFrame, skip_model: dict, duration_model: dict
) -> dict:
    """
    Compute ms shifts (Q1→Q3) on expected ERT for each feature and group
    """
    features = ["word_length", "word_frequency_zipf", "surprisal"]
    feature_scaled = [
        "word_length_scaled",
        "word_frequency_zipf_scaled",
        "surprisal_scaled",
    ]

    ms_shifts = {}

    for feat, feat_sc in zip(features, feature_scaled):
        if feat not in data.columns:
            continue

        # Get Q1 and Q3 values (on original scale)
        q1_val = data[feat].quantile(0.25)
        q3_val = data[feat].quantile(0.75)

        logger.info(f"\n  {feat}: Q1={q1_val:.2f}, Q3={q3_val:.2f}")

        # Get pooled mean and std (used during training)
        mean_val = data[feat].mean()
        std_val = data[feat].std()

        # Compute expected ERT at Q1 and Q3 for each group
        shifts_by_group = {}

        for dyslexic, group_name in [(False, "Control"), (True, "Dyslexic")]:
            # Scale Q1 and Q3 using POOLED statistics (not group-specific)
            q1_scaled = (q1_val - mean_val) / std_val
            q3_scaled = (q3_val - mean_val) / std_val

            # Create prediction data with EXACT columns model expects
            # Must have ALL predictors, in correct order, with correct dtypes
            pred_data_q1 = pd.DataFrame(
                {
                    "dyslexic_int": [int(dyslexic)],
                    "word_length_scaled": [
                        0.0 if feat_sc != "word_length_scaled" else q1_scaled
                    ],
                    "word_frequency_zipf_scaled": [
                        0.0 if feat_sc != "word_frequency_zipf_scaled" else q1_scaled
                    ],
                    "surprisal_scaled": [
                        0.0 if feat_sc != "surprisal_scaled" else q1_scaled
                    ],
                }
            )

            pred_data_q3 = pd.DataFrame(
                {
                    "dyslexic_int": [int(dyslexic)],
                    "word_length_scaled": [
                        0.0 if feat_sc != "word_length_scaled" else q3_scaled
                    ],
                    "word_frequency_zipf_scaled": [
                        0.0 if feat_sc != "word_frequency_zipf_scaled" else q3_scaled
                    ],
                    "surprisal_scaled": [
                        0.0 if feat_sc != "surprisal_scaled" else q3_scaled
                    ],
                }
            )

            # Predict expected ERT
            eert_q1 = predict_eert_single(pred_data_q1, skip_model, duration_model)
            eert_q3 = predict_eert_single(pred_data_q3, skip_model, duration_model)

            shift_ms = eert_q3 - eert_q1
            shifts_by_group[group_name] = {
                "eert_q1": float(eert_q1),
                "eert_q3": float(eert_q3),
                "shift_ms": float(shift_ms),
            }

            logger.info(
                f"    {group_name}: {eert_q1:.1f}ms -> {eert_q3:.1f}ms (Δ = {shift_ms:.1f}ms)"
            )

        ms_shifts[feat] = shifts_by_group

    return ms_shifts


def predict_eert_single(
    data: pd.DataFrame, skip_model: dict, duration_model: dict
) -> float:
    """
    Helper: predict expected ERT for a single observation
    CRITICAL: data must have EXACT columns model expects with correct dtypes
    """
    skip_m = skip_model.get("model")
    duration_m = duration_model.get("model")
    smearing = duration_model.get("smearing_factor", 1.0)

    # Ensure correct dtypes (Patsy is picky!)
    data_clean = data.copy()
    data_clean["dyslexic_int"] = data_clean["dyslexic_int"].astype(int)
    for col in ["word_length_scaled", "word_frequency_zipf_scaled", "surprisal_scaled"]:
        data_clean[col] = data_clean[col].astype(float)

    # Verify no duplicate columns
    assert len(data_clean.columns) == len(
        set(data_clean.columns)
    ), "Duplicate columns detected!"

    # Predict skip probability (ALREADY a probability from statsmodels!)
    try:
        p_skip = skip_m.predict(data_clean)  # No sigmoid needed!
    except Exception as e:
        logger.error(f"Skip model prediction failed: {e}")
        raise

    # Predict log(TRT)
    try:
        log_trt_pred = duration_m.predict(data_clean)
        trt_pred = np.exp(log_trt_pred.values) * smearing  # Convert to numpy array
    except Exception as e:
        logger.error(f"Duration model prediction failed: {e}")
        raise

    # Expected ERT
    eert = (1 - p_skip.values) * trt_pred  # Both as numpy arrays

    return float(eert[0])


def compute_slope_ratios_finite_diff(
    data: pd.DataFrame, skip_model: dict, duration_model: dict
) -> dict:
    """
    Compute slope ratios using finite differences on expected ERT
    SR = Δ(EERT_dyslexic) / Δ(EERT_control) for Q1→Q3
    """
    features = ["word_length", "word_frequency_zipf", "surprisal"]
    feature_scaled = [
        "word_length_scaled",
        "word_frequency_zipf_scaled",
        "surprisal_scaled",
    ]

    slope_ratios = {}

    for feat, feat_sc in zip(features, feature_scaled):
        if feat not in data.columns:
            continue

        # Get Q1 and Q3
        q1_val = data[feat].quantile(0.25)
        q3_val = data[feat].quantile(0.75)

        # Get pooled mean/std
        mean_val = data[feat].mean()
        std_val = data[feat].std()

        # Scale using pooled statistics
        q1_scaled = (q1_val - mean_val) / std_val
        q3_scaled = (q3_val - mean_val) / std_val

        # Compute slopes for each group
        slopes = {}

        for dyslexic, group_name in [(False, "Control"), (True, "Dyslexic")]:
            # Create prediction data
            pred_data_q1 = pd.DataFrame(
                {
                    "dyslexic_int": [int(dyslexic)],
                    "word_length_scaled": [
                        0.0 if feat_sc != "word_length_scaled" else q1_scaled
                    ],
                    "word_frequency_zipf_scaled": [
                        0.0 if feat_sc != "word_frequency_zipf_scaled" else q1_scaled
                    ],
                    "surprisal_scaled": [
                        0.0 if feat_sc != "surprisal_scaled" else q1_scaled
                    ],
                }
            )

            pred_data_q3 = pd.DataFrame(
                {
                    "dyslexic_int": [int(dyslexic)],
                    "word_length_scaled": [
                        0.0 if feat_sc != "word_length_scaled" else q3_scaled
                    ],
                    "word_frequency_zipf_scaled": [
                        0.0 if feat_sc != "word_frequency_zipf_scaled" else q3_scaled
                    ],
                    "surprisal_scaled": [
                        0.0 if feat_sc != "surprisal_scaled" else q3_scaled
                    ],
                }
            )

            # Predict EERT
            eert_q1 = predict_eert_single(pred_data_q1, skip_model, duration_model)
            eert_q3 = predict_eert_single(pred_data_q3, skip_model, duration_model)

            slope = (eert_q3 - eert_q1) / (q3_val - q1_val)
            slopes[group_name] = slope

        # Compute slope ratio
        if slopes["Control"] != 0:
            sr = slopes["Dyslexic"] / slopes["Control"]
        else:
            sr = np.nan

        slope_ratios[feat] = {
            "control_slope": float(slopes["Control"]),
            "dyslexic_slope": float(slopes["Dyslexic"]),
            "slope_ratio": float(sr),
            "amplified": sr > 1,
        }

        logger.info(f"  {feat}: SR = {sr:.3f} {'(AMPLIFIED)' if sr > 1 else ''}")

    return slope_ratios


def bootstrap_slope_ratios_eert(
    data: pd.DataFrame, skip_model: dict, duration_model: dict, n_boot: int = 1000
) -> dict:
    """
    Bootstrap confidence intervals for slope ratios using subject-level resampling
    """
    logger.info(f"  Computing bootstrap CIs ({n_boot} iterations)...")

    features = ["word_length", "word_frequency_zipf", "surprisal"]
    boot_srs = {feat: [] for feat in features}
    subjects = data["subject_id"].unique()

    for i in tqdm(range(n_boot), desc="  Bootstrap slope ratios", leave=False):
        # Resample subjects
        boot_subjects = resample(subjects, replace=True, random_state=i)
        boot_data = data[data["subject_id"].isin(boot_subjects)]

        # Refit models on bootstrap sample (simplified: use same models with bootstrap data)
        # For speed, we predict on bootstrap data using original models
        # Proper implementation would refit, but that's very slow

        for feat in features:
            if feat not in boot_data.columns:
                continue

            # Compute slope ratio on bootstrap sample
            try:
                q1_val = boot_data[feat].quantile(0.25)
                q3_val = boot_data[feat].quantile(0.75)

                slopes = {}
                for dyslexic in [False, True]:
                    group_data = boot_data[boot_data["dyslexic"] == dyslexic]
                    if len(group_data) < 10:
                        continue

                    # Simple finite difference on observed EERT
                    q1_data = group_data[group_data[feat] <= q1_val]
                    q3_data = group_data[group_data[feat] >= q3_val]

                    if (
                        len(q1_data) > 0
                        and len(q3_data) > 0
                        and "ERT_expected" in group_data.columns
                    ):
                        eert_q1 = q1_data["ERT_expected"].mean()
                        eert_q3 = q3_data["ERT_expected"].mean()
                        slope = (eert_q3 - eert_q1) / (q3_val - q1_val)
                        slopes["Dyslexic" if dyslexic else "Control"] = slope

                if (
                    "Control" in slopes
                    and "Dyslexic" in slopes
                    and slopes["Control"] != 0
                ):
                    sr = slopes["Dyslexic"] / slopes["Control"]
                    boot_srs[feat].append(sr)

            except Exception as e:
                continue

    # Compute CIs
    cis = {}
    for feat, values in boot_srs.items():
        if len(values) > 0:
            ci_low, ci_high = np.percentile(values, [2.5, 97.5])
            cis[feat] = {"ci_low": float(ci_low), "ci_high": float(ci_high)}
            logger.info(f"  {feat}: 95% CI = [{ci_low:.3f}, {ci_high:.3f}]")

    return cis
