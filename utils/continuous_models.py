# utils/continuous_models.py
"""
Continuous two-part model analysis using statsmodels
Part B of hypothesis testing
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
    Run two-part continuous models:
    1. Skipping model (logistic)
    2. Duration model (log-TRT | fixated)
    3. Recombine to Expected ERT

    Returns:
        Dictionary with model results
    """
    logger.info("=" * 60)
    logger.info("PART B: CONTINUOUS MODELS (TWO-PART)")
    logger.info("=" * 60)

    results = {}

    # 1. Skipping model
    logger.info("\n1. SKIPPING MODEL (Logistic)")
    skip_model = fit_skipping_model(data)
    results["skip_model"] = skip_model

    # 2. Duration model (on fixated words only)
    logger.info("\n2. DURATION MODEL (Log-TRT | Fixated)")
    duration_model = fit_duration_model(data)
    results["duration_model"] = duration_model

    # 3. Extract slope ratios
    logger.info("\n3. SLOPE RATIOS (Dyslexic Amplification)")
    slope_ratios = extract_slope_ratios(skip_model, duration_model)
    results["slope_ratios"] = slope_ratios

    # 4. Bootstrap SEs for slope ratios
    logger.info("\n4. BOOTSTRAP CONFIDENCE INTERVALS")
    slope_ratio_cis = bootstrap_slope_ratios(data, n_boot=100)  # Reduced for speed
    results["slope_ratio_cis"] = slope_ratio_cis

    # 5. Recombine to expected ERT
    logger.info("\n5. RECOMBINING TO EXPECTED ERT")
    data_with_pred = predict_expected_ert(data, skip_model, duration_model)
    results["predictions"] = data_with_pred

    return results


def fit_skipping_model(data: pd.DataFrame) -> dict:
    """
    Fit logistic mixed model for skipping using statsmodels
    Skip ~ Group * (Length + Zipf + Surprisal) + (1|Subject)
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        logger.error("statsmodels required. Install with: pip install statsmodels")
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
        model = smf.logit(formula, data=data_clean).fit(disp=False)

        logger.info("  Model fitted successfully")

        # Extract key interactions
        params = model.params
        pvalues = model.pvalues

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

        # Create results DataFrame in expected format
        results_df = pd.DataFrame({"Estimate": params, "P-val": pvalues})

        return {"model": model, "results": results_df}

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
        # Use mixed linear model with subject random effect
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

        # Create results DataFrame
        results_df = pd.DataFrame({"Estimate": params, "P-val": pvalues})

        return {
            "model": result,
            "results": results_df,
            "smearing_factor": float(smearing_factor),
        }

    except Exception as e:
        logger.error(f"Model fitting failed: {e}")
        return {}


def extract_slope_ratios(skip_model: dict, duration_model: dict) -> dict:
    """
    Extract slope ratios (Dyslexic / Control) for each feature
    SR > 1 indicates amplification
    """
    if not skip_model or not duration_model:
        return {}

    skip_results = skip_model.get("results")
    duration_results = duration_model.get("results")

    if skip_results is None or duration_results is None:
        return {}

    features = ["word_length_scaled", "word_frequency_zipf_scaled", "surprisal_scaled"]
    feature_names = ["Length", "Frequency", "Surprisal"]

    slope_ratios = {}

    for feat, name in zip(features, feature_names):
        # Duration model slope ratio
        try:
            main_effect = duration_results.loc[feat, "Estimate"]
            interaction = duration_results.loc[f"dyslexic_int:{feat}", "Estimate"]

            dyslexic_slope = main_effect + interaction
            control_slope = main_effect

            sr = dyslexic_slope / control_slope if control_slope != 0 else np.nan

            slope_ratios[name] = {
                "control_slope": float(control_slope),
                "dyslexic_slope": float(dyslexic_slope),
                "slope_ratio": float(sr),
                "amplified": sr > 1,
            }

            logger.info(f"  {name}: SR = {sr:.3f} {'(AMPLIFIED)' if sr > 1 else ''}")
        except Exception as e:
            logger.warning(f"  Could not compute slope ratio for {name}: {e}")
            slope_ratios[name] = {
                "control_slope": np.nan,
                "dyslexic_slope": np.nan,
                "slope_ratio": np.nan,
                "amplified": False,
            }

    return slope_ratios


def bootstrap_slope_ratios(data: pd.DataFrame, n_boot: int = 1000) -> dict:
    """
    Bootstrap confidence intervals for slope ratios (faster participant-level approach)
    """
    logger.info(f"  Computing bootstrap CIs ({n_boot} iterations)...")

    # First, compute participant-level slopes
    from scipy.stats import linregress
    from tqdm import tqdm

    participant_slopes = []
    for subject in data["subject_id"].unique():
        subj_data = data[data["subject_id"] == subject]
        dyslexic = subj_data["dyslexic"].iloc[0]

        # Only use fixated words
        fixated = subj_data[subj_data["was_fixated"] == True]

        if len(fixated) > 10:  # Need enough data points
            for feature, name in [
                ("word_length_scaled", "Length"),
                ("word_frequency_zipf_scaled", "Frequency"),
                ("surprisal_scaled", "Surprisal"),
            ]:

                clean = fixated[[feature, "total_reading_time"]].dropna()
                if len(clean) > 10:
                    slope, _, _, _, _ = linregress(
                        clean[feature], clean["total_reading_time"]
                    )
                    participant_slopes.append(
                        {
                            "subject_id": subject,
                            "dyslexic": dyslexic,
                            "feature": name,
                            "slope": slope,
                        }
                    )

    slopes_df = pd.DataFrame(participant_slopes)

    # Bootstrap slope ratios
    boot_srs = {"Length": [], "Frequency": [], "Surprisal": []}
    subjects = slopes_df["subject_id"].unique()

    for i in tqdm(range(n_boot), desc="  Bootstrap slope ratios", leave=False):
        boot_subjects = resample(subjects, replace=True, random_state=i)
        boot_data = slopes_df[slopes_df["subject_id"].isin(boot_subjects)]

        for feature in ["Length", "Frequency", "Surprisal"]:
            feat_data = boot_data[boot_data["feature"] == feature]

            control_slope = feat_data[~feat_data["dyslexic"]]["slope"].mean()
            dyslexic_slope = feat_data[feat_data["dyslexic"]]["slope"].mean()

            if control_slope != 0:
                sr = dyslexic_slope / control_slope
                boot_srs[feature].append(sr)

    # Compute CIs
    cis = {}
    for feat, values in boot_srs.items():
        if len(values) > 0:
            ci_low, ci_high = np.percentile(values, [2.5, 97.5])
            cis[feat] = {"ci_low": float(ci_low), "ci_high": float(ci_high)}
            logger.info(f"  {feat}: 95% CI = [{ci_low:.3f}, {ci_high:.3f}]")

    return cis


def predict_expected_ert(
    data: pd.DataFrame, skip_model: dict, duration_model: dict
) -> pd.DataFrame:
    """
    Recombine skip and duration models to predict Expected ERT
    E[ERT] = (1 - p_skip) * E[TRT | fixated]
    """
    if not skip_model or not duration_model:
        logger.warning("Models not available for prediction")
        return data

    data_copy = data.copy()

    # Get models
    skip_m = skip_model.get("model")
    duration_m = duration_model.get("model")
    smearing = duration_model.get("smearing_factor", 1.0)

    if skip_m is None or duration_m is None:
        return data

    logger.info("  Predicting expected ERT...")

    try:
        # Prepare data for prediction
        data_copy["dyslexic_int"] = data_copy["dyslexic"].astype(int)

        # Remove missing values for prediction
        required_cols = [
            "dyslexic_int",
            "word_length_scaled",
            "word_frequency_zipf_scaled",
            "surprisal_scaled",
        ]
        pred_data = data_copy.dropna(subset=required_cols)

        # Predict skip probability
        log_odds = skip_m.predict(pred_data)
        p_skip = 1 / (1 + np.exp(-log_odds))

        # Predict log(TRT)
        log_trt_pred = duration_m.predict(pred_data)
        trt_pred = np.exp(log_trt_pred) * smearing

        # Combine
        pred_data["ERT_predicted"] = (1 - p_skip) * trt_pred

        # Merge back
        data_copy = data_copy.merge(
            pred_data[["ERT_predicted"]], left_index=True, right_index=True, how="left"
        )

        logger.info(
            f"  Predicted ERT for {(~data_copy['ERT_predicted'].isna()).sum():,} words"
        )
        logger.info(
            f"    Mean predicted ERT: {data_copy['ERT_predicted'].mean():.2f}ms"
        )
        logger.info(f"    Mean observed ERT: {data_copy['ERT'].mean():.2f}ms")

        return data_copy

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return data
