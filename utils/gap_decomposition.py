# utils/gap_decomposition.py
"""
Gap decomposition analysis - quantify how much of group gap is explained
Part C of hypothesis testing - REFACTORED to decompose observed ERT
"""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
logger = logging.getLogger(__name__)


def run_gap_decomposition(data: pd.DataFrame) -> dict:
    """
    Hierarchical gap decomposition on OBSERVED ERT

    Fits models on observed ERT and computes:
    1. Modelled group gaps (Δk) from predictions
    2. Composition gaps (covariate differences)
    3. % gap explained from Δk reduction
    4. Oaxaca decomposition (composition vs coefficient effects)

    Returns:
        Dictionary with gaps, R², and % explained
    """
    logger.info("=" * 60)
    logger.info("PART C: GAP DECOMPOSITION (ON OBSERVED ERT)")
    logger.info("=" * 60)

    results = {}

    # Fit hierarchical models on OBSERVED ERT
    models = fit_hierarchical_models_main_effects(data)
    results["models"] = models

    # Compute modelled group gaps (Δk) from predictions
    logger.info("\nComputing modelled group gaps from predictions on common sample...")
    gaps = compute_modelled_group_gaps(data, models)
    results["gaps"] = gaps
    results["observed_gaps"] = gaps  # Alias for plotting compatibility

    # Compute composition gaps (counterfactual: both groups at control slopes)
    composition_gaps = compute_composition_gaps(data, models)
    results["composition_gaps"] = composition_gaps

    # Compute R² on OBSERVED ERT
    r2_values = compute_r2_on_observed_ert(data, models)
    results["r2_values"] = r2_values

    # Compute % explained from gap reduction (Δk)
    pct_explained = compute_percent_explained_from_gaps(gaps)
    results["percent_explained"] = pct_explained

    # Fit interaction models
    logger.info("\nFitting interaction models...")
    interaction_models = fit_interaction_models(data)
    results["interaction_models"] = interaction_models

    # Oaxaca decomposition on OBSERVED ERT
    decomposition = decompose_gap_oaxaca_observed(data, models, interaction_models)
    results["oaxaca_decomposition"] = decomposition

    # Summary
    print_gap_summary(gaps, composition_gaps, pct_explained, r2_values)

    return results


def fit_hierarchical_models_main_effects(data: pd.DataFrame) -> dict:
    """
    Fit hierarchical OLS models on OBSERVED ERT
    Use same complete-case sample for all models
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        logger.error("statsmodels required")
        return {}

    data_copy = data.copy()
    data_copy["dyslexic_int"] = data_copy["dyslexic"].astype(int)

    # Ensure ERT exists
    if "ERT" not in data_copy.columns:
        logger.warning(
            "ERT not found - creating from was_fixated and total_reading_time"
        )
        data_copy["ERT"] = np.where(
            data_copy["was_fixated"], data_copy["total_reading_time"], 0
        )

    # Use common complete-case sample
    required_all = [
        "ERT",
        "dyslexic_int",
        "word_length_scaled",
        "word_frequency_zipf_scaled",
        "surprisal_scaled",
        "subject_id",
    ]

    data_clean_common = data_copy.dropna(subset=required_all)

    logger.info(
        f"Using common sample of {len(data_clean_common):,} observations for all models"
    )

    # Formulas - FIT ON OBSERVED ERT
    formulas = {
        "M0": "ERT ~ dyslexic_int",
        "M1": "ERT ~ dyslexic_int + word_length_scaled",
        "M2": "ERT ~ dyslexic_int + word_length_scaled + word_frequency_zipf_scaled",
        "M3": "ERT ~ dyslexic_int + word_length_scaled + word_frequency_zipf_scaled + surprisal_scaled",
    }

    models = {}

    for name, formula in formulas.items():
        logger.info(f"\nFitting {name} (main effects only)...")
        logger.info(f"  Formula: {formula}")

        try:
            model = smf.ols(formula, data=data_clean_common)
            result = model.fit(
                cov_type="cluster", cov_kwds={"groups": data_clean_common["subject_id"]}
            )

            models[name] = {
                "model": result,
                "results": result,
                "formula": formula,
                "data_clean": data_clean_common,
            }

            logger.info(f"  {name} fitted successfully (OLS with clustered SEs)")

        except Exception as e:
            logger.error(f"  {name} fitting failed: {e}")
            models[name] = None

    return models


def fit_interaction_models(data: pd.DataFrame) -> dict:
    """
    Fit models WITH interactions on OBSERVED ERT
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {}

    data_copy = data.copy()
    data_copy["dyslexic_int"] = data_copy["dyslexic"].astype(int)

    # Ensure ERT exists
    if "ERT" not in data_copy.columns:
        data_copy["ERT"] = np.where(
            data_copy["was_fixated"], data_copy["total_reading_time"], 0
        )

    # Interaction models - FIT ON OBSERVED ERT
    formulas = {
        "M1_int": "ERT ~ dyslexic_int * word_length_scaled",
        "M2_int": "ERT ~ dyslexic_int * (word_length_scaled + word_frequency_zipf_scaled)",
        "M3_int": "ERT ~ dyslexic_int * (word_length_scaled + word_frequency_zipf_scaled + surprisal_scaled)",
    }

    models = {}

    for name, formula in formulas.items():
        try:
            required = [
                "ERT",
                "dyslexic_int",
                "word_length_scaled",
                "word_frequency_zipf_scaled",
                "surprisal_scaled",
                "subject_id",
            ]
            data_clean = data_copy.dropna(subset=required)

            logger.info(f"  Fitting {name}...")

            model = smf.ols(formula, data=data_clean)
            result = model.fit(
                cov_type="cluster", cov_kwds={"groups": data_clean["subject_id"]}
            )

            models[name] = {
                "model": result,
                "results": result,
                "data_clean": data_clean,
            }

            logger.info(f"    {name} fitted successfully")

        except Exception as e:
            logger.warning(f"  {name} fitting failed: {e}")
            models[name] = None

    return models


def compute_modelled_group_gaps(data: pd.DataFrame, models: dict) -> dict:
    """
    Compute modelled group gaps (Δk) from predictions on common sample

    For each model:
    1. Predict with dyslexic=1 for everyone
    2. Predict with dyslexic=0 for everyone
    3. Gap = mean(pred_dyslexic) - mean(pred_control)

    This is the "modelled group effect" at the pooled covariate distribution.
    """
    logger.info("  Computing modelled group gaps (Δk)...")

    gaps = {}

    for name, model_dict in models.items():
        if model_dict is None:
            continue

        model = model_dict["model"]
        data_clean = model_dict["data_clean"]

        try:
            # Get predictor columns
            pred_cols = [
                col
                for col in data_clean.columns
                if col not in ["ERT", "subject_id", "dyslexic"]
            ]

            # Predict with dyslexic=1 for everyone
            data_as_dys = data_clean[pred_cols].copy()
            data_as_dys["dyslexic_int"] = 1
            pred_as_dys = model.predict(data_as_dys).mean()

            # Predict with dyslexic=0 for everyone
            data_as_ctrl = data_clean[pred_cols].copy()
            data_as_ctrl["dyslexic_int"] = 0
            pred_as_ctrl = model.predict(data_as_ctrl).mean()

            # Modelled gap
            gap = pred_as_dys - pred_as_ctrl
            gaps[name] = float(gap)

            logger.info(f"    {name}: Δ{name[-1]} = {gap:.2f}ms")

        except Exception as e:
            logger.error(f"    {name} failed: {e}")
            gaps[name] = np.nan

    return gaps


def compute_composition_gaps(data: pd.DataFrame, models: dict) -> dict:
    """
    Compute composition gaps: covariate differences holding slopes constant

    For each model:
    1. Split into dyslexic and control groups
    2. Predict both groups with dyslexic=0 (control slopes)
    3. Gap = mean(pred_dys_at_ctrl_slopes) - mean(pred_ctrl)

    This measures: "How much do groups differ due to reading different
    words, if they had the same sensitivity?"
    """
    logger.info("  Computing composition gaps...")

    gaps = {}

    for name, model_dict in models.items():
        if model_dict is None:
            continue

        model = model_dict["model"]
        data_clean = model_dict["data_clean"]

        try:
            # Split by actual group
            dys_data = data_clean[data_clean["dyslexic_int"] == 1].copy()
            ctrl_data = data_clean[data_clean["dyslexic_int"] == 0].copy()

            # Get predictor columns
            pred_cols = [
                col
                for col in data_clean.columns
                if col not in ["ERT", "subject_id", "dyslexic"]
            ]

            # Force both groups to control slopes (dyslexic=0)
            dys_at_ctrl = dys_data[pred_cols].copy()
            dys_at_ctrl["dyslexic_int"] = 0

            ctrl_at_ctrl = ctrl_data[pred_cols].copy()
            ctrl_at_ctrl["dyslexic_int"] = 0

            # Predict
            pred_dys = model.predict(dys_at_ctrl).mean()
            pred_ctrl = model.predict(ctrl_at_ctrl).mean()

            # Composition gap
            gap = pred_dys - pred_ctrl
            gaps[name] = float(gap)

            logger.info(f"    {name}: Composition gap = {gap:.2f}ms")

        except Exception as e:
            logger.error(f"    {name} failed: {e}")
            gaps[name] = np.nan

    return gaps


def compute_r2_on_observed_ert(data: pd.DataFrame, models: dict) -> dict:
    """
    Compute R² on OBSERVED ERT
    Measures how well the model predicts actual reading times
    """
    logger.info("  Computing R² on observed ERT...")

    r2_values = {}

    for name, model_dict in models.items():
        if model_dict is None:
            continue

        model = model_dict["model"]
        data_clean = model_dict["data_clean"]

        try:
            # Get observed ERT and predictions
            observed = data_clean["ERT"]
            predicted = model.fittedvalues

            # Compute R²
            r2 = r2_score(observed, predicted)
            r2_values[name] = float(r2)

            logger.info(f"    {name}: R² = {r2:.4f}")

        except Exception as e:
            logger.error(f"    {name} failed: {e}")
            r2_values[name] = np.nan

    return r2_values


def compute_percent_explained_from_gaps(gaps: dict, floor_ms: float = 5.0) -> dict:
    """
    Compute % gap explained from Δk reduction

    % Explained = 100 * (Δ0 - Δk) / Δ0

    This measures how much the modelled group gap shrinks as we add predictors.
    """
    baseline = gaps.get("M0", np.nan)

    # Safety check
    if np.isnan(baseline) or abs(baseline) < floor_ms:
        logger.warning(
            f"  Baseline gap too small ({baseline:.2f}ms), % explained undefined"
        )
        return {}

    pct_explained = {"M0": 0.0}

    for name in ["M1", "M2", "M3"]:
        if name in gaps:
            pct = 100.0 * (baseline - gaps[name]) / baseline
            pct_explained[name] = float(pct)

    return pct_explained


def decompose_gap_oaxaca_observed(
    data: pd.DataFrame, main_models: dict, interaction_models: dict
) -> dict:
    """
    Oaxaca/Blinder decomposition on OBSERVED ERT
    Total Observed Gap = Composition Effect + Coefficient Effect

    Composition: E[Y|X_dys,β_control] - E[Y|X_ctl,β_control]
    Coefficient: E[Y|X_pooled,β_dys] - E[Y|X_pooled,β_control]
    """
    logger.info("\nOaxaca/Blinder Decomposition (on Observed ERT):")

    if "M3" not in main_models or main_models["M3"] is None:
        logger.warning("  Main model M3 not available")
        return {}

    if "M3_int" not in interaction_models or interaction_models["M3_int"] is None:
        logger.warning("  Interaction model M3_int not available")
        return {}

    main_model = main_models["M3"]["model"]
    int_model = interaction_models["M3_int"]["model"]
    data_clean = main_models["M3"]["data_clean"]

    try:
        # Get predictor columns
        pred_cols = [
            col
            for col in data_clean.columns
            if col not in ["ERT", "subject_id", "dyslexic"]
        ]

        # Split into actual groups
        dys_data = data_clean[data_clean["dyslexic_int"] == 1].copy()
        ctl_data = data_clean[data_clean["dyslexic_int"] == 0].copy()

        # 1. COMPOSITION EFFECT
        # Use each group's actual X, but force control coefficients
        dys_at_ctrl_coef = dys_data[pred_cols].copy()
        dys_at_ctrl_coef["dyslexic_int"] = 0

        ctl_at_ctrl_coef = ctl_data[pred_cols].copy()
        ctl_at_ctrl_coef["dyslexic_int"] = 0

        composition_gap = (
            main_model.predict(dys_at_ctrl_coef).mean()
            - main_model.predict(ctl_at_ctrl_coef).mean()
        )

        # 2. COEFFICIENT EFFECT
        # Use pooled X, toggle group coefficients
        pooled_x = data_clean[pred_cols].copy()

        pooled_at_dys_coef = pooled_x.copy()
        pooled_at_dys_coef["dyslexic_int"] = 1

        pooled_at_ctrl_coef = pooled_x.copy()
        pooled_at_ctrl_coef["dyslexic_int"] = 0

        coefficient_gap = (
            int_model.predict(pooled_at_dys_coef).mean()
            - int_model.predict(pooled_at_ctrl_coef).mean()
        )

        # 3. TOTAL OBSERVED GAP
        # Use interaction model with actual group assignments
        dys_actual = dys_data[pred_cols].copy()
        dys_actual["dyslexic_int"] = 1

        ctl_actual = ctl_data[pred_cols].copy()
        ctl_actual["dyslexic_int"] = 0

        observed_gap = (
            int_model.predict(dys_actual).mean() - int_model.predict(ctl_actual).mean()
        )

        # Calculate percentages
        composition_pct = (
            (composition_gap / observed_gap * 100) if observed_gap != 0 else 0
        )
        coefficient_pct = (
            (coefficient_gap / observed_gap * 100) if observed_gap != 0 else 0
        )
        residual = observed_gap - composition_gap - coefficient_gap

        decomposition = {
            "observed_gap": float(observed_gap),
            "composition_effect": float(composition_gap),
            "coefficient_effect": float(coefficient_gap),
            "composition_pct": float(composition_pct),
            "coefficient_pct": float(coefficient_pct),
            "residual_gap": float(residual),
        }

        logger.info(f"  Total observed gap: {observed_gap:.2f}ms")
        logger.info(
            f"  Composition effect: {composition_gap:.2f}ms ({composition_pct:.1f}%)"
        )
        logger.info(
            f"  Coefficient effect: {coefficient_gap:.2f}ms ({coefficient_pct:.1f}%)"
        )
        logger.info(f"  Residual: {residual:.2f}ms")

        # Add interpretation
        if abs(composition_pct) < 5:
            logger.info("\n  → Groups read similar-difficulty words (composition ≈ 0%)")
            logger.info(
                "  → Gap is due to differential sensitivity (coefficient effect)"
            )
        elif composition_pct > 50:
            logger.info("\n  → Gap is primarily due to covariate differences")
            logger.info(f"  → Dyslexics read {composition_pct:.0f}% harder words")

        return decomposition

    except Exception as e:
        logger.error(f"  Decomposition failed: {e}")
        import traceback

        traceback.print_exc()
        return {}


def print_gap_summary(
    gaps: dict, composition_gaps: dict, pct_explained: dict, r2_values: dict
) -> None:
    """Print clean summary of gap decomposition"""
    logger.info("\n" + "=" * 60)
    logger.info("GAP DECOMPOSITION SUMMARY (ON OBSERVED ERT)")
    logger.info("=" * 60)

    if "M0" not in gaps:
        logger.warning("  No baseline gap available")
        return

    logger.info(f"\nBaseline modelled gap (Δ0): {gaps['M0']:.2f}ms")

    for i in range(1, 4):
        name = f"M{i}"
        feature_name = ["Length", "Frequency", "Surprisal"][i - 1]

        if name in gaps:
            gap = gaps[name]
            comp_gap = composition_gaps.get(name, np.nan)
            pct = pct_explained.get(name, np.nan)

            if name in r2_values and f"M{i-1}" in r2_values:
                delta_r2 = r2_values[name] - r2_values[f"M{i-1}"]
            else:
                delta_r2 = np.nan

            logger.info(f"\n{name} (+{feature_name}):")
            logger.info(f"  Modelled gap (Δ{i}): {gap:.2f}ms")
            logger.info(f"  Composition gap: {comp_gap:.2f}ms")
            logger.info(f"  % Gap explained: {pct:.1f}%")
            logger.info(f"  ΔR²: {delta_r2:.4f}")
