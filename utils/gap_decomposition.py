# utils/gap_decomposition.py
"""
Gap decomposition analysis - quantify how much of group gap is explained
Part C of hypothesis testing - REFACTORED to use main effects only
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
    Hierarchical gap decomposition using main-effects-only models:
    M0: ERT_expected ~ Group
    M1: M0 + Length (main effect)
    M2: M1 + Frequency (main effect)
    M3: M2 + Surprisal (main effect)

    This measures COMPOSITION effects (explained by covariate differences)

    Then add interactions separately to measure COEFFICIENT effects

    Returns:
        Dictionary with gaps, R², and % explained
    """
    logger.info("=" * 60)
    logger.info("PART C: GAP DECOMPOSITION")
    logger.info("=" * 60)

    results = {}

    # Fit hierarchical models (main effects only)
    models = fit_hierarchical_models_main_effects(data)
    results["models"] = models

    # Compute gaps at observed covariates
    gaps = compute_gaps_for_all_models(data, models)
    results["gaps"] = gaps

    # Compute R²
    r2_values = compute_marginal_r2(data, models)
    results["r2_values"] = r2_values

    # Compute % explained
    pct_explained = compute_percent_explained(gaps)
    results["percent_explained"] = pct_explained

    # OPTIONAL: Fit interaction models to separate composition vs coefficients
    logger.info("\nFitting interaction models (Oaxaca/Blinder decomposition)...")
    interaction_models = fit_interaction_models(data)
    results["interaction_models"] = interaction_models

    # Decompose gap into composition vs coefficients
    decomposition = decompose_gap_oaxaca(data, models, interaction_models)
    results["oaxaca_decomposition"] = decomposition
    # Summary
    logger.info("\nGAP DECOMPOSITION SUMMARY:")

    if "M0" in gaps:
        logger.info(f"  Baseline gap (M0): {gaps['M0']:.2f}ms")
        for i in range(1, 4):
            model_name = f"M{i}"
            if model_name in gaps:
                gap = gaps[model_name]
                pct = pct_explained.get(model_name, np.nan)
                if f"M{i}" in r2_values and f"M{i-1}" in r2_values:
                    delta_r2 = r2_values[f"M{i}"] - r2_values[f"M{i-1}"]
                else:
                    delta_r2 = np.nan
                logger.info(
                    f"  M{i}: Gap={gap:.2f}ms, Explained={pct:.1f}%, DeltaR²={delta_r2:.4f}"
                )
    else:
        logger.warning("  Gap computation failed - no baseline gap available")

    return results


def fit_hierarchical_models_main_effects(data: pd.DataFrame) -> dict:
    """
    Fit hierarchical sequence of OLS models with MAIN EFFECTS ONLY
    This isolates the composition effect (covariate differences between groups)
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        logger.error("statsmodels required")
        return {}

    data_copy = data.copy()
    data_copy["dyslexic_int"] = data_copy["dyslexic"].astype(int)

    # Use ERT_expected (not raw ERT with zeros)
    if "ERT_expected" not in data_copy.columns:
        logger.error("ERT_expected not found - run continuous models first")
        return {}

    # Main effects only (no interactions)
    formulas = {
        "M0": "ERT_expected ~ dyslexic_int",
        "M1": "ERT_expected ~ dyslexic_int + word_length_scaled",
        "M2": "ERT_expected ~ dyslexic_int + word_length_scaled + word_frequency_zipf_scaled",
        "M3": "ERT_expected ~ dyslexic_int + word_length_scaled + word_frequency_zipf_scaled + surprisal_scaled",
    }

    models = {}

    for name, formula in formulas.items():
        logger.info(f"\nFitting {name} (main effects only)...")
        logger.info(f"  Formula: {formula}")

        try:
            # Define required columns
            if name == "M0":
                required = ["ERT_expected", "dyslexic_int", "subject_id"]
            elif name == "M1":
                required = [
                    "ERT_expected",
                    "dyslexic_int",
                    "word_length_scaled",
                    "subject_id",
                ]
            elif name == "M2":
                required = [
                    "ERT_expected",
                    "dyslexic_int",
                    "word_length_scaled",
                    "word_frequency_zipf_scaled",
                    "subject_id",
                ]
            else:  # M3
                required = [
                    "ERT_expected",
                    "dyslexic_int",
                    "word_length_scaled",
                    "word_frequency_zipf_scaled",
                    "surprisal_scaled",
                    "subject_id",
                ]

            data_clean = data_copy.dropna(subset=required)

            # Fit OLS with clustered SEs
            model = smf.ols(formula, data=data_clean)
            result = model.fit(
                cov_type="cluster", cov_kwds={"groups": data_clean["subject_id"]}
            )

            models[name] = {
                "model": result,
                "results": result,
                "formula": formula,
                "data_clean": data_clean,
            }

            logger.info(f"  {name} fitted successfully (OLS with clustered SEs)")

        except Exception as e:
            logger.error(f"  {name} fitting failed: {e}")
            models[name] = None

    return models


def fit_interaction_models(data: pd.DataFrame) -> dict:
    """
    Fit models WITH interactions to separate coefficient effects
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {}

    data_copy = data.copy()
    data_copy["dyslexic_int"] = data_copy["dyslexic"].astype(int)

    # Interaction models
    formulas = {
        "M1_int": "ERT_expected ~ dyslexic_int * word_length_scaled",
        "M2_int": "ERT_expected ~ dyslexic_int * (word_length_scaled + word_frequency_zipf_scaled)",
        "M3_int": "ERT_expected ~ dyslexic_int * (word_length_scaled + word_frequency_zipf_scaled + surprisal_scaled)",
    }

    models = {}

    for name, formula in formulas.items():
        try:
            required = [
                "ERT_expected",
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


def compute_gaps_for_all_models(data: pd.DataFrame, models: dict) -> dict:
    """
    Compute counterfactual group gap for each model
    Gap = E[ERT_expected | Group=Dyslexic, X] - E[ERT_expected | Group=Control, X]
    where X are the observed covariates
    """
    logger.info("\nComputing counterfactual gaps...")

    gaps = {}

    for name, model_dict in models.items():
        if model_dict is None:
            continue

        model = model_dict["model"]
        data_clean = model_dict["data_clean"]

        try:
            # Create counterfactual data with Group=Control (dyslexic_int=0)
            # CRITICAL: Only include predictor columns (not DV)
            pred_cols = [
                col
                for col in data_clean.columns
                if col not in ["ERT_expected", "subject_id"]
            ]

            data_ctrl = data_clean[pred_cols].copy()
            data_ctrl["dyslexic_int"] = 0
            pred_ctrl = model.predict(data_ctrl)

            # Create counterfactual data with Group=Dyslexic (dyslexic_int=1)
            data_dys = data_clean[pred_cols].copy()
            data_dys["dyslexic_int"] = 1
            pred_dys = model.predict(data_dys)

            # Gap = mean difference
            gap = pred_dys.mean() - pred_ctrl.mean()
            gaps[name] = float(gap)

            logger.info(f"  {name}: Gap = {gap:.2f}ms")

        except Exception as e:
            logger.error(f"  {name} gap computation failed: {e}")
            gaps[name] = np.nan

    return gaps


def decompose_gap_oaxaca(
    data: pd.DataFrame, main_models: dict, interaction_models: dict
) -> dict:
    """
    Oaxaca/Blinder decomposition:
    Total Gap = Composition Effect + Coefficient Effect

    Composition: explained by covariate differences (from main-effects models)
    Coefficient: explained by differential slopes (from interaction models)
    """
    logger.info("\nOaxaca/Blinder Decomposition:")

    if "M3" not in main_models or main_models["M3"] is None:
        logger.warning("  Main model M3 not available for decomposition")
        return {}

    if "M3_int" not in interaction_models or interaction_models["M3_int"] is None:
        logger.warning("  Interaction model M3_int not available for decomposition")
        return {}

    main_model = main_models["M3"]["model"]
    int_model = interaction_models["M3_int"]["model"]
    data_clean = main_models["M3"]["data_clean"]

    try:
        # Get predictor columns (exclude DV and grouping vars)
        pred_cols = [
            col
            for col in data_clean.columns
            if col not in ["ERT_expected", "subject_id"]
        ]

        # Main effects model gaps
        data_ctrl_main = data_clean[pred_cols].copy()
        data_ctrl_main["dyslexic_int"] = 0
        pred_ctrl_main = main_model.predict(data_ctrl_main)

        data_dys_main = data_clean[pred_cols].copy()
        data_dys_main["dyslexic_int"] = 1
        pred_dys_main = main_model.predict(data_dys_main)

        gap_main = pred_dys_main.mean() - pred_ctrl_main.mean()

        # Interaction model gaps
        data_ctrl_int = data_clean[pred_cols].copy()
        data_ctrl_int["dyslexic_int"] = 0
        pred_ctrl_int = int_model.predict(data_ctrl_int)

        data_dys_int = data_clean[pred_cols].copy()
        data_dys_int["dyslexic_int"] = 1
        pred_dys_int = int_model.predict(data_dys_int)

        gap_int = pred_dys_int.mean() - pred_ctrl_int.mean()

        # Baseline gap (M0)
        if "M0" in main_models and main_models["M0"] is not None:
            m0_model = main_models["M0"]["model"]
            m0_data = main_models["M0"]["data_clean"]
            m0_pred_cols = [
                col
                for col in m0_data.columns
                if col not in ["ERT_expected", "subject_id"]
            ]

            m0_ctrl = m0_data[m0_pred_cols].copy()
            m0_ctrl["dyslexic_int"] = 0
            m0_dys = m0_data[m0_pred_cols].copy()
            m0_dys["dyslexic_int"] = 1

            baseline_gap = (
                m0_model.predict(m0_dys).mean() - m0_model.predict(m0_ctrl).mean()
            )
        else:
            baseline_gap = gap_main

        # Composition effect ≈ reduction from baseline to main-effects model
        composition_effect = baseline_gap - gap_main

        # Coefficient effect ≈ additional gap captured by interactions
        coefficient_effect = gap_int - gap_main

        decomposition = {
            "baseline_gap": float(baseline_gap),
            "composition_effect": float(composition_effect),
            "coefficient_effect": float(coefficient_effect),
            "composition_pct": (
                float(composition_effect / baseline_gap * 100)
                if baseline_gap != 0
                else 0
            ),
            "coefficient_pct": (
                float(coefficient_effect / baseline_gap * 100)
                if baseline_gap != 0
                else 0
            ),
            "residual_gap": float(gap_int),
        }

        logger.info(f"  Baseline gap: {baseline_gap:.2f}ms")
        logger.info(
            f"  Composition effect: {composition_effect:.2f}ms ({decomposition['composition_pct']:.1f}%)"
        )
        logger.info(
            f"  Coefficient effect: {coefficient_effect:.2f}ms ({decomposition['coefficient_pct']:.1f}%)"
        )
        logger.info(f"  Residual gap: {gap_int:.2f}ms")

        return decomposition

    except Exception as e:
        logger.error(f"  Decomposition failed: {e}")
        return {}


def compute_marginal_r2(data: pd.DataFrame, models: dict) -> dict:
    """
    Compute R² for each model
    CRITICAL: Use ERT_expected (the DV we fit on), not ERT (observed)
    """
    logger.info("\nComputing R²...")

    r2_values = {}

    for name, model_dict in models.items():
        if model_dict is None:
            continue

        model = model_dict["model"]
        data_clean = model_dict["data_clean"]

        try:
            pred = model.fittedvalues
            # FIX: Use ERT_expected (what we fit on), not ERT
            r2 = r2_score(data_clean["ERT_expected"], pred)
            r2_values[name] = float(r2)

            logger.info(f"  {name}: R² = {r2:.4f}")

        except Exception as e:
            logger.error(f"  {name} R² computation failed: {e}")
            r2_values[name] = np.nan

    return r2_values


def compute_percent_explained(gaps: dict) -> dict:
    """
    Compute % of gap explained by each model
    % Explained = 100 * (Gap₀ - Gapᵢ) / Gap₀
    """
    gap_0 = gaps.get("M0", np.nan)

    if np.isnan(gap_0) or gap_0 == 0:
        logger.warning("Baseline gap is 0 or NaN, cannot compute % explained")
        return {}

    pct_explained = {}

    for name, gap in gaps.items():
        if name == "M0":
            pct_explained[name] = 0.0
        else:
            pct = 100 * (gap_0 - gap) / gap_0
            pct_explained[name] = float(pct)

    return pct_explained
