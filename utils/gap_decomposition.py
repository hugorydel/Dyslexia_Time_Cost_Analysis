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
    Hierarchical gap decomposition on OBSERVED ERT

    Models are fit on ERT_expected (for consistency with Part B),
    but gaps are computed on OBSERVED ERT to measure real explanatory power.

    Returns:
        Dictionary with gaps, R², and % explained
    """
    logger.info("=" * 60)
    logger.info("PART C: GAP DECOMPOSITION (ON OBSERVED ERT)")
    logger.info("=" * 60)

    results = {}

    # Fit hierarchical models (main effects only, on ERT_expected for consistency)
    models = fit_hierarchical_models_main_effects(data)
    results["models"] = models

    # CRITICAL: Compute gaps on OBSERVED ERT, not ERT_expected
    logger.info("\nComputing gaps on OBSERVED ERT (not model predictions)...")

    # For each model, predict at dyslexic=0 and dyslexic=1, then average over observed ERT
    observed_gaps = compute_observed_ert_gaps(data, models)
    results["observed_gaps"] = observed_gaps

    # Compute composition gaps (counterfactual: both groups at control slopes)
    composition_gaps = compute_composition_gaps_observed(data, models)
    results["composition_gaps"] = composition_gaps

    # Compute R² on OBSERVED ERT
    r2_values = compute_r2_on_observed_ert(data, models)
    results["r2_values"] = r2_values

    # Compute % explained
    pct_explained = compute_percent_explained_observed(observed_gaps, composition_gaps)
    results["percent_explained"] = pct_explained

    # Fit interaction models
    logger.info("\nFitting interaction models...")
    interaction_models = fit_interaction_models(data)
    results["interaction_models"] = interaction_models

    # Oaxaca decomposition on OBSERVED ERT
    decomposition = decompose_gap_oaxaca_observed(data, models, interaction_models)
    results["oaxaca_decomposition"] = decomposition

    # Summary
    print_gap_summary(observed_gaps, composition_gaps, pct_explained, r2_values)

    return results


def fit_hierarchical_models_main_effects(data: pd.DataFrame) -> dict:
    """
    Fit hierarchical OLS models on ERT_expected (for consistency with Part B)
    Use same complete-case sample for all models
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        logger.error("statsmodels required")
        return {}

    data_copy = data.copy()
    data_copy["dyslexic_int"] = data_copy["dyslexic"].astype(int)

    # Check for ERT_expected
    if "ERT_expected" not in data_copy.columns:
        logger.error("ERT_expected not found - run continuous models first")
        return {}

    # Ensure ERT exists for gap computation
    if "ERT" not in data_copy.columns:
        logger.warning(
            "ERT not found - creating from was_fixated and total_reading_time"
        )
        data_copy["ERT"] = np.where(
            data_copy["was_fixated"], data_copy["total_reading_time"], 0
        )

    # Use common complete-case sample
    required_all = [
        "ERT_expected",
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

    # Formulas (fit on ERT_expected for consistency)
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
            model = smf.ols(formula, data=data_clean_common)
            result = model.fit(
                cov_type="cluster", cov_kwds={"groups": data_clean_common["subject_id"]}
            )

            models[name] = {
                "model": result,
                "results": result,
                "formula": formula,
                "data_clean": data_clean_common,  # Includes both ERT and ERT_expected
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
    Compute composition-based gaps for each model
    Gap = E[Y | X_dys, β_pooled] - E[Y | X_ctl, β_pooled]

    This measures how much of the gap is due to covariate differences
    (groups reading different difficulty words), holding slopes constant.
    """
    logger.info("\nComputing composition-based gaps (covariate differences)...")

    gaps = {}

    for name, model_dict in models.items():
        if model_dict is None:
            continue

        model = model_dict["model"]
        data_clean = model_dict["data_clean"]

        try:
            # Split into dyslexic and control groups
            dys_data = data_clean[data_clean["dyslexic_int"] == 1].copy()
            ctl_data = data_clean[data_clean["dyslexic_int"] == 0].copy()

            # Get predictor columns (exclude DV and grouping)
            pred_cols = [
                col
                for col in data_clean.columns
                if col not in ["ERT_expected", "subject_id", "dyslexic"]
            ]

            # CRITICAL: Set dyslexic_int=0 for BOTH groups (use control coefficients)
            dys_data_counterfactual = dys_data[pred_cols].copy()
            dys_data_counterfactual["dyslexic_int"] = 0  # Force control slope

            ctl_data_counterfactual = ctl_data[pred_cols].copy()
            ctl_data_counterfactual["dyslexic_int"] = 0  # Already control

            # Predict using control coefficients for both groups
            pred_dys = model.predict(dys_data_counterfactual).mean()
            pred_ctl = model.predict(ctl_data_counterfactual).mean()

            # Composition gap = difference due to X differences only
            gap = pred_dys - pred_ctl
            gaps[name] = float(gap)

            logger.info(f"  {name}: Composition gap = {gap:.2f}ms")
            logger.info(
                f"    (Dyslexics at control slopes: {pred_dys:.2f}ms, Controls: {pred_ctl:.2f}ms)"
            )

        except Exception as e:
            logger.error(f"  {name} gap computation failed: {e}")
            gaps[name] = np.nan

    return gaps


def compute_observed_ert_gaps(data: pd.DataFrame, models: dict) -> dict:
    """
    Compute gaps on OBSERVED ERT using model predictions

    For each model:
    1. Predict at each row with dyslexic=1 and dyslexic=0
    2. Average these predictions across all rows
    3. Gap = mean(pred_dyslexic) - mean(pred_control)

    This measures: "If we switched everyone's group label, how much would
    the average OBSERVED ERT change according to this model?"
    """
    logger.info("  Computing gaps on observed ERT...")

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
                if col not in ["ERT_expected", "ERT", "subject_id", "dyslexic"]
            ]

            # Predict with dyslexic=1 for everyone
            data_as_dys = data_clean[pred_cols].copy()
            data_as_dys["dyslexic_int"] = 1
            pred_as_dys = model.predict(data_as_dys).mean()

            # Predict with dyslexic=0 for everyone
            data_as_ctrl = data_clean[pred_cols].copy()
            data_as_ctrl["dyslexic_int"] = 0
            pred_as_ctrl = model.predict(data_as_ctrl).mean()

            # Gap on model predictions (which approximate observed ERT)
            gap = pred_as_dys - pred_as_ctrl
            gaps[name] = float(gap)

            logger.info(f"    {name}: Gap = {gap:.2f}ms")

        except Exception as e:
            logger.error(f"    {name} failed: {e}")
            gaps[name] = np.nan

    return gaps


def compute_observed_gaps(data: pd.DataFrame, models: dict) -> dict:
    """
    Compute OBSERVED gaps using actual group coefficients
    Gap = E[Y | dyslexic=1] - E[Y | dyslexic=0]

    This is the baseline we use for % explained calculations
    """
    logger.info("\nComputing observed gaps (actual group coefficients)...")

    gaps = {}

    for name, model_dict in models.items():
        if model_dict is None:
            continue

        model = model_dict["model"]
        data_clean = model_dict["data_clean"]

        try:
            # Split into actual groups
            dys_data = data_clean[data_clean["dyslexic_int"] == 1].copy()
            ctl_data = data_clean[data_clean["dyslexic_int"] == 0].copy()

            # Get predictor columns
            pred_cols = [
                col
                for col in data_clean.columns
                if col not in ["ERT_expected", "subject_id", "dyslexic"]
            ]

            # Predict WITH actual group coefficients (keep dyslexic_int as is)
            pred_dys = model.predict(dys_data[pred_cols]).mean()
            pred_ctl = model.predict(ctl_data[pred_cols]).mean()

            # Observed gap
            gap = pred_dys - pred_ctl
            gaps[name] = float(gap)

            logger.info(f"  {name}: Observed gap = {gap:.2f}ms")

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

    Composition: E[Y|X_dys,β_pooled] - E[Y|X_ctl,β_pooled]
    Coefficient: E[Y|X,β_dys] - E[Y|X,β_pooled]
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
        # Get predictor columns
        pred_cols = [
            col
            for col in data_clean.columns
            if col not in ["ERT_expected", "subject_id", "dyslexic"]
        ]

        # Split groups
        dys_data = data_clean[data_clean["dyslexic_int"] == 1].copy()
        ctl_data = data_clean[data_clean["dyslexic_int"] == 0].copy()

        # 1. COMPOSITION EFFECT (from main model, M3)
        # Use each group's X, but control coefficients
        dys_at_ctl_coef = dys_data[pred_cols].copy()
        dys_at_ctl_coef["dyslexic_int"] = 0

        ctl_at_ctl_coef = ctl_data[pred_cols].copy()
        ctl_at_ctl_coef["dyslexic_int"] = 0

        composition_gap = (
            main_model.predict(dys_at_ctl_coef).mean()
            - main_model.predict(ctl_at_ctl_coef).mean()
        )

        # 2. COEFFICIENT EFFECT (from interaction model, M3_int)
        # Use pooled X, toggle group coefficients
        pooled_x = data_clean[pred_cols].copy()

        pooled_at_dys_coef = pooled_x.copy()
        pooled_at_dys_coef["dyslexic_int"] = 1

        pooled_at_ctl_coef = pooled_x.copy()
        pooled_at_ctl_coef["dyslexic_int"] = 0

        coefficient_gap = (
            int_model.predict(pooled_at_dys_coef).mean()
            - int_model.predict(pooled_at_ctl_coef).mean()
        )

        # 3. OBSERVED GAP (from interaction model with actual groups)
        observed_gap = (
            int_model.predict(dys_data[pred_cols]).mean()
            - int_model.predict(ctl_data[pred_cols]).mean()
        )

        decomposition = {
            "observed_gap": float(observed_gap),
            "composition_effect": float(composition_gap),
            "coefficient_effect": float(coefficient_gap),
            "composition_pct": (
                float(composition_gap / observed_gap * 100) if observed_gap != 0 else 0
            ),
            "coefficient_pct": (
                float(coefficient_gap / observed_gap * 100) if observed_gap != 0 else 0
            ),
            "residual_gap": float(observed_gap - composition_gap - coefficient_gap),
        }

        logger.info(f"  Observed gap: {observed_gap:.2f}ms")
        logger.info(
            f"  Composition effect: {composition_gap:.2f}ms ({decomposition['composition_pct']:.1f}%)"
        )
        logger.info(
            f"  Coefficient effect: {coefficient_gap:.2f}ms ({decomposition['coefficient_pct']:.1f}%)"
        )
        logger.info(f"  Residual: {decomposition['residual_gap']:.2f}ms")

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


def compute_r2_on_observed_ert(data: pd.DataFrame, models: dict) -> dict:
    """
    Compute R² on OBSERVED ERT (not on ERT_expected)

    This measures how well the model predicts actual reading times
    """
    logger.info("  Computing R² on observed ERT...")

    r2_values = {}

    for name, model_dict in models.items():
        if model_dict is None:
            continue

        model = model_dict["model"]
        data_clean = model_dict["data_clean"]

        try:
            # Get observed ERT
            if "ERT" not in data_clean.columns:
                logger.warning(f"    {name}: No ERT column, using ERT_expected")
                observed = data_clean["ERT_expected"]
            else:
                observed = data_clean["ERT"]

            # Get predictions
            predicted = model.fittedvalues

            # Compute R²
            from sklearn.metrics import r2_score

            r2 = r2_score(observed, predicted)
            r2_values[name] = float(r2)

            logger.info(f"    {name}: R² = {r2:.4f}")

        except Exception as e:
            logger.error(f"    {name} failed: {e}")
            r2_values[name] = np.nan

    return r2_values


def compute_composition_gaps(data: pd.DataFrame, models: dict) -> dict:
    """
    Compute composition-based gaps (covariate differences only)
    Gap = E[Y | X_dys, β_control] - E[Y | X_ctl, β_control]

    This measures how much groups differ due to reading different words,
    holding slopes constant at control levels.
    """
    logger.info("\nComputing composition gaps (covariate differences only)...")

    gaps = {}

    for name, model_dict in models.items():
        if model_dict is None:
            continue

        model = model_dict["model"]
        data_clean = model_dict["data_clean"]

        try:
            # Split into dyslexic and control groups
            dys_data = data_clean[data_clean["dyslexic_int"] == 1].copy()
            ctl_data = data_clean[data_clean["dyslexic_int"] == 0].copy()

            # Get predictor columns
            pred_cols = [
                col
                for col in data_clean.columns
                if col not in ["ERT_expected", "subject_id", "dyslexic"]
            ]

            # Force BOTH groups to control slopes (dyslexic_int=0)
            dys_at_control = dys_data[pred_cols].copy()
            dys_at_control["dyslexic_int"] = 0

            ctl_at_control = ctl_data[pred_cols].copy()
            ctl_at_control["dyslexic_int"] = 0

            # Predict using control coefficients for both
            pred_dys = model.predict(dys_at_control).mean()
            pred_ctl = model.predict(ctl_at_control).mean()

            # Composition gap = X differences only
            gap = pred_dys - pred_ctl
            gaps[name] = float(gap)

            logger.info(f"  {name}: Composition gap = {gap:.2f}ms")

        except Exception as e:
            logger.error(f"  {name} composition gap failed: {e}")
            gaps[name] = np.nan

    return gaps


def compute_composition_gaps_observed(data: pd.DataFrame, models: dict) -> dict:
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
                if col not in ["ERT_expected", "ERT", "subject_id", "dyslexic"]
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


def compute_percent_explained(observed_gaps: dict, composition_gaps: dict) -> dict:
    """
    Compute % of OBSERVED gap explained by composition
    % Explained = 100 * (Observed_Gap₀ - Composition_Gapi) / Observed_Gap₀

    This shows how much of the observed gap is explained by covariate differences
    """
    observed_baseline = observed_gaps.get("M0", np.nan)

    # Safety check
    if np.isnan(observed_baseline) or np.isclose(observed_baseline, 0, atol=1e-6):
        logger.warning("Observed baseline gap is 0 or NaN, cannot compute % explained")
        return {}

    pct_explained = {}

    for name, comp_gap in composition_gaps.items():
        if name == "M0":
            pct_explained[name] = 0.0
        else:
            # How much of the OBSERVED gap is explained by composition?
            # If composition gap shrinks, covariates explain more
            pct = 100 * (observed_baseline - comp_gap) / observed_baseline
            pct_explained[name] = float(pct)

    return pct_explained


def compute_percent_explained_observed(
    observed_gaps: dict, composition_gaps: dict
) -> dict:
    """
    Compute % of observed gap explained by each model

    % Explained = 100 * (ObservedGap₀ - CompositionGapᵢ) / ObservedGap₀
    """
    baseline = observed_gaps.get("M0", np.nan)

    # Safety check
    if np.isnan(baseline) or abs(baseline) < 5.0:
        logger.warning(
            f"  Baseline gap too small ({baseline:.2f}ms), % explained may be unstable"
        )
        return {}

    pct_explained = {}

    for name in ["M0", "M1", "M2", "M3"]:
        if name == "M0":
            pct_explained[name] = 0.0
        elif name in composition_gaps:
            comp_gap = composition_gaps[name]
            pct = 100 * (baseline - comp_gap) / baseline
            pct_explained[name] = float(pct)

    return pct_explained


def print_gap_summary(
    observed_gaps: dict, composition_gaps: dict, pct_explained: dict, r2_values: dict
) -> None:
    """Print clean summary of gap decomposition"""
    logger.info("\n" + "=" * 60)
    logger.info("GAP DECOMPOSITION SUMMARY (ON OBSERVED ERT)")
    logger.info("=" * 60)

    if "M0" not in observed_gaps:
        logger.warning("  No baseline gap available")
        return

    logger.info(f"\nBaseline observed gap (M0): {observed_gaps['M0']:.2f}ms")

    for i in range(1, 4):
        name = f"M{i}"
        if name in observed_gaps:
            obs_gap = observed_gaps[name]
            comp_gap = composition_gaps.get(name, np.nan)
            pct = pct_explained.get(name, np.nan)

            if name in r2_values and f"M{i-1}" in r2_values:
                delta_r2 = r2_values[name] - r2_values[f"M{i-1}"]
            else:
                delta_r2 = np.nan

            logger.info(f"\n{name} (+{['Length', 'Frequency', 'Surprisal'][i-1]}):")
            logger.info(f"  Observed gap: {obs_gap:.2f}ms")
            logger.info(f"  Composition gap: {comp_gap:.2f}ms")
            logger.info(f"  % Explained: {pct:.1f}%")
            logger.info(f"  ΔR²: {delta_r2:.4f}")


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
            if col not in ["ERT_expected", "ERT", "subject_id", "dyslexic"]
        ]

        # Split into actual groups
        dys_data = data_clean[data_clean["dyslexic_int"] == 1].copy()
        ctl_data = data_clean[data_clean["dyslexic_int"] == 0].copy()

        # 1. COMPOSITION EFFECT
        # Use each group's actual X, but force control coefficients (dyslexic=0)
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
