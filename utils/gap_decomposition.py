# utils/gap_decomposition.py
"""
Gap decomposition analysis - quantify how much of group gap is explained
Part C of hypothesis testing
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
    Hierarchical gap decomposition:
    M0: ERT ~ Group
    M1: M0 + Length
    M2: M1 + Frequency
    M3: M2 + Surprisal

    Returns:
        Dictionary with gaps, R², and % explained
    """
    logger.info("=" * 60)
    logger.info("PART C: GAP DECOMPOSITION")
    logger.info("=" * 60)

    results = {}

    # Fit hierarchical models
    models = fit_hierarchical_models(data)
    results["models"] = models

    # Compute gaps
    gaps = compute_gaps_for_all_models(data, models)
    results["gaps"] = gaps

    # Compute R²
    r2_values = compute_marginal_r2(data, models)
    results["r2_values"] = r2_values

    # Compute % explained
    pct_explained = compute_percent_explained(gaps)
    results["percent_explained"] = pct_explained

    # Summary
    logger.info("\nGAP DECOMPOSITION SUMMARY:")
    logger.info(f"  Baseline gap (M0): {gaps['M0']:.2f}ms")
    for i in range(1, 4):
        gap = gaps[f"M{i}"]
        pct = pct_explained[f"M{i}"]
        delta_r2 = r2_values[f"M{i}"] - r2_values[f"M{i-1}"]
        logger.info(
            f"  M{i}: Gap={gap:.2f}ms, Explained={pct:.1f}%, ΔR²={delta_r2:.4f}"
        )

    return results


def fit_hierarchical_models(data: pd.DataFrame) -> dict:
    """
    Fit hierarchical sequence of models using statsmodels
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        logger.error("statsmodels required")
        return {}

    data_copy = data.copy()
    data_copy["dyslexic_int"] = data_copy["dyslexic"].astype(int)

    formulas = {
        "M0": "ERT ~ dyslexic_int",
        "M1": "ERT ~ dyslexic_int * word_length_scaled",
        "M2": "ERT ~ dyslexic_int * (word_length_scaled + word_frequency_zipf_scaled)",
        "M3": "ERT ~ dyslexic_int * (word_length_scaled + word_frequency_zipf_scaled + surprisal_scaled)",
    }

    models = {}

    for name, formula in formulas.items():
        logger.info(f"\nFitting {name}...")
        logger.info(f"  Formula: {formula}")

        try:
            # Clean data
            if name == "M0":
                required = ["ERT", "dyslexic_int", "subject_id"]
            elif name == "M1":
                required = ["ERT", "dyslexic_int", "word_length_scaled", "subject_id"]
            elif name == "M2":
                required = [
                    "ERT",
                    "dyslexic_int",
                    "word_length_scaled",
                    "word_frequency_zipf_scaled",
                    "subject_id",
                ]
            else:  # M3
                required = [
                    "ERT",
                    "dyslexic_int",
                    "word_length_scaled",
                    "word_frequency_zipf_scaled",
                    "surprisal_scaled",
                    "subject_id",
                ]

            data_clean = data_copy.dropna(subset=required)

            # Fit mixed model
            model = smf.mixedlm(
                formula,
                data=data_clean,
                groups=data_clean["subject_id"],
                re_formula="1",
            )
            result = model.fit(method="lbfgs", disp=False)

            models[name] = {"model": result, "results": result, "formula": formula}

            logger.info(f"  {name} fitted successfully")

        except Exception as e:
            logger.error(f"  {name} fitting failed: {e}")
            models[name] = None

    return models


def compute_gaps_for_all_models(data: pd.DataFrame, models: dict) -> dict:
    """
    Compute counterfactual group gap for each model
    Gap = E[ERT | Group=Dyslexic] - E[ERT | Group=Control]
    """
    logger.info("\nComputing counterfactual gaps...")

    gaps = {}

    data_copy = data.copy()

    for name, model_dict in models.items():
        if model_dict is None:
            continue

        model = model_dict["model"]

        try:
            # Predict with Group=Control
            data_ctrl = data_copy.copy()
            data_ctrl["dyslexic_int"] = 0
            pred_ctrl = model.predict(
                data_ctrl, use_rfx=False, verify_predictions=False
            )

            # Predict with Group=Dyslexic
            data_dys = data_copy.copy()
            data_dys["dyslexic_int"] = 1
            pred_dys = model.predict(data_dys, use_rfx=False, verify_predictions=False)

            # Compute gap
            gap = pred_dys.mean() - pred_ctrl.mean()
            gaps[name] = float(gap)

            logger.info(f"  {name}: Gap = {gap:.2f}ms")

        except Exception as e:
            logger.error(f"  {name} gap computation failed: {e}")
            gaps[name] = np.nan

    return gaps


def compute_marginal_r2(data: pd.DataFrame, models: dict) -> dict:
    """
    Compute marginal R² (fixed effects only) for each model
    """
    logger.info("\nComputing marginal R²...")

    r2_values = {}

    for name, model_dict in models.items():
        if model_dict is None:
            continue

        model = model_dict["model"]

        try:
            # Predict with fixed effects only
            pred = model.predict(data, use_rfx=False, verify_predictions=False)

            # R²
            r2 = r2_score(data["ERT"], pred)
            r2_values[name] = float(r2)

            logger.info(f"  {name}: R² = {r2:.4f}")

        except Exception as e:
            logger.error(f"  {name} R² computation failed: {e}")
            r2_values[name] = np.nan

    return r2_values


def compute_percent_explained(gaps: dict) -> dict:
    """
    Compute % of gap explained by each model
    % Explained = 100 * (Gap₀ - Gapₖ) / Gap₀
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
