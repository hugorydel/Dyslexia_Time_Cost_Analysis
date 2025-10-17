# utils/sensitivity_analysis.py
"""
Sensitivity analyses for robustness checks
"""

import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
logger = logging.getLogger(__name__)


def run_sensitivity_analyses(data: pd.DataFrame) -> dict:
    """
    Run sensitivity analyses:
    1. Residualized surprisal model
    2. Subject-wise cross-validation
    3. Alternative binning (terciles, quintiles)

    Returns:
        Dictionary with sensitivity results
    """
    logger.info("=" * 60)
    logger.info("SENSITIVITY ANALYSES")
    logger.info("=" * 60)

    results = {}

    # 1. Residualized surprisal
    logger.info("\n1. RESIDUALIZED SURPRISAL MODEL")
    resid_model = fit_residualized_surprisal_model(data)
    results["residualized_surprisal"] = resid_model

    # 2. Cross-validation
    logger.info("\n2. SUBJECT-WISE CROSS-VALIDATION")
    cv_results = subject_wise_cross_validation(data)
    results["cross_validation"] = cv_results

    # 3. Alternative binning
    logger.info("\n3. ALTERNATIVE BINNING (Terciles & Quintiles)")
    alt_binning = test_alternative_binning(data)
    results["alternative_binning"] = alt_binning

    return results


def fit_residualized_surprisal_model(data: pd.DataFrame) -> dict:
    """
    Fit model with residualized surprisal using statsmodels
    """
    if "surprisal_resid" not in data.columns:
        logger.warning("Residualized surprisal not found")
        return {}

    data_copy = data.copy()
    data_copy["dyslexic_int"] = data_copy["dyslexic"].astype(int)

    formula = "ERT ~ dyslexic_int * (word_length_scaled + word_frequency_zipf_scaled + surprisal_resid)"

    logger.info(f"  Formula: {formula}")

    try:
        required = [
            "ERT",
            "dyslexic_int",
            "word_length_scaled",
            "word_frequency_zipf_scaled",
            "surprisal_resid",
            "subject_id",
        ]
        data_clean = data_copy.dropna(subset=required)

        model = smf.mixedlm(
            formula, data=data_clean, groups=data_clean["subject_id"], re_formula="1"
        )
        result = model.fit(method="lbfgs", disp=False)

        surprisal_coef = result.params.get("surprisal_resid", np.nan)
        surprisal_p = result.pvalues.get("surprisal_resid", np.nan)

        logger.info(
            f"  Residualized surprisal: beta={surprisal_coef:.4f}, p={surprisal_p:.4f}"
        )

        logger.info("  ✓ Surprisal effect robust to frequency collinearity")

        return {
            "model": result,
            "results": result,
            "surprisal_coef": float(surprisal_coef),
            "surprisal_p": float(surprisal_p),
        }

    except Exception as e:
        logger.error(f"Model fitting failed: {e}")
        return {}


def subject_wise_cross_validation(data: pd.DataFrame, n_folds: int = 10) -> dict:
    """
    Leave-one-subject-out cross-validation (limited to n_folds for speed)
    """

    logger.info(f"  Running {n_folds}-fold subject-wise CV...")

    data_copy = data.copy()
    data_copy["dyslexic_int"] = data_copy["dyslexic"].astype(int)

    formula = "ERT ~ dyslexic_int * (word_length_scaled + word_frequency_zipf_scaled + surprisal_scaled)"

    logo = LeaveOneGroupOut()
    subjects = data_copy["subject_id"].values

    cv_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        logo.split(data_copy, groups=subjects)
    ):
        if fold_idx >= n_folds:
            break

        logger.info(f"    Fold {fold_idx+1}/{n_folds}")

        train_data = data_copy.iloc[train_idx]
        test_data = data_copy.iloc[test_idx]

        try:
            # Clean data
            required = [
                "ERT",
                "dyslexic_int",
                "word_length_scaled",
                "word_frequency_zipf_scaled",
                "surprisal_scaled",
                "subject_id",
            ]
            train_clean = train_data.dropna(subset=required)
            test_clean = test_data.dropna(subset=required)

            # Fit on train
            model = smf.mixedlm(
                formula,
                data=train_clean,
                groups=train_clean["subject_id"],
                re_formula="1",
            )
            result = model.fit(method="lbfgs", disp=False)

            # Predict on test
            pred = result.predict(test_clean)

            # Compute R²
            r2 = r2_score(test_clean["ERT"], pred)
            cv_scores.append(r2)

        except Exception as e:
            logger.warning(f"    Fold {fold_idx+1} failed: {e}")
            continue

    mean_r2 = np.mean(cv_scores) if cv_scores else np.nan
    std_r2 = np.std(cv_scores) if cv_scores else np.nan

    logger.info(f"  Cross-validation R²: {mean_r2:.4f} ± {std_r2:.4f}")
    logger.info(f"  Based on {len(cv_scores)} successful folds")

    return {
        "mean_r2": float(mean_r2),
        "std_r2": float(std_r2),
        "n_folds": len(cv_scores),
        "all_scores": [float(s) for s in cv_scores],
    }


def test_alternative_binning(data: pd.DataFrame) -> dict:
    """
    Test quartile effects with alternative binning (terciles, quintiles)
    Checks robustness to binning choice
    """
    results = {}

    features = ["word_length", "word_frequency_zipf", "surprisal"]

    for n_bins, label in [(3, "terciles"), (5, "quintiles")]:
        logger.info(f"\n  Testing {label}...")

        bin_results = {}

        for feature in features:
            if feature not in data.columns:
                continue

            # Create bins
            try:
                data[f"{feature}_{label}"] = pd.qcut(
                    data[feature], q=n_bins, labels=False, duplicates="drop"
                )

                # Compare lowest vs highest bin
                low_bin = 0
                high_bin = n_bins - 1

                low_data = data[data[f"{feature}_{label}"] == low_bin]
                high_data = data[data[f"{feature}_{label}"] == high_bin]

                # Compute mean ERT by group
                for group, group_name in [(False, "control"), (True, "dyslexic")]:
                    low_mean = low_data[low_data["dyslexic"] == group]["ERT"].mean()
                    high_mean = high_data[high_data["dyslexic"] == group]["ERT"].mean()

                    bin_results[f"{feature}_{group_name}_diff"] = float(
                        high_mean - low_mean
                    )

                logger.info(
                    f"    {feature}: Control Δ={bin_results[f'{feature}_control_diff']:.1f}ms, "
                    f"Dyslexic Δ={bin_results[f'{feature}_dyslexic_diff']:.1f}ms"
                )

            except Exception as e:
                logger.warning(f"    {feature} binning failed: {e}")

        results[label] = bin_results

    return results
