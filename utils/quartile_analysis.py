# utils/quartile_analysis.py
"""
Quartile comparison analysis (Q1 vs Q3)
Part A of hypothesis testing
"""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.utils import resample
from tqdm import tqdm

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def run_quartile_analysis(data: pd.DataFrame) -> dict:
    """
    Run Q1 vs Q3 analysis for each feature

    Returns:
        Dictionary with results for each feature
    """
    logger.info("=" * 60)
    logger.info("PART A: QUARTILE ANALYSIS (Q1 vs Q3)")
    logger.info("=" * 60)

    results = {}

    features = ["word_length", "word_frequency_zipf", "surprisal"]

    for feature in features:
        logger.info(f"\nAnalyzing {feature}...")

        # Compute participant means
        participant_df = compute_participant_quartile_means(data, feature)

        # Fit mixed model
        model_results = fit_quartile_mixed_model(data, feature)

        # Bootstrap confidence intervals
        boot_ci = bootstrap_quartile_effects(data, feature, n_boot=1000)

        results[feature] = {
            "participant_means": participant_df,
            "model": model_results,
            "bootstrap_ci": boot_ci,
        }

    return results


def compute_participant_quartile_means(
    data: pd.DataFrame, feature: str
) -> pd.DataFrame:
    """
    Compute participant-level means in Q1 and Q3
    """
    quartile_col = f"{feature}_quartile"

    if quartile_col not in data.columns:
        logger.warning(f"Quartile column {quartile_col} not found")
        return pd.DataFrame()

    participant_means = []

    for subject in data["subject_id"].unique():
        subj_data = data[data["subject_id"] == subject]
        dyslexic = subj_data["dyslexic"].iloc[0]

        q1_data = subj_data[subj_data[quartile_col] == "Q1"]
        q3_data = subj_data[subj_data[quartile_col] == "Q4"]

        if len(q1_data) > 0 and len(q3_data) > 0:
            participant_means.append(
                {
                    "subject_id": subject,
                    "dyslexic": dyslexic,
                    "Q1_mean": q1_data["ERT"].mean(),
                    "Q3_mean": q3_data["ERT"].mean(),
                    "Q3_minus_Q1": q3_data["ERT"].mean() - q1_data["ERT"].mean(),
                }
            )

    df = pd.DataFrame(participant_means)

    logger.info(f"  Computed means for {len(df)} participants")
    logger.info(f"    Control Q3-Q1: {df[~df['dyslexic']]['Q3_minus_Q1'].mean():.1f}ms")
    logger.info(f"    Dyslexic Q3-Q1: {df[df['dyslexic']]['Q3_minus_Q1'].mean():.1f}ms")

    return df


def fit_quartile_mixed_model(data: pd.DataFrame, feature: str) -> dict:
    """
    Fit mixed model: ERT ~ Group * Bin + (1|Subject) + (1|Word)
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        logger.error("statsmodels required for mixed models")
        return {}

    quartile_col = f"{feature}_quartile"

    # Filter to Q1 and Q4 only
    q_data = data[data[quartile_col].isin(["Q1", "Q4"])].copy()
    q_data["bin"] = (q_data[quartile_col] == "Q4").astype(int)
    q_data["dyslexic_int"] = q_data["dyslexic"].astype(int)

    logger.info(f"  Fitting mixed model on {len(q_data):,} words...")

    try:
        # Try with both random effects
        formula = "ERT ~ dyslexic_int * bin"
        model = smf.mixedlm(
            formula, data=q_data, groups=q_data["subject_id"], re_formula="1"
        )
        result = model.fit(method="bfgs")

        # Extract key coefficients
        coefs = {
            "intercept": result.params.get("Intercept", np.nan),
            "group": result.params.get("dyslexic_int", np.nan),
            "bin": result.params.get("bin", np.nan),
            "interaction": result.params.get("dyslexic_int:bin", np.nan),
        }

        # Extract p-values
        pvals = {
            "group_p": result.pvalues.get("dyslexic_int", np.nan),
            "bin_p": result.pvalues.get("bin", np.nan),
            "interaction_p": result.pvalues.get("dyslexic_int:bin", np.nan),
        }

        logger.info(
            f"  Group×Bin interaction: beta={coefs['interaction']:.2f}, p={pvals['interaction_p']:.4f}"
        )

        return {"coefficients": coefs, "pvalues": pvals, "summary": result.summary()}

    except Exception as e:
        logger.error(f"Model fitting failed: {e}")
        return {}


def bootstrap_quartile_effects(
    data: pd.DataFrame, feature: str, n_boot: int = 1000
) -> dict:
    """
    Bootstrap confidence intervals using participant-level resampling
    Much faster than refitting mixed models
    """
    logger.info(f"  Computing bootstrap CIs ({n_boot} iterations)...")

    from tqdm import tqdm

    quartile_col = f"{feature}_quartile"

    # Get participant-level Q3-Q1 differences
    participant_diffs = []
    for subject in data["subject_id"].unique():
        subj_data = data[data["subject_id"] == subject]
        dyslexic = subj_data["dyslexic"].iloc[0]

        q1_data = subj_data[subj_data[quartile_col] == "Q1"]
        q3_data = subj_data[subj_data[quartile_col] == "Q4"]

        if len(q1_data) > 0 and len(q3_data) > 0:
            participant_diffs.append(
                {
                    "subject_id": subject,
                    "dyslexic": dyslexic,
                    "Q3_minus_Q1": q3_data["ERT"].mean() - q1_data["ERT"].mean(),
                }
            )

    diff_df = pd.DataFrame(participant_diffs)

    # Bootstrap the interaction effect (difference of differences)
    boot_interactions = []
    subjects = diff_df["subject_id"].unique()

    for i in tqdm(range(n_boot), desc=f"  {feature} bootstrap", leave=False):
        # Resample subjects with replacement
        boot_subjects = resample(subjects, replace=True, random_state=i)
        boot_data = diff_df[diff_df["subject_id"].isin(boot_subjects)]

        # Compute group means
        control_mean = boot_data[~boot_data["dyslexic"]]["Q3_minus_Q1"].mean()
        dyslexic_mean = boot_data[boot_data["dyslexic"]]["Q3_minus_Q1"].mean()

        # Interaction = difference of differences
        interaction = dyslexic_mean - control_mean
        boot_interactions.append(interaction)

    # Compute CI
    ci_low, ci_high = np.percentile(boot_interactions, [2.5, 97.5])

    ci = {
        "interaction_ci_low": float(ci_low),
        "interaction_ci_high": float(ci_high),
    }

    logger.info(
        f"  Bootstrap 95% CI for interaction: "
        f"[{ci['interaction_ci_low']:.2f}, {ci['interaction_ci_high']:.2f}]"
    )

    return ci
