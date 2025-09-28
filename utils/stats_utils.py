"""
Statistical analysis utilities for dyslexia research
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """
    Calculate Cohen's d effect size between two groups

    Args:
        group1: First group data
        group2: Second group data

    Returns:
        Cohen's d effect size
    """
    if len(group1) == 0 or len(group2) == 0:
        return np.nan

    # Remove NaN values
    group1 = group1.dropna()
    group2 = group2.dropna()

    if len(group1) == 0 or len(group2) == 0:
        return np.nan

    # Calculate pooled standard deviation
    pooled_std = np.sqrt(
        ((len(group1) - 1) * group1.std() ** 2 + (len(group2) - 1) * group2.std() ** 2)
        / (len(group1) + len(group2) - 2)
    )

    if pooled_std == 0:
        return np.nan

    # Calculate Cohen's d
    cohens_d = (group2.mean() - group1.mean()) / pooled_std
    return cohens_d


def calculate_group_summary_stats(
    data: pd.DataFrame, skipping_analysis: Dict = None
) -> Dict[str, Any]:
    """
    Calculate summary statistics by group for key measures

    Args:
        data: DataFrame with dyslexic column and eye-tracking measures
        skipping_analysis: Optional enhanced skipping analysis results

    Returns:
        Dictionary with group statistics for each measure
    """
    if "dyslexic" not in data.columns:
        logger.warning("No dyslexic column found - cannot calculate group statistics")
        return {}

    group_stats = {}

    # Word-level measures by group
    word_level_measures = [
        "total_reading_time",
        "first_fixation_duration",
        "gaze_duration",
        "word_go_past_time",
        "n_fixations",
        "regression_probability",
    ]

    # Add skipping_probability to measures if we have enhanced analysis
    if skipping_analysis and "skipping_by_group" in skipping_analysis:
        word_level_measures.append("skipping_probability")

    for measure in word_level_measures:
        if measure == "skipping_probability" and skipping_analysis:
            # Use enhanced skipping analysis results instead of data
            skipping_by_group = skipping_analysis.get("skipping_by_group", {})

            control_stats = skipping_by_group.get("control", {})
            dyslexic_stats = skipping_by_group.get("dyslexic", {})

            group_stats[f"{measure}_by_group"] = {
                **{f"control_{k}": v for k, v in control_stats.items()},
                **{f"dyslexic_{k}": v for k, v in dyslexic_stats.items()},
            }

            # Add Cohen's d if we have both groups
            if control_stats and dyslexic_stats:
                # Calculate Cohen's d from group means and stds
                control_mean = control_stats.get("mean", 0)
                dyslexic_mean = dyslexic_stats.get("mean", 0)
                control_std = control_stats.get("std", 0)
                dyslexic_std = dyslexic_stats.get("std", 0)
                control_n = control_stats.get("count", 0)
                dyslexic_n = dyslexic_stats.get("count", 0)

                if (
                    control_n > 0
                    and dyslexic_n > 0
                    and (control_std > 0 or dyslexic_std > 0)
                ):
                    # Pooled standard deviation
                    pooled_std = np.sqrt(
                        (
                            (control_n - 1) * control_std**2
                            + (dyslexic_n - 1) * dyslexic_std**2
                        )
                        / (control_n + dyslexic_n - 2)
                    )
                    if pooled_std > 0:
                        cohens_d = (dyslexic_mean - control_mean) / pooled_std
                        group_stats[f"{measure}_by_group"]["cohens_d"] = float(cohens_d)

        elif measure in data.columns:
            # Use regular data for other measures
            group_summary = (
                data.groupby("dyslexic")[measure].agg(["mean", "std", "count"]).round(2)
            )

            # Extract values safely
            control_stats = {}
            dyslexic_stats = {}

            if False in group_summary.index:
                control_stats = {
                    "control_mean": float(group_summary.loc[False, "mean"]),
                    "control_std": float(group_summary.loc[False, "std"]),
                    "control_count": int(group_summary.loc[False, "count"]),
                }

            if True in group_summary.index:
                dyslexic_stats = {
                    "dyslexic_mean": float(group_summary.loc[True, "mean"]),
                    "dyslexic_std": float(group_summary.loc[True, "std"]),
                    "dyslexic_count": int(group_summary.loc[True, "count"]),
                }

            group_stats[f"{measure}_by_group"] = {**control_stats, **dyslexic_stats}

            # Calculate effect size (Cohen's d) if both groups exist
            if False in group_summary.index and True in group_summary.index:
                control_data = data[data["dyslexic"] == False][measure]
                dyslexic_data = data[data["dyslexic"] == True][measure]

                cohens_d = calculate_cohens_d(control_data, dyslexic_data)
                if not np.isnan(cohens_d):
                    group_stats[f"{measure}_by_group"]["cohens_d"] = float(cohens_d)

    # Subject-level measures by group (averaged within subjects first)
    subject_level = (
        data.groupby(["subject_id", "dyslexic"])
        .agg(
            {
                measure: "mean"
                for measure in word_level_measures
                if measure in data.columns and measure != "skipping_probability"
            }
        )
        .reset_index()
    )

    for measure in word_level_measures:
        if measure == "skipping_probability" and skipping_analysis:
            # Use enhanced skipping analysis for subject-level skipping
            subject_skipping = skipping_analysis.get(
                "subject_level_skipping_by_group", {}
            )

            control_stats = subject_skipping.get("control", {})
            dyslexic_stats = subject_skipping.get("dyslexic", {})

            group_stats[f"{measure}_subject_level_by_group"] = {
                **{f"control_{k}": v for k, v in control_stats.items()},
                **{f"dyslexic_{k}": v for k, v in dyslexic_stats.items()},
            }

        elif measure in subject_level.columns:
            subj_group_summary = (
                subject_level.groupby("dyslexic")[measure]
                .agg(["mean", "std", "count"])
                .round(2)
            )

            # Extract values safely
            control_stats = {}
            dyslexic_stats = {}

            if False in subj_group_summary.index:
                control_stats = {
                    "control_mean": float(subj_group_summary.loc[False, "mean"]),
                    "control_std": float(subj_group_summary.loc[False, "std"]),
                    "control_n_subjects": int(subj_group_summary.loc[False, "count"]),
                }

            if True in subj_group_summary.index:
                dyslexic_stats = {
                    "dyslexic_mean": float(subj_group_summary.loc[True, "mean"]),
                    "dyslexic_std": float(subj_group_summary.loc[True, "std"]),
                    "dyslexic_n_subjects": int(subj_group_summary.loc[True, "count"]),
                }

            group_stats[f"{measure}_subject_level_by_group"] = {
                **control_stats,
                **dyslexic_stats,
            }

    return group_stats


def calculate_basic_statistics(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate basic descriptive statistics for the dataset

    Args:
        data: DataFrame with eye-tracking data

    Returns:
        Dictionary with basic statistics
    """
    stats = {
        "data_shape": [int(x) for x in data.shape],
        "columns": list(data.columns),
        "subjects": int(data["subject_id"].nunique()),
        "missing_data": {
            str(k): int(v) for k, v in data.isnull().sum().to_dict().items()
        },
    }

    # Trial and word statistics
    if "trial_id" in data.columns:
        stats["trials"] = int(data["trial_id"].nunique())

    if "word_text" in data.columns:
        stats["unique_words"] = int(data["word_text"].nunique())

    # Group statistics
    if "dyslexic" in data.columns:
        subject_groups = data.groupby("subject_id")["dyslexic"].first()
        stats["dyslexic_subjects"] = int(subject_groups.sum())
        stats["control_subjects"] = int(len(subject_groups) - subject_groups.sum())
        stats["dyslexic_proportion"] = float(subject_groups.mean())

    # Word and reading time statistics
    if "word_length" in data.columns:
        stats["word_stats"] = {
            "mean_length": float(data["word_length"].mean()),
            "std_length": float(data["word_length"].std()),
            "min_length": int(data["word_length"].min()),
            "max_length": int(data["word_length"].max()),
        }

    if "total_reading_time" in data.columns:
        stats["reading_time_stats"] = {
            "mean": float(data["total_reading_time"].mean()),
            "std": float(data["total_reading_time"].std()),
            "min": float(data["total_reading_time"].min()),
            "max": float(data["total_reading_time"].max()),
        }

    return stats


def generate_analysis_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate final analysis summary

    Args:
        results: Dictionary with analysis results

    Returns:
        Summary dictionary
    """
    return {
        "hypotheses_tested": 3,
        "data_type": "Word-level eye-tracking from ExtractedFeatures",
        "key_findings": ["Analysis completed with proper word-level data"],
        "recommendations": [
            "Results now based on actual words rather than character fixations"
        ],
    }
