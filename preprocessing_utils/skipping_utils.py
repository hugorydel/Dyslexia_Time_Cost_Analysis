"""
Simple skipping analysis utilities for dyslexia research
Uses ExtractedFeatures data directly
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_skipping_from_extracted_features(data: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate skipping probabilities directly from ExtractedFeatures data

    Args:
        data: DataFrame with ExtractedFeatures data including n_fixations column

    Returns:
        Dictionary with skipping analysis results
    """
    # Use the mapped column name (n_fixations, not number_of_fixations)
    fixation_col = "n_fixations"

    if fixation_col not in data.columns:
        logger.warning(f"No {fixation_col} column found - cannot calculate skipping")
        return {}

    # Simple approach: words with 0 fixations were skipped
    data["skipped"] = data[fixation_col] == 0
    data["was_fixated"] = data[fixation_col] > 0

    # Calculate overall statistics
    total_words = len(data)
    skipped_words = data["skipped"].sum()
    fixated_words = data["was_fixated"].sum()
    overall_skipping_rate = skipped_words / total_words if total_words > 0 else 0

    logger.info(f"Skipping analysis from ExtractedFeatures:")
    logger.info(f"  Total words: {total_words:,}")
    logger.info(f"  Skipped: {skipped_words:,} ({overall_skipping_rate*100:.1f}%)")
    logger.info(f"  Fixated: {fixated_words:,} ({(1-overall_skipping_rate)*100:.1f}%)")

    results = {
        "overall_skipping_rate": overall_skipping_rate,
        "total_words": total_words,
        "skipped_words": int(skipped_words),
        "fixated_words": int(fixated_words),
        "analysis_method": "direct_from_extracted_features",
    }

    # Group-level analysis if dyslexic column exists
    if "dyslexic" in data.columns:
        group_results = analyze_skipping_by_groups_simple(data)
        results.update(group_results)

    return results


def analyze_skipping_by_groups_simple(data: pd.DataFrame) -> Dict[str, any]:
    """
    Analyze skipping patterns by dyslexic groups using simple approach
    """
    results = {}

    # Word-level skipping by group
    group_skipping = data.groupby("dyslexic")["skipped"].agg(["mean", "std", "count"])

    results["skipping_by_group"] = {}
    if False in group_skipping.index:
        results["skipping_by_group"]["control"] = {
            "mean": float(group_skipping.loc[False, "mean"]),
            "std": float(group_skipping.loc[False, "std"]),
            "count": int(group_skipping.loc[False, "count"]),
        }

    if True in group_skipping.index:
        results["skipping_by_group"]["dyslexic"] = {
            "mean": float(group_skipping.loc[True, "mean"]),
            "std": float(group_skipping.loc[True, "std"]),
            "count": int(group_skipping.loc[True, "count"]),
        }

    # Subject-level skipping rates
    subject_skipping = (
        data.groupby(["subject_id", "dyslexic"])["skipped"].mean().reset_index()
    )
    subject_group_skipping = subject_skipping.groupby("dyslexic")["skipped"].agg(
        ["mean", "std", "count"]
    )

    results["subject_level_skipping_by_group"] = {}
    if False in subject_group_skipping.index:
        results["subject_level_skipping_by_group"]["control"] = {
            "mean": float(subject_group_skipping.loc[False, "mean"]),
            "std": float(subject_group_skipping.loc[False, "std"]),
            "n_subjects": int(subject_group_skipping.loc[False, "count"]),
        }

    if True in subject_group_skipping.index:
        results["subject_level_skipping_by_group"]["dyslexic"] = {
            "mean": float(subject_group_skipping.loc[True, "mean"]),
            "std": float(subject_group_skipping.loc[True, "std"]),
            "n_subjects": int(subject_group_skipping.loc[True, "count"]),
        }

    return results
