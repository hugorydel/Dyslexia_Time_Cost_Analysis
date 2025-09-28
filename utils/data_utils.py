"""
Data loading and preprocessing utilities for dyslexia analysis
"""

import logging
import re
from pathlib import Path
from typing import Optional, Set

import numpy as np
import pandas as pd

from column_mappings import apply_mapping

logger = logging.getLogger(__name__)


def load_extracted_features(features_path: Path) -> pd.DataFrame:
    """
    Load all ExtractedFeatures CSV files - these contain the actual word-level data
    """
    csv_files = list(features_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {features_path}")

    logger.info(f"Loading {len(csv_files)} participant files...")

    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Extract subject ID from filename (P01, P02, etc.)
            p_match = re.search(r"P(\d+)", csv_file.stem)
            if p_match:
                subject_id = f"P{p_match.group(1).zfill(2)}"
            else:
                subject_id = csv_file.stem
            df["subject_id"] = subject_id
            all_data.append(df)
        except Exception as e:
            logger.warning(f"Error loading {csv_file.name}: {e}")
            continue

    if not all_data:
        raise ValueError("No feature files could be loaded successfully")

    combined_data = pd.concat(all_data, ignore_index=True)
    # Apply column mapping
    combined_data = apply_mapping(combined_data, "extracted_features")

    return combined_data


def load_participant_stats(stats_path: Path) -> pd.DataFrame:
    """Load participant statistics for group assignment"""
    participant_stats_file = stats_path / "participant_stats.csv"

    if participant_stats_file.exists():
        try:
            stats_df = pd.read_csv(participant_stats_file)
            # Apply column mapping
            stats_df = apply_mapping(stats_df, "participant_stats")
            return stats_df
        except Exception as e:
            logger.warning(f"Error loading participant stats: {e}")

    return pd.DataFrame()


def merge_with_participant_stats(
    word_data: pd.DataFrame, participant_stats: pd.DataFrame
) -> pd.DataFrame:
    """Merge word data with participant statistics"""
    if participant_stats.empty:
        return word_data

    try:
        merged = word_data.merge(
            participant_stats,
            on="subject_id",
            how="left",
            suffixes=("", "_stats"),
        )
        return merged
    except Exception as e:
        logger.warning(f"Could not merge with participant stats: {e}")
        return word_data


def identify_dyslexic_subjects(
    data: pd.DataFrame, participant_stats: pd.DataFrame
) -> tuple[Set[str], dict[str, list]]:
    """
    Identify dyslexic subjects from participant statistics

    Args:
        data: Word-level data (for fallback subject list)
        participant_stats: Participant statistics with dyslexia labels

    Returns:
        Tuple of (dyslexic_subject_set, subject_lists_dict)
        subject_lists_dict contains 'dyslexic' and 'control' keys with sorted lists

    Raises:
        ValueError: If dyslexic subjects cannot be identified
    """
    if not participant_stats.empty and "dyslexia" in participant_stats.columns:
        # Create dyslexic mask
        dyslexic_mask = (
            participant_stats["dyslexia"]
            .astype(str)
            .str.lower()
            .isin(["yes", "y", "1", "true", "dyslexic"])
        )

        if dyslexic_mask.any():
            dyslexic_subjects = set(
                participant_stats[dyslexic_mask]["subject_id"].values
            )
            control_subjects = set(
                participant_stats[~dyslexic_mask]["subject_id"].values
            )

            logger.info(
                f"Groups identified: {len(dyslexic_subjects)} dyslexic, {len(control_subjects)} control"
            )

            # Return subject lists for JSON output
            subject_lists = {
                "dyslexic": sorted(list(dyslexic_subjects)),
                "control": sorted(list(control_subjects)),
            }

            return dyslexic_subjects, subject_lists

    # No fallback - raise error for proper scientific practice
    raise ValueError(
        "Cannot identify dyslexic vs control subjects. "
        "Required participant statistics with 'dyslexia' column not found or invalid."
    )


def clean_word_data(data: pd.DataFrame, config) -> pd.DataFrame:
    """
    Clean and filter word-level data while preserving skipped words
    """
    if data.empty:
        raise ValueError("Input data is empty")

    initial_rows = len(data)
    logger.info(f"Starting with {initial_rows:,} total word entries")

    # Step 1: Remove rows with missing critical identifiers
    data = data.dropna(subset=["subject_id", "word_text"])
    data = data[data["word_text"].astype(str).str.strip() != ""]

    after_missing = len(data)
    if initial_rows > after_missing:
        logger.info(
            f"Removed {initial_rows - after_missing:,} words with missing identifiers"
        )

    # Step 2: Identify skipped vs fixated words BEFORE removing reading times
    # Use the mapped column name (n_fixations, not number_of_fixations)
    fixation_col = "n_fixations"

    if fixation_col in data.columns:
        # Identify skipped words (0 fixations) vs fixated words (1+ fixations)
        skipped_words = (data[fixation_col] == 0).sum()
        fixated_words = (data[fixation_col] > 0).sum()
        nan_fixations = data[fixation_col].isna().sum()

        logger.info(f"Word fixation patterns:")
        logger.info(f"  - Skipped words (0 fixations): {skipped_words:,}")
        logger.info(f"  - Fixated words (1+ fixations): {fixated_words:,}")
        logger.info(f"  - Unknown fixations (NaN): {nan_fixations:,}")

        # Remove only words with NaN fixation counts (true missing data)
        if nan_fixations > 0:
            data = data.dropna(subset=[fixation_col])
            logger.info(f"Removed {nan_fixations:,} words with unknown fixation status")

        # Now handle reading times appropriately
        if "total_reading_time" in data.columns:
            # For fixated words, reading time should exist
            fixated_mask = data[fixation_col] > 0
            fixated_with_nan_rt = fixated_mask & data["total_reading_time"].isna()

            if fixated_with_nan_rt.sum() > 0:
                logger.info(
                    f"Found {fixated_with_nan_rt.sum():,} fixated words with missing reading times - removing"
                )
                data = data[~fixated_with_nan_rt]

            # For skipped words, NaN reading time is expected and OK
            skipped_mask = data[fixation_col] == 0
            skipped_with_rt = skipped_mask & data["total_reading_time"].notna()

            if skipped_with_rt.sum() > 0:
                logger.warning(
                    f"Found {skipped_with_rt.sum():,} skipped words with reading times - unusual but keeping"
                )

            # Apply reading time filters ONLY to fixated words
            if fixated_mask.sum() > 0:
                max_duration = getattr(config, "MAX_FIXATION_DURATION", 8000)
                min_duration = getattr(config, "MIN_FIXATION_DURATION", 80)

                fixated_data = data[fixated_mask]
                rt_99th = fixated_data["total_reading_time"].quantile(0.99)
                logger.info(
                    f"Fixated words reading times: median {fixated_data['total_reading_time'].median():.0f}ms, 99th percentile {rt_99th:.0f}ms"
                )

                # Count outliers among fixated words only
                too_long = (fixated_data["total_reading_time"] > max_duration).sum()
                too_short = (fixated_data["total_reading_time"] < min_duration).sum()

                # Apply filters only to fixated words
                outlier_mask = fixated_mask & (
                    (data["total_reading_time"] < min_duration)
                    | (data["total_reading_time"] > max_duration)
                )

                if outlier_mask.sum() > 0:
                    logger.info(
                        f"Removed {outlier_mask.sum():,} fixated words with extreme reading times ({too_short:,} too short, {too_long:,} too long)"
                    )
                    data = data[~outlier_mask]

    else:
        logger.warning(
            f"No {fixation_col} column found - cannot distinguish skipped vs fixated words"
        )

    # Step 3: Filter extremely long words (data entry errors)
    if "word_text" in data.columns and hasattr(config, "MAX_WORD_LENGTH"):
        word_lengths = data["word_text"].str.len()
        max_length = config.MAX_WORD_LENGTH

        too_long_words = (word_lengths > max_length).sum()
        if too_long_words > 0:
            logger.info(
                f"Removed {too_long_words:,} words longer than {max_length} characters (likely data errors)"
            )
            data = data[word_lengths <= max_length]

    final_rows = len(data)
    retention_rate = final_rows / initial_rows * 100

    # Final breakdown
    if fixation_col in data.columns:
        final_skipped = (data[fixation_col] == 0).sum()
        final_fixated = (data[fixation_col] > 0).sum()
        logger.info(
            f"Final dataset: {final_skipped:,} skipped words, {final_fixated:,} fixated words"
        )

    logger.info(
        f"Data cleaning complete: {final_rows:,} words retained ({retention_rate:.1f}% of original)"
    )

    if len(data) == 0:
        raise ValueError("No data remaining after preprocessing")

    return data


def create_additional_measures(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional measures for analysis using correct column names
    """
    # Use the mapped column name (n_fixations, not number_of_fixations)
    fixation_col = "n_fixations"

    # Simple skipping analysis: words with 0 fixations were skipped
    if fixation_col in data.columns:
        data["skipped"] = data[fixation_col] == 0
        data["was_fixated"] = data[fixation_col] > 0
        data["skipping_probability"] = data["skipped"].astype(float)
    else:
        data["skipped"] = np.nan
        data["was_fixated"] = np.nan
        data["skipping_probability"] = np.nan

    # Regression probability - can be estimated from go-past time vs gaze duration
    if "word_go_past_time" in data.columns and "gaze_duration" in data.columns:
        # If go-past time > gaze duration, there was likely a regression
        data["regression_probability"] = (
            data["word_go_past_time"] > data["gaze_duration"]
        ).astype(float)
    else:
        data["regression_probability"] = 0.1  # Placeholder

    # Word length
    if "word_text" in data.columns:
        data["word_length"] = data["word_text"].str.len()

    return data
