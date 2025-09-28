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
    """Clean and filter word-level data"""
    if data.empty:
        raise ValueError("Input data is empty")

    initial_rows = len(data)

    # Remove rows with missing critical data
    data = data.dropna(subset=["subject_id", "word_text"])
    data = data[data["word_text"].astype(str).str.strip() != ""]

    # Filter extreme outliers in eye measures
    if "total_reading_time" in data.columns:
        max_duration = getattr(config, "MAX_FIXATION_DURATION", 5000)
        data = data[
            (data["total_reading_time"] >= 50)
            & (data["total_reading_time"] <= max_duration)
        ]

    # Filter extremely long words if specified
    if "word_text" in data.columns and hasattr(config, "MAX_WORD_LENGTH"):
        word_lengths = data["word_text"].str.len()
        data = data[word_lengths <= config.MAX_WORD_LENGTH]

    if len(data) == 0:
        raise ValueError("No data remaining after preprocessing")

    filtered_count = initial_rows - len(data)
    if filtered_count > 0:
        logger.info(
            f"Filtered {filtered_count:,} outlier words ({filtered_count/initial_rows*100:.1f}%)"
        )

    return data


def create_additional_measures(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional measures for analysis

    Args:
        data: DataFrame with basic eye-tracking measures

    Returns:
        DataFrame with additional computed measures
    """
    # Mark all words in this dataset as fixated (by definition of ExtractedFeatures)
    if "n_fixations" in data.columns:
        data["was_fixated"] = data["n_fixations"] > 0
        # Initialize skipping probability - will be updated by skipping analysis if available
        data["skipping_probability"] = (
            0.0  # For fixated words, skipping probability is 0
        )
        data["skipped"] = False
    else:
        data["skipping_probability"] = np.nan
        data["skipped"] = np.nan

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


def create_text_data_from_words(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create text data by reconstructing sentences from word sequences

    Args:
        data: Word-level data with trial and position information

    Returns:
        DataFrame with reconstructed sentence text
    """
    sentence_data = []

    for (subject_id, trial_id), trial_data in data.groupby(["subject_id", "trial_id"]):
        trial_words = trial_data.sort_values("word_position")["word_text"]
        sentence_text = " ".join(trial_words.astype(str))

        sentence_data.append(
            {
                "subject_id": subject_id,
                "trial_id": trial_id,
                "sentence_text": sentence_text,
                "n_words": len(trial_words),
            }
        )

    return pd.DataFrame(sentence_data)
