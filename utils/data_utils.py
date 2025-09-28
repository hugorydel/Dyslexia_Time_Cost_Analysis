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

    Args:
        features_path: Path to ExtractedFeatures directory

    Returns:
        Combined DataFrame with all participant data
    """
    csv_files = list(features_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {features_path}")

    logger.info(f"Loading {len(csv_files)} participant feature files...")

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

    logger.info(f"Combined word-level data shape: {combined_data.shape}")
    logger.info(f"Columns: {list(combined_data.columns)}")

    return combined_data


def load_participant_stats(stats_path: Path) -> pd.DataFrame:
    """
    Load participant statistics for group assignment

    Args:
        stats_path: Path to DatasetStatistics directory

    Returns:
        DataFrame with participant statistics
    """
    participant_stats_file = stats_path / "participant_stats.csv"

    if participant_stats_file.exists():
        try:
            stats_df = pd.read_csv(participant_stats_file)

            # Apply column mapping
            stats_df = apply_mapping(stats_df, "participant_stats")

            logger.info(f"Loaded participant stats: {stats_df.shape}")
            return stats_df
        except Exception as e:
            logger.warning(f"Error loading participant stats: {e}")

    return pd.DataFrame()


def merge_with_participant_stats(
    word_data: pd.DataFrame, participant_stats: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge word data with participant statistics

    Args:
        word_data: DataFrame with word-level data
        participant_stats: DataFrame with participant statistics

    Returns:
        Merged DataFrame
    """
    if participant_stats.empty:
        return word_data

    try:
        merged = word_data.merge(
            participant_stats,
            on="subject_id",
            how="left",
            suffixes=("", "_stats"),
        )
        logger.info(f"Merged with participant stats. New shape: {merged.shape}")
        return merged

    except Exception as e:
        logger.warning(f"Could not merge with participant stats: {e}")
        return word_data


def identify_dyslexic_subjects(
    data: pd.DataFrame, participant_stats: pd.DataFrame
) -> Set[str]:
    """
    Identify dyslexic subjects from participant statistics

    Args:
        data: Word-level data (for fallback subject list)
        participant_stats: Participant statistics with dyslexia labels

    Returns:
        Set of subject IDs for dyslexic participants

    Raises:
        ValueError: If dyslexic subjects cannot be identified
    """
    if not participant_stats.empty and "dyslexia" in participant_stats.columns:
        logger.info("Found 'dyslexia' column in participant stats!")

        dyslexia_values = participant_stats["dyslexia"].value_counts()
        logger.info(f"Dyslexia column values: {dyslexia_values.to_dict()}")

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
                f"Dyslexic subjects: {len(dyslexic_subjects)} - {sorted(list(dyslexic_subjects))}"
            )
            logger.info(
                f"Control subjects: {len(control_subjects)} - {sorted(list(control_subjects))}"
            )

            return dyslexic_subjects

    # No fallback - raise error for proper scientific practice
    raise ValueError(
        "Cannot identify dyslexic vs control subjects. "
        "Required participant statistics with 'dyslexia' column not found or invalid. "
        "Please ensure participant_stats.csv contains a 'dyslexia' column with clear group labels."
    )


def clean_word_data(data: pd.DataFrame, config) -> pd.DataFrame:
    """
    Clean and filter word-level data

    Args:
        data: Raw word-level data
        config: Configuration object with filtering parameters

    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning word-level data...")

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

    logger.info(f"Filtered data: {initial_rows} -> {len(data)} rows")

    if len(data) == 0:
        raise ValueError("No data remaining after preprocessing")

    return data


def create_additional_measures(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional measures for analysis

    Args:
        data: DataFrame with basic eye-tracking measures

    Returns:
        DataFrame with additional computed measures
    """
    # Note: Skipping probability cannot be calculated from ExtractedFeatures alone
    # because this dataset only contains words that were actually fixated.
    # Skipped words are not included in this data.
    # For proper skipping analysis, we would need FixationReports or InterestAreaReports.

    # Mark all words in this dataset as fixated (by definition)
    if "n_fixations" in data.columns:
        data["was_fixated"] = data["n_fixations"] > 0
        # Set skipping to NaN to indicate we cannot calculate this measure
        data["skipping_probability"] = np.nan
        logger.warning(
            "Skipping probability cannot be calculated from ExtractedFeatures data - "
            "this dataset only contains fixated words. Skipped words are not included."
        )
    else:
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
