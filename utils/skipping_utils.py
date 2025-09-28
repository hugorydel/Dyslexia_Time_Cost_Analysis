"""
Word skipping analysis utilities for dyslexia research
Calculates proper skipping probabilities by comparing full text with fixated words
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_results_files(copco_path: Path) -> pd.DataFrame:
    """Load RESULTS_FILE.txt files to get complete text for skipping analysis"""
    raw_data_path = copco_path / "RawData"

    if not raw_data_path.exists():
        logger.warning(f"RawData directory not found at {raw_data_path}")
        return pd.DataFrame()

    all_raw_data = []
    participant_dirs = list(raw_data_path.glob("P*"))

    for participant_dir in participant_dirs:
        # Extract participant ID
        participant_match = re.search(r"P(\d+)", participant_dir.name)
        if not participant_match:
            continue

        subject_id = f"P{participant_match.group(1).zfill(2)}"

        # Look for RESULTS_FILE.txt
        results_files = list(participant_dir.glob("RESULTS_FILE.txt"))
        if not results_files:
            continue

        try:
            # Load RESULTS_FILE.txt - tab separated with headers
            df = pd.read_csv(results_files[0], sep="\t", encoding="utf-8")
            df["subject_id"] = subject_id

            # Keep only experiment trials (not practice)
            if "condition" in df.columns:
                df = df[df["condition"] == "experiment"]

            # Clean up text column - remove quotes
            if "text" in df.columns:
                df["text"] = df["text"].str.strip('"')

            all_raw_data.append(df)

        except Exception as e:
            logger.warning(f"Error loading {results_files[0]}: {e}")
            continue

    if not all_raw_data:
        logger.warning("No raw data could be loaded")
        return pd.DataFrame()

    raw_df = pd.concat(all_raw_data, ignore_index=True)
    return raw_df


def simple_tokenize(text: str) -> List[str]:
    """
    Simple tokenization that should match eye-tracking word segmentation

    Args:
        text: Input text string

    Returns:
        List of word tokens
    """
    if pd.isna(text) or text == "":
        return []

    # Remove quotes and normalize whitespace
    text = str(text).strip()
    text = re.sub(r'["""]', "", text)
    text = re.sub(r"\s+", " ", text)

    # Split on whitespace and punctuation, but preserve word boundaries
    # This is a simple approach that should be close to eye-tracking segmentation
    words = []

    # Split by whitespace first
    tokens = text.split()

    for token in tokens:
        # Further split on punctuation but keep the parts
        # Use a simple approach: split on major punctuation
        parts = re.split(r'([.,:;!?()"\'-])', token)
        for part in parts:
            part = part.strip()
            if part and part not in ".,;:!?()\"'-":
                words.append(part)

    return words


def create_trial_word_lists(results_data: pd.DataFrame) -> pd.DataFrame:
    """Create word lists for each trial from RESULTS_FILE data"""
    trial_words = []

    for _, trial in results_data.iterrows():
        if pd.isna(trial.get("text", "")):
            continue

        words = simple_tokenize(trial["text"])

        # Use Trial_Index_ as the key for matching with ExtractedFeatures
        trial_id = trial.get("Trial_Index_", trial.get("Trial_Index", -1))
        speech_id = trial.get("speechid", -1)
        paragraph_id = trial.get("paragraphid", -1)

        for word_pos, word in enumerate(words):
            if word.strip():  # Only include non-empty words
                trial_words.append(
                    {
                        "subject_id": trial["subject_id"],
                        "trial_id": trial_id,
                        "speech_id": speech_id,
                        "paragraph_id": paragraph_id,
                        "word_position": word_pos + 1,  # 1-indexed
                        "word_text": word.lower().strip(),  # Normalize case
                        "condition": trial.get("condition", "unknown"),
                    }
                )

    return pd.DataFrame(trial_words)


def calculate_trial_based_skipping(
    extracted_features: pd.DataFrame, trial_words: pd.DataFrame
) -> Dict[str, any]:
    """Calculate skipping probabilities using trial-based matching"""

    # Group fixated words by trial for faster lookup
    fixated_by_trial = {}
    for _, row in extracted_features.iterrows():
        trial_key = (row["subject_id"], row.get("trial_id", -1))
        if trial_key not in fixated_by_trial:
            fixated_by_trial[trial_key] = set()

        # Normalize word text for comparison
        word = str(row.get("word_text", "")).lower().strip()
        word_pos = row.get("word_position", -1)

        # Add both word text and position-based keys for matching
        fixated_by_trial[trial_key].add(word)
        fixated_by_trial[trial_key].add(f"{word_pos}_{word}")

    # Calculate skipping for each word in trial_words
    skipped_words = []
    fixated_words = []

    for _, word_row in trial_words.iterrows():
        trial_key = (word_row["subject_id"], word_row["trial_id"])
        word = word_row["word_text"].lower().strip()
        word_pos = word_row["word_position"]

        # Check if this word was fixated in this trial
        fixated_in_trial = fixated_by_trial.get(trial_key, set())

        was_fixated = (
            word in fixated_in_trial or f"{word_pos}_{word}" in fixated_in_trial
        )

        if was_fixated:
            fixated_words.append(word_row)
        else:
            skipped_words.append(word_row)

    total_words = len(trial_words)
    n_fixated = len(fixated_words)
    n_skipped = len(skipped_words)
    skipping_rate = n_skipped / total_words if total_words > 0 else 0

    return {
        "overall_skipping_rate": skipping_rate,
        "total_words": total_words,
        "fixated_words": n_fixated,
        "skipped_words": n_skipped,
        "trial_words_df": trial_words,
        "fixated_words_df": (
            pd.DataFrame(fixated_words) if fixated_words else pd.DataFrame()
        ),
        "skipped_words_df": (
            pd.DataFrame(skipped_words) if skipped_words else pd.DataFrame()
        ),
    }


def analyze_skipping_by_groups(
    skipping_results: Dict, extracted_features: pd.DataFrame
) -> Dict[str, any]:
    """
    Analyze skipping patterns by dyslexic groups

    Args:
        skipping_results: Results from calculate_trial_based_skipping
        extracted_features: ExtractedFeatures data with dyslexic labels

    Returns:
        Dictionary with group-level skipping analysis
    """
    trial_words_df = skipping_results["trial_words_df"]

    if trial_words_df.empty:
        return {}

    # Get dyslexic labels from extracted_features
    subject_dyslexic = (
        extracted_features.groupby("subject_id")["dyslexic"].first().to_dict()
    )

    # Add dyslexic labels to trial words
    trial_words_df = trial_words_df.copy()
    trial_words_df["dyslexic"] = trial_words_df["subject_id"].map(subject_dyslexic)

    # Remove words where we don't have dyslexic labels
    trial_words_df = trial_words_df.dropna(subset=["dyslexic"])

    if trial_words_df.empty:
        return {}

    # Mark which words were skipped
    skipped_words_df = skipping_results.get("skipped_words_df", pd.DataFrame())
    if not skipped_words_df.empty:
        # Create a key for skipped words
        skipped_keys = set(
            zip(
                skipped_words_df["subject_id"],
                skipped_words_df["trial_id"],
                skipped_words_df["word_position"],
            )
        )

        trial_words_df["skipped"] = trial_words_df.apply(
            lambda row: (row["subject_id"], row["trial_id"], row["word_position"])
            in skipped_keys,
            axis=1,
        )
    else:
        trial_words_df["skipped"] = False

    results = {}

    # Word-level skipping by group
    if "dyslexic" in trial_words_df.columns:
        group_skipping = trial_words_df.groupby("dyslexic")["skipped"].agg(
            ["mean", "std", "count"]
        )

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
            trial_words_df.groupby(["subject_id", "dyslexic"])["skipped"]
            .mean()
            .reset_index()
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


def enhanced_skipping_analysis(
    copco_path: Path, extracted_features: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict]:
    """Complete pipeline for enhanced skipping analysis using trial-based matching"""
    try:
        # Load RESULTS_FILE data
        results_data = load_results_files(copco_path)

        if results_data.empty:
            return extracted_features, {}

        # Create trial word lists
        trial_words = create_trial_word_lists(results_data)

        if trial_words.empty:
            return extracted_features, {}

        # Calculate trial-based skipping
        skipping_results = calculate_trial_based_skipping(
            extracted_features, trial_words
        )

        # Analyze by groups
        group_analysis = analyze_skipping_by_groups(
            skipping_results, extracted_features
        )

        # Combine results
        final_results = {**skipping_results, **group_analysis}

        # Update extracted_features to reflect that these are fixated words
        enhanced_features = extracted_features.copy()
        enhanced_features["skipping_probability"] = (
            0.0  # All words in ExtractedFeatures were fixated
        )
        enhanced_features["skipped"] = False

        return enhanced_features, final_results

    except Exception as e:
        logger.error(f"Error in enhanced skipping analysis: {e}")
        return extracted_features, {}
