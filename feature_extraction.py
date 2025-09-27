#!/usr/bin/env python3
"""
Advanced Feature Extraction for Dyslexia Time Cost Analysis
Extracts the four key predictors: Length, Frequency, Preview, Predictability
"""

import logging
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Comprehensive feature extraction for word-level reading analysis
    """

    def __init__(self, config):
        self.config = config
        self.nlp = None
        self.frequency_dict = {}
        self.setup_nlp_models()

    def setup_nlp_models(self):
        """Initialize NLP models and resources"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✓ spaCy model loaded")
        except OSError:
            logger.warning("spaCy model not found, using basic tokenization")

    def extract_word_features(
        self, fixation_data: pd.DataFrame, text_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract all four key features for each word fixation

        Parameters:
        -----------
        fixation_data : DataFrame with columns like:
            - subject_id, trial_id, word_id, fixation_duration,
            - x_position, y_position, word_text, etc.
        text_data : DataFrame with sentence/text information

        Returns:
        --------
        DataFrame with added feature columns
        """
        logger.info("Starting comprehensive feature extraction...")

        # Create copy to avoid modifying original
        data = fixation_data.copy()

        # 1. Word Length Features
        data = self._extract_length_features(data)

        # 2. Lexical Frequency Features
        data = self._extract_frequency_features(data)

        # 3. Parafoveal Preview Features
        data = self._extract_preview_features(data)

        # 4. Contextual Predictability Features
        data = self._extract_predictability_features(data, text_data)

        # 5. Additional control variables
        data = self._extract_control_features(data)

        logger.info(f"Feature extraction completed. Shape: {data.shape}")
        return data

    def _extract_length_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract word length related features"""
        logger.info("Extracting length features...")

        # Basic length
        data["word_length"] = data["word_text"].str.len()

        # Length categories for analysis
        data["length_category"] = pd.cut(
            data["word_length"],
            bins=[0, 3, 5, 8, np.inf],
            labels=["short", "medium", "long", "very_long"],
        )

        # Character composition features
        data["n_vowels"] = data["word_text"].apply(
            lambda x: len(re.findall(r"[aeiouAEIOU]", x))
        )
        data["n_consonants"] = data["word_length"] - data["n_vowels"]

        # Orthographic complexity
        data["has_double_letters"] = data["word_text"].apply(
            lambda x: bool(re.search(r"(.)\1", x))
        )

        return data

    def _extract_frequency_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract lexical frequency features"""
        logger.info("Extracting frequency features...")

        if not self.frequency_dict:
            self._build_frequency_dict(data)

        # Map frequencies
        data["log_frequency"] = (
            data["word_text"]
            .str.lower()
            .map(lambda x: np.log(self.frequency_dict.get(x, 1) + 1))
        )

        # Frequency categories
        freq_quantiles = data["log_frequency"].quantile([0.25, 0.5, 0.75])
        data["frequency_category"] = pd.cut(
            data["log_frequency"],
            bins=[
                -np.inf,
                freq_quantiles[0.25],
                freq_quantiles[0.5],
                freq_quantiles[0.75],
                np.inf,
            ],
            labels=["very_low", "low", "medium", "high"],
        )

        # Word class information (if available)
        if self.nlp:
            data["pos_tag"] = self._get_pos_tags(data["word_text"])
            data["is_content_word"] = data["pos_tag"].isin(
                ["NOUN", "VERB", "ADJ", "ADV"]
            )

        return data

    def _extract_preview_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract parafoveal preview related features"""
        logger.info("Extracting preview features...")

        # Sort by subject, trial, and word position for sequential analysis
        data = data.sort_values(["subject_id", "trial_id", "word_position"])

        # Launch site distance (distance from previous fixation)
        data["prev_x_position"] = data.groupby(["subject_id", "trial_id"])[
            "x_position"
        ].shift(1)
        data["launch_site_distance"] = abs(data["x_position"] - data["prev_x_position"])

        # Handle first fixations in trials (no previous fixation)
        data["launch_site_distance"].fillna(0, inplace=True)

        # Launch site categories based on literature
        launch_bins = self.config.get("LAUNCH_SITE_BINS", [0, 2, 4, 8, 15])
        data["launch_distance_category"] = pd.cut(
            data["launch_site_distance"],
            bins=launch_bins + [np.inf],
            labels=[f"dist_{i}" for i in range(len(launch_bins))],
        )

        # Word spacing (approximate from x-positions and word lengths)
        data["next_word_x"] = data.groupby(["subject_id", "trial_id"])[
            "x_position"
        ].shift(-1)
        data["word_spacing"] = data["next_word_x"] - (
            data["x_position"] + data["word_length"] * 8
        )  # Assume ~8px per char

        # Preview benefit proxy: combination of launch distance and target length
        data["preview_benefit_score"] = (
            1
            / (1 + data["launch_site_distance"])  # Closer = better preview
            * 1
            / (1 + data["word_length"] / 5)  # Shorter = easier to preview
        )

        # Crowding effects
        data["visual_crowding"] = data["word_length"] * (
            1 / (1 + data["word_spacing"].fillna(10))
        )  # Closer spacing = more crowding

        return data

    def _extract_predictability_features(
        self, data: pd.DataFrame, text_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract contextual predictability features"""
        logger.info("Extracting predictability features...")

        # Group by trial to process sentences
        predictability_scores = []

        for trial_id, trial_data in tqdm(
            data.groupby("trial_id"), desc="Computing predictability"
        ):

            # Get sentence text
            sentence = self._get_sentence_text(trial_id, text_data)
            if not sentence:
                # Fallback: reconstruct from word_text
                sentence = " ".join(
                    trial_data.sort_values("word_position")["word_text"]
                )

            # Compute predictability scores for each word
            word_predictabilities = self._compute_word_predictabilities(sentence)

            # Match with trial data
            for idx, (_, row) in enumerate(trial_data.iterrows()):
                if idx < len(word_predictabilities):
                    predictability_scores.append(word_predictabilities[idx])
                else:
                    predictability_scores.append(0.5)  # Default moderate predictability

        data["predictability"] = predictability_scores
        data["surprisal"] = -np.log(np.maximum(data["predictability"], 1e-10))

        # Predictability categories
        pred_quantiles = data["predictability"].quantile([0.25, 0.5, 0.75])
        data["predictability_category"] = pd.cut(
            data["predictability"],
            bins=[
                0,
                pred_quantiles[0.25],
                pred_quantiles[0.5],
                pred_quantiles[0.75],
                1.0,
            ],
            labels=["very_low", "low", "medium", "high"],
        )

        # Position in sentence (early words often less predictable)
        data["word_position_norm"] = data.groupby("trial_id")[
            "word_position"
        ].transform(lambda x: x / x.max())

        return data

    def _extract_control_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract additional control variables"""
        logger.info("Extracting control features...")

        # Trial-level features
        data["trial_position"] = data.groupby("subject_id")["trial_id"].transform(
            lambda x: pd.factorize(x)[0] + 1
        )
        data["trial_position_norm"] = data.groupby("subject_id")[
            "trial_position"
        ].transform(lambda x: x / x.max())

        # Previous word effects (spillover)
        data["prev_word_length"] = data.groupby(["subject_id", "trial_id"])[
            "word_length"
        ].shift(1)
        data["prev_log_frequency"] = data.groupby(["subject_id", "trial_id"])[
            "log_frequency"
        ].shift(1)

        # Reading direction indicators
        data["is_regression"] = (
            data.groupby(["subject_id", "trial_id"])["word_position"].diff() < 0
        )
        data["is_first_pass"] = (
            ~data.groupby(["subject_id", "trial_id", "word_position"])
            .cumcount()
            .astype(bool)
        )

        return data

    def _build_frequency_dict(self, data: pd.DataFrame):
        """Build frequency dictionary from corpus or external source"""
        logger.info("Building frequency dictionary...")

        # Option 1: Use word frequencies from the current corpus
        word_counts = Counter(data["word_text"].str.lower())

        # Option 2: Load from external corpus (e.g., Google Books, SUBTLEX)
        # For now, using simple corpus frequency
        total_words = sum(word_counts.values())
        self.frequency_dict = {
            word: count / total_words for word, count in word_counts.items()
        }

        logger.info(f"Frequency dictionary built with {len(self.frequency_dict)} words")

    def _get_pos_tags(self, words: pd.Series) -> pd.Series:
        """Get part-of-speech tags using spaCy"""
        if not self.nlp:
            return pd.Series(["UNKNOWN"] * len(words))

        pos_tags = []
        for word in tqdm(words, desc="POS tagging"):
            doc = self.nlp(str(word))
            if doc:
                pos_tags.append(doc[0].pos_)
            else:
                pos_tags.append("UNKNOWN")

        return pd.Series(pos_tags)

    def _get_sentence_text(self, trial_id: int, text_data: pd.DataFrame) -> str:
        """Extract sentence text for given trial"""
        if text_data is not None and not text_data.empty:
            trial_text = text_data[text_data["trial_id"] == trial_id]
            if not trial_text.empty and "sentence_text" in trial_text.columns:
                return trial_text.iloc[0]["sentence_text"]
        return ""

    def _compute_word_predictabilities(self, sentence: str) -> List[float]:
        """
        Compute predictability scores for each word in sentence
        Using simple n-gram approach (can be enhanced with transformers)
        """
        words = sentence.split()
        predictabilities = []

        for i, word in enumerate(words):
            if i == 0:
                # First word - use corpus frequency as proxy
                pred = self.frequency_dict.get(word.lower(), 0.001)
            else:
                # Use bigram/trigram predictability (simplified)
                context = " ".join(words[max(0, i - 2) : i])
                pred = self._estimate_conditional_probability(word, context)

            predictabilities.append(min(max(pred, 0.001), 0.999))  # Clamp values

        return predictabilities

    def _estimate_conditional_probability(self, word: str, context: str) -> float:
        """Estimate P(word|context) using simple heuristic"""
        # Simplified implementation - in practice, use proper language model
        word_freq = self.frequency_dict.get(word.lower(), 0.001)

        # Boost probability if word is common given POS context
        if self.nlp:
            context_doc = self.nlp(context)
            word_doc = self.nlp(word)

            if context_doc and word_doc:
                # Simple heuristic based on POS sequence likelihood
                boost = 1.0
                if (
                    context_doc[-1].pos_ in ["DET", "ADJ"]
                    and word_doc[0].pos_ == "NOUN"
                ):
                    boost = 2.0
                elif (
                    context_doc[-1].pos_ in ["AUX", "MODAL"]
                    and word_doc[0].pos_ == "VERB"
                ):
                    boost = 2.0

                return min(word_freq * boost, 0.999)

        return word_freq

    def save_features(self, data: pd.DataFrame, filepath: str):
        """Save processed features to file"""
        data.to_pickle(filepath)
        logger.info(f"Features saved to {filepath}")

    def load_features(self, filepath: str) -> pd.DataFrame:
        """Load preprocessed features"""
        data = pd.read_pickle(filepath)
        logger.info(f"Features loaded from {filepath}")
        return data

    def get_feature_summary(self, data: pd.DataFrame) -> Dict:
        """Generate summary statistics of extracted features"""
        feature_cols = [
            "word_length",
            "log_frequency",
            "launch_site_distance",
            "predictability",
            "surprisal",
            "preview_benefit_score",
        ]

        summary = {}
        for col in feature_cols:
            if col in data.columns:
                summary[col] = {
                    "mean": data[col].mean(),
                    "std": data[col].std(),
                    "min": data[col].min(),
                    "max": data[col].max(),
                    "missing": data[col].isnull().sum(),
                }

        return summary


def main():
    """Test feature extraction with sample data"""
    # This would normally be called from main.py
    pass


if __name__ == "__main__":
    main()
