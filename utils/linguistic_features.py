#!/usr/bin/env python3
"""
Linguistic Feature Computation for Danish Eye-Tracking Data
Implements word frequency (Zipf scale) and surprisal computation
Based on CopCo corpus standards and current best practices
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DanishLinguisticFeatures:
    """
    Compute linguistic features for Danish text:
    - Word frequency (from korpus.dsl.dk lemma lists, Zipf transformed)
    - Surprisal (using transformer models)
    - Word length (character count)
    """

    def __init__(self, lemma_file: str = "utils/lemma/lemma-10k-2017-in.txt"):
        """
        Initialize with Danish frequency lists

        Args:
            lemma_file: Path to the lemma frequency file (CopCo standard)
        """
        self.lemma_file = Path(lemma_file)
        self.freq_dict = None
        self.total_corpus_tokens = None

        # Load frequency data
        self._load_frequency_dict()

        # Surprisal model (lazy loading)
        self._surprisal_model = None
        self._surprisal_tokenizer = None

    def _load_frequency_dict(self):
        """Load Danish lemma frequencies from CopCo processing pipeline"""

        if not self.lemma_file.exists():
            raise FileNotFoundError(
                f"Lemma frequency file not found: {self.lemma_file}\n"
                f"Download from: https://github.com/norahollenstein/copco-processing"
            )

        logger.info(f"Loading Danish frequency list from {self.lemma_file}")

        # Try different separators (tab or whitespace)
        try:
            freq_df = pd.read_csv(
                self.lemma_file,
                sep="\t",
                names=["lemma", "frequency"],
                encoding="utf-8",
            )
        except:
            freq_df = pd.read_csv(
                self.lemma_file,
                sep=r"\s+",
                names=["lemma", "frequency"],
                encoding="utf-8",
            )

        # Create lookup dictionary (lowercase for matching)
        self.freq_dict = dict(zip(freq_df["lemma"].str.lower(), freq_df["frequency"]))

        # Calculate total corpus size for Zipf transformation
        self.total_corpus_tokens = freq_df["frequency"].sum()

        logger.info(
            f"Loaded {len(self.freq_dict):,} lemma frequencies "
            f"({self.total_corpus_tokens:,} total corpus tokens)"
        )

    def compute_word_frequency(
        self, words: pd.Series, return_zipf: bool = True
    ) -> pd.Series:
        """
        Compute word frequency from Danish lemma lists

        Args:
            words: Series of word strings
            return_zipf: If True, return Zipf-transformed frequencies (recommended)
                        If False, return raw frequencies

        Returns:
            Series of frequency values (Zipf scale 1-7 or raw counts)
        """

        # Match words to frequency dictionary (case-insensitive)
        frequencies = words.str.lower().map(self.freq_dict)

        # Handle missing frequencies (words not in top 10K)
        n_missing = frequencies.isna().sum()
        if n_missing > 0:
            pct_missing = (n_missing / len(words)) * 100
            logger.warning(
                f"{n_missing:,} words ({pct_missing:.1f}%) not found in frequency list. "
                f"Assigning minimum frequency."
            )
            # Assign frequency of 1 to unknown words (will be low Zipf)
            frequencies = frequencies.fillna(1)

        if return_zipf:
            # Transform to Zipf scale: log10(freq per billion) + 3
            # This gives interpretable scores from ~1 (rare) to ~7 (very frequent)
            zipf_freq = np.log10((frequencies / self.total_corpus_tokens) * 1e9) + 3
            return zipf_freq
        else:
            return frequencies

    def _load_surprisal_model(self, model_name: str = "Maltehb/danish-bert-botxo"):
        """
        Lazy load transformer model for surprisal computation

        Args:
            model_name: HuggingFace model identifier
                       Default: Danish BERT (best for Danish)
                       Alternative: "bert-base-multilingual-cased" (mBERT)
        """
        if self._surprisal_model is not None:
            return  # Already loaded

        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            logger.info(f"Loading transformer model: {model_name}")

            self._surprisal_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._surprisal_model = AutoModelForMaskedLM.from_pretrained(model_name)
            self._surprisal_model.eval()  # Set to evaluation mode

            # Move to GPU if available
            if torch.cuda.is_available():
                self._surprisal_model = self._surprisal_model.cuda()
                logger.info("Using GPU for surprisal computation")

            logger.info("Transformer model loaded successfully")

        except ImportError:
            raise ImportError(
                "Transformers library required for surprisal computation.\n"
                "Install with: pip install transformers torch"
            )

    def compute_surprisal(
        self,
        data: pd.DataFrame,
        text_col: str = "word_text",
        sentence_col: str = "sentence_id",
        model_name: str = "Maltehb/danish-bert-botxo",
        batch_size: int = 32,
    ) -> pd.Series:
        """
        Compute surprisal for each word using masked language model

        Note: This is computationally expensive. For large datasets,
        consider computing on a subset or using GPU acceleration.

        Args:
            data: DataFrame with words and sentence structure
            text_col: Column containing word text
            sentence_col: Column indicating sentence boundaries
            model_name: HuggingFace model identifier
            batch_size: Batch size for processing

        Returns:
            Series of surprisal values (-log probability)
        """

        self._load_surprisal_model(model_name)

        logger.info("Computing surprisal values (this may take a while)...")

        import torch
        from tqdm import tqdm

        surprisal_values = np.full(len(data), np.nan)

        # Group by sentence
        grouped = data.groupby(sentence_col)

        for sent_id, sent_data in tqdm(grouped, desc="Processing sentences"):
            words = sent_data[text_col].tolist()
            indices = sent_data.index.tolist()

            # Reconstruct sentence
            sentence = " ".join(words)

            # Tokenize
            inputs = self._surprisal_tokenizer(
                sentence, return_tensors="pt", add_special_tokens=True
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Get model predictions
            with torch.no_grad():
                outputs = self._surprisal_model(**inputs)
                logits = outputs.logits

            # Compute surprisal for each word
            # This is simplified - word-to-token alignment needed for multi-token words
            token_ids = inputs["input_ids"][0]

            # For each word position, calculate surprisal
            word_positions = self._align_words_to_tokens(
                words, self._surprisal_tokenizer.convert_ids_to_tokens(token_ids)
            )

            for i, (word, word_idx) in enumerate(zip(words, indices)):
                if word_positions[i] is not None:
                    token_pos = word_positions[i]
                    if token_pos > 0 and token_pos < len(token_ids) - 1:
                        # Get probability of this token given context
                        token_id = token_ids[token_pos]
                        probs = torch.softmax(logits[0, token_pos - 1], dim=-1)
                        prob = probs[token_id].item()

                        # Surprisal = -log2(probability)
                        surprisal = (
                            -np.log2(prob) if prob > 0 else 20.0
                        )  # Cap at 20 bits
                        surprisal_values[word_idx] = surprisal

        logger.info(
            f"Computed surprisal for {(~np.isnan(surprisal_values)).sum():,} words"
        )

        return pd.Series(surprisal_values, index=data.index)

    def _align_words_to_tokens(self, words, tokens):
        """
        Align words to tokenizer tokens (simplified version)

        Note: This is a basic implementation. For production use,
        consider more sophisticated alignment (e.g., using token offsets)
        """
        positions = []
        token_idx = 1  # Skip [CLS]

        for word in words:
            # Simple matching - may need refinement for subword tokenization
            positions.append(token_idx)
            # Approximate token count for this word
            word_tokens = self._surprisal_tokenizer.tokenize(word)
            token_idx += len(word_tokens)

        return positions

    def add_all_features(
        self,
        data: pd.DataFrame,
        compute_surprisal: bool = False,
        text_col: str = "word_text",
    ) -> pd.DataFrame:
        """
        Add all linguistic features to dataframe

        Args:
            data: Input dataframe with word-level data
            compute_surprisal: Whether to compute surprisal (slow!)
            text_col: Column containing word text

        Returns:
            DataFrame with added columns:
                - word_length: character count (if not present)
                - word_frequency_raw: raw frequency count
                - word_frequency_zipf: Zipf-transformed frequency (1-7 scale)
                - surprisal: -log2 probability (if compute_surprisal=True)
        """

        data = data.copy()

        # 1. Word length (if not already present)
        if "word_length" not in data.columns:
            logger.info("Computing word length...")
            data["word_length"] = data[text_col].str.len()

        # 2. Word frequency
        logger.info("Computing word frequency...")
        data["word_frequency_raw"] = self.compute_word_frequency(
            data[text_col], return_zipf=False
        )
        data["word_frequency_zipf"] = self.compute_word_frequency(
            data[text_col], return_zipf=True
        )

        # 3. Surprisal (optional, computationally expensive)
        if compute_surprisal:
            if "sentence_id" not in data.columns:
                logger.warning("sentence_id column required for surprisal. Skipping.")
            else:
                data["surprisal"] = self.compute_surprisal(data, text_col=text_col)

        logger.info("Linguistic features added successfully")

        # Log summary statistics
        logger.info("\nFeature Summary:")
        logger.info(
            f"  Word length: mean={data['word_length'].mean():.2f}, "
            f"std={data['word_length'].std():.2f}"
        )
        logger.info(
            f"  Zipf frequency: mean={data['word_frequency_zipf'].mean():.2f}, "
            f"std={data['word_frequency_zipf'].std():.2f}"
        )
        if compute_surprisal and "surprisal" in data.columns:
            logger.info(
                f"  Surprisal: mean={data['surprisal'].mean():.2f}, "
                f"std={data['surprisal'].std():.2f}"
            )

        return data


def validate_linguistic_features(data: pd.DataFrame) -> Dict[str, any]:
    """
    Validate computed linguistic features against expected ranges

    Returns:
        Dictionary with validation results and warnings
    """

    results = {"valid": True, "warnings": []}

    # Expected ranges based on research
    expected_ranges = {
        "word_length": (1, 29),  # From CopCo stats
        "word_frequency_zipf": (0, 8),  # Zipf scale typically 1-7
        "surprisal": (0, 25),  # bits, typically 2-15
    }

    for feature, (min_val, max_val) in expected_ranges.items():
        if feature in data.columns:
            actual_min = data[feature].min()
            actual_max = data[feature].max()

            if actual_min < min_val or actual_max > max_val:
                results["warnings"].append(
                    f"{feature}: range [{actual_min:.2f}, {actual_max:.2f}] "
                    f"outside expected [{min_val}, {max_val}]"
                )
                results["valid"] = False

    # Check for excessive missing data
    for col in ["word_frequency_zipf", "surprisal"]:
        if col in data.columns:
            missing_pct = data[col].isna().sum() / len(data) * 100
            if missing_pct > 10:
                results["warnings"].append(
                    f"{col}: {missing_pct:.1f}% missing values (high)"
                )

    return results
