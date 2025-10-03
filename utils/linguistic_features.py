#!/usr/bin/env python3
"""
Enhanced Linguistic Feature Computation for Danish Eye-Tracking Data
Key improvements:
- Robust surprisal computation with proper token alignment
- Better error handling and validation
- Performance optimizations with caching
- Danish-specific text normalization
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DanishLinguisticFeatures:
    """
    Compute linguistic features for Danish text with enhanced robustness
    """

    def __init__(
        self,
        lemma_file: str = "danish_frequencies/danish_leipzig_for_analysis.txt",
        use_proportions: bool = True,
        cache_size: int = 10000,
    ):
        self.lemma_file = Path(lemma_file)
        self.use_proportions = use_proportions
        self.cache_size = cache_size
        self.freq_dict = None
        self.total_corpus_tokens = None
        self.min_freq = None

        self._load_frequency_dict()

        # Lazy-loaded surprisal components
        self._surprisal_model = None
        self._surprisal_tokenizer = None
        self._device = None

    def _load_frequency_dict(self):
        """Load and validate Danish word frequencies"""

        if not self.lemma_file.exists():
            raise FileNotFoundError(
                f"Frequency file not found: {self.lemma_file}\n"
                f"Run download_leipzig_danish.py to generate this file."
            )

        logger.info(f"Loading Danish frequency list from {self.lemma_file}")

        try:
            freq_df = pd.read_csv(
                self.lemma_file,
                sep="\t",
                header=None,
                names=["word", "value"],
                encoding="utf-8",
                on_bad_lines="warn",
            )
        except Exception as e:
            logger.error(f"Failed to read frequency file: {e}")
            raise

        # Validation
        if freq_df["value"].isna().any():
            n_missing = freq_df["value"].isna().sum()
            logger.warning(f"Found {n_missing} missing values in frequency file")
            freq_df = freq_df.dropna()

        if (freq_df["value"] < 0).any():
            raise ValueError("Negative frequencies found in file")

        if self.use_proportions:
            if (freq_df["value"] > 1).any():
                logger.warning(
                    f"Found {(freq_df['value'] > 1).sum()} values > 1 "
                    f"in proportion mode. Check file format."
                )

            self.freq_dict = dict(zip(freq_df["word"].str.lower(), freq_df["value"]))

            # Set minimum frequency as 1 occurrence per billion
            self.min_freq = 1e-9
            self.total_corpus_tokens = None

            logger.info(
                f"Loaded {len(self.freq_dict):,} word proportions "
                f"(min: {freq_df['value'].min():.2e}, max: {freq_df['value'].max():.2e})"
            )
        else:
            self.freq_dict = dict(zip(freq_df["word"].str.lower(), freq_df["value"]))
            self.total_corpus_tokens = freq_df["value"].sum()

            # Minimum frequency: 1 occurrence
            self.min_freq = 1 / self.total_corpus_tokens

            logger.info(
                f"Loaded {len(self.freq_dict):,} word frequencies "
                f"({self.total_corpus_tokens:,} total corpus tokens)"
            )

    @lru_cache(maxsize=10000)
    def _get_word_frequency(self, word: str) -> float:
        """Cached frequency lookup"""
        return self.freq_dict.get(word.lower(), self.min_freq)

    def compute_word_frequency(
        self, words: pd.Series, return_zipf: bool = True
    ) -> pd.Series:
        """
        Compute word frequency with robust handling of missing values
        """

        # Use cached lookup for better performance
        frequencies = words.apply(self._get_word_frequency)

        # Count missing (assigned min_freq)
        n_missing = (frequencies == self.min_freq).sum()
        if n_missing > 0:
            pct_missing = (n_missing / len(words)) * 100
            logger.warning(
                f"{n_missing:,} words ({pct_missing:.1f}%) not in frequency list"
            )

        if return_zipf:
            # Ensure no zeros before log
            frequencies = frequencies.clip(lower=1e-10)

            if self.use_proportions:
                zipf_freq = np.log10(frequencies * 1e9) + 3
            else:
                zipf_freq = np.log10((frequencies / self.total_corpus_tokens) * 1e9) + 3

            return zipf_freq
        else:
            return frequencies

    def _load_surprisal_model(self, model_name: str = "Maltehb/danish-bert-botxo"):
        """Load transformer model with device detection"""

        if self._surprisal_model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            logger.info(f"Loading transformer model: {model_name}")

            self._surprisal_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._surprisal_model = AutoModelForMaskedLM.from_pretrained(model_name)
            self._surprisal_model.eval()

            # Device detection
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                self._surprisal_model = self._surprisal_model.to(self._device)
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self._device = torch.device("cpu")
                logger.info("Using CPU (consider GPU for faster processing)")

        except ImportError:
            raise ImportError(
                "Required libraries not found.\n"
                "Install with: pip install transformers torch"
            )

    def _compute_sentence_surprisal(self, words: List[str]) -> List[float]:
        """
        Compute surprisal for words in a sentence with proper alignment
        """
        import torch

        sentence = " ".join(words)

        # Tokenize with offset mapping for alignment
        encoding = self._surprisal_tokenizer(
            sentence,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        input_ids = encoding["input_ids"].to(self._device)
        offset_mapping = encoding["offset_mapping"][0]

        # Get model predictions
        with torch.no_grad():
            outputs = self._surprisal_model(input_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]

        # Calculate character positions for each word
        word_positions = []
        char_pos = 0
        for word in words:
            word_positions.append((char_pos, char_pos + len(word)))
            char_pos += len(word) + 1  # +1 for space

        # Align words to tokens and compute surprisal
        surprisals = []

        for word_start, word_end in word_positions:
            # Find tokens that overlap with this word
            token_indices = []
            for i, (tok_start, tok_end) in enumerate(offset_mapping):
                if tok_start >= word_start and tok_start < word_end:
                    token_indices.append(i)

            if not token_indices:
                surprisals.append(np.nan)
                continue

            # Use first token of word for surprisal
            token_idx = token_indices[0]

            if token_idx == 0 or token_idx >= len(input_ids[0]) - 1:
                # Skip [CLS] and [SEP]
                surprisals.append(np.nan)
                continue

            # Get probability of token given left context
            token_id = input_ids[0, token_idx]
            prev_logits = logits[token_idx - 1]
            probs = torch.softmax(prev_logits, dim=-1)
            prob = probs[token_id].item()

            # Surprisal = -log2(probability)
            if prob > 0:
                surprisal = -np.log2(prob)
            else:
                surprisal = 20.0  # Cap at 20 bits

            surprisals.append(surprisal)

        return surprisals

    def compute_surprisal(
        self,
        data: pd.DataFrame,
        text_col: str = "word_text",
        sentence_col: str = "sentence_id",
        model_name: str = "Maltehb/danish-bert-botxo",
        max_sentences: Optional[int] = None,
    ) -> pd.Series:
        """
        Compute surprisal with proper token alignment

        Args:
            max_sentences: Limit processing to first N sentences (for testing)
        """

        self._load_surprisal_model(model_name)

        logger.info("Computing surprisal values...")

        surprisal_values = np.full(len(data), np.nan)
        grouped = data.groupby(sentence_col)

        if max_sentences:
            groups_to_process = list(grouped)[:max_sentences]
            logger.info(f"Processing {max_sentences} sentences (limited for testing)")
        else:
            groups_to_process = list(grouped)

        for sent_id, sent_data in tqdm(groups_to_process, desc="Sentences"):
            words = sent_data[text_col].tolist()
            indices = sent_data.index.tolist()

            try:
                surprisals = self._compute_sentence_surprisal(words)

                for idx, surp in zip(indices, surprisals):
                    surprisal_values[idx] = surp

            except Exception as e:
                logger.warning(
                    f"Failed to compute surprisal for sentence {sent_id}: {e}"
                )
                continue

        valid_count = (~np.isnan(surprisal_values)).sum()
        logger.info(f"Computed surprisal for {valid_count:,} / {len(data):,} words")

        return pd.Series(surprisal_values, index=data.index)

    def add_all_features(
        self,
        data: pd.DataFrame,
        compute_surprisal: bool = False,
        text_col: str = "word_text",
        max_sentences_surprisal: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Add all linguistic features with comprehensive logging
        """

        data = data.copy()

        # 1. Word length
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

        # 3. Surprisal (optional)
        if compute_surprisal:
            if "sentence_id" not in data.columns:
                logger.warning("sentence_id required for surprisal. Skipping.")
            else:
                data["surprisal"] = self.compute_surprisal(
                    data, text_col=text_col, max_sentences=max_sentences_surprisal
                )

        # Summary statistics
        self._log_feature_summary(data, compute_surprisal)

        return data

    def _log_feature_summary(self, data: pd.DataFrame, has_surprisal: bool):
        """Log comprehensive feature statistics"""

        logger.info("\n" + "=" * 60)
        logger.info("FEATURE SUMMARY")
        logger.info("=" * 60)

        for feature in ["word_length", "word_frequency_zipf", "surprisal"]:
            if feature not in data.columns:
                continue

            values = data[feature].dropna()
            if len(values) == 0:
                continue

            logger.info(f"\n{feature}:")
            logger.info(f"  Count:  {len(values):,}")
            logger.info(f"  Mean:   {values.mean():.2f}")
            logger.info(f"  Std:    {values.std():.2f}")
            logger.info(f"  Median: {values.median():.2f}")
            logger.info(f"  Range:  [{values.min():.2f}, {values.max():.2f}]")
            logger.info(
                f"  Q1-Q3:  [{values.quantile(0.25):.2f}, {values.quantile(0.75):.2f}]"
            )

        logger.info("=" * 60 + "\n")


def validate_linguistic_features(data: pd.DataFrame) -> Dict:
    """Enhanced validation with detailed diagnostics"""

    results = {"valid": True, "warnings": [], "statistics": {}}

    expected_ranges = {
        "word_length": (1, 30),
        "word_frequency_zipf": (0, 8),
        "surprisal": (0, 25),
    }

    for feature, (min_expected, max_expected) in expected_ranges.items():
        if feature not in data.columns:
            continue

        values = data[feature].dropna()
        if len(values) == 0:
            results["warnings"].append(f"{feature}: No valid values found")
            results["valid"] = False
            continue

        actual_min = values.min()
        actual_max = values.max()

        # Range check
        if actual_min < min_expected or actual_max > max_expected:
            results["warnings"].append(
                f"{feature}: range [{actual_min:.2f}, {actual_max:.2f}] "
                f"outside expected [{min_expected}, {max_expected}]"
            )
            results["valid"] = False

        # Missing data check
        missing_pct = data[feature].isna().sum() / len(data) * 100
        if missing_pct > 10:
            results["warnings"].append(
                f"{feature}: {missing_pct:.1f}% missing (threshold: 10%)"
            )

        # Store statistics
        results["statistics"][feature] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(actual_min),
            "max": float(actual_max),
            "missing_pct": float(missing_pct),
        }

    return results
