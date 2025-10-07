#!/usr/bin/env python3
"""
Linguistic Feature Computation for Danish Eye-Tracking Data
Implements word frequency (Zipf scale) and surprisal computation
Updated to use Leipzig Corpora frequency data
"""

import logging
import unicodedata
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DanishLinguisticFeatures:
    """
    Compute linguistic features for Danish text:
    - Word frequency (from Leipzig Corpora, Zipf transformed)
    - Surprisal (using transformer models)
    - Word length (character count)
    """

    def __init__(
        self,
        lemma_file: str = "danish_frequencies/danish_leipzig_for_analysis.txt",
        use_proportions: bool = True,
        strip_punctuation: bool = True,
    ):
        """
        Initialize with Danish frequency lists

        Args:
            lemma_file: Path to the frequency file
                       Default: Leipzig Corpora processed file (word\tproportion format)
            use_proportions: If True, expects file to contain proportions
                           If False, expects raw frequency counts
            strip_punctuation: If True, strip leading/trailing punctuation before lookup
        """
        self.lemma_file = Path(lemma_file)
        self.use_proportions = use_proportions
        self.strip_punctuation = strip_punctuation
        self.freq_dict = None
        self.total_corpus_tokens = None

        # Load frequency data
        self._load_frequency_dict()

        # Surprisal model (lazy loading)
        self._surprisal_model = None
        self._surprisal_tokenizer = None

    def _load_frequency_dict(self):
        """Load Danish word frequencies from Leipzig Corpora"""

        if not self.lemma_file.exists():
            raise FileNotFoundError(
                f"Frequency file not found: {self.lemma_file}\n"
                f"Run download_leipzig_danish.py to generate this file."
            )

        logger.info(f"Loading Danish frequency list from {self.lemma_file}")

        # Load the frequency file (format: word\tproportion or word\tfrequency)
        freq_df = pd.read_csv(
            self.lemma_file,
            sep="\t",
            header=None,
            names=["word", "value"],
            encoding="utf-8",
        )

        # Check for duplicates after lowercasing
        freq_df["word_lower"] = freq_df["word"].str.lower()
        duplicate_count = len(freq_df) - freq_df["word_lower"].nunique()

        if duplicate_count > 0:
            logger.warning(
                f"Found {duplicate_count:,} duplicate words after lowercasing. "
                f"This suggests the frequency file was not properly standardized. "
                f"Re-run download_leipzig_danish.py to fix this issue."
            )

        if self.use_proportions:
            # File contains proportions (value between 0 and 1)
            # Use word_lower for dictionary keys
            self.freq_dict = dict(zip(freq_df["word_lower"], freq_df["value"]))
            # For proportions, we don't need total_corpus_tokens
            self.total_corpus_tokens = None
            logger.info(
                f"Loaded {len(self.freq_dict):,} word proportions from Leipzig Corpora"
            )
        else:
            # File contains raw frequencies
            self.freq_dict = dict(zip(freq_df["word_lower"], freq_df["value"]))
            self.total_corpus_tokens = freq_df["value"].sum()
            logger.info(
                f"Loaded {len(self.freq_dict):,} word frequencies "
                f"({self.total_corpus_tokens:,} total corpus tokens)"
            )

    @staticmethod
    def _clean_word_for_lookup(word: str, strip_punct: bool = True) -> str:
        """
        Clean word for frequency lookup

        Args:
            word: Original word
            strip_punct: Whether to strip punctuation

        Returns:
            Cleaned word (lowercase, optionally stripped of punctuation)
            Returns empty string for words with encoding corruption
        """
        if pd.isna(word):
            return ""

        word = str(word)

        # Normalize Unicode (handle combining characters like é = e + ́)
        word = unicodedata.normalize("NFKC", word)

        # Convert to lowercase
        word = word.lower()

        if strip_punct:
            # Comprehensive punctuation stripping including Unicode variants
            # Includes: straight quotes, curly quotes, dashes, brackets, etc.
            punctuation = (
                '\'"""'
                "‚„«»‹›[](){}.,;:!?¿¡…–—−-"
                "\u2018\u2019\u201c\u201d"  # Curly single and double quotes
                "\u2013\u2014\u2015"  # En dash, em dash, horizontal bar
                "\u00ab\u00bb"  # Guillemets
                "\u2026"  # Ellipsis
            )
            word = word.strip(punctuation)

            # CRITICAL: Check for internal '?' after stripping trailing punctuation
            # If '?' remains, it's encoding corruption (e.g., blæredygtige → bl?redygtige)
            if "?" in word:
                return ""  # Skip corrupted words

            # Check if what remains is only punctuation or whitespace
            if not word or not any(c.isalnum() for c in word):
                return ""

        return word

    def _get_word_frequency(self, word: str) -> float:
        """
        Get frequency for a single word with robust handling

        Args:
            word: Word to lookup

        Returns:
            Frequency value (proportion or count)
        """
        if pd.isna(word):
            return self.min_freq

        # Clean word for lookup
        word_clean = self._clean_word_for_lookup(word, self.strip_punctuation)

        if not word_clean:  # Empty after cleaning (was punctuation-only)
            return self.min_freq

        return self.freq_dict.get(word_clean, self.min_freq)

    @property
    def min_freq(self) -> float:
        """Minimum frequency for unknown words"""
        if self.use_proportions:
            return 1e-9  # Very small proportion
        else:
            return 1  # Count of 1

    def compute_word_frequency(
        self, words: pd.Series, log_missing: bool = True
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute word frequency (both raw and Zipf) with robust handling

        Args:
            words: Series of words to lookup
            log_missing: Whether to log missing words statistics

        Returns:
            Tuple of (raw_frequencies, zipf_frequencies)
        """

        # Use cached lookup for better performance
        frequencies = words.apply(self._get_word_frequency)

        # Identify missing words (assigned min_freq) - only log once if requested
        if log_missing:
            missing_mask = frequencies == self.min_freq
            n_missing = missing_mask.sum()

            if n_missing > 0:
                pct_missing = (n_missing / len(words)) * 100

                # Get unique missing words
                missing_words_raw = words[missing_mask].unique()

                # Clean them the same way we do for lookup
                missing_words_cleaned = []
                corrupted_count = 0

                for w in missing_words_raw:
                    cleaned = self._clean_word_for_lookup(w, self.strip_punctuation)
                    if cleaned == "":
                        # This was either punctuation-only or corrupted (has internal ?)
                        if "?" in str(w):
                            corrupted_count += 1
                        # Don't add to cleaned list
                    else:
                        # Only add if it doesn't contain ? (true corruption indicator)
                        if "?" not in str(w):
                            missing_words_cleaned.append(cleaned)
                        else:
                            corrupted_count += 1

                # Remove duplicates after cleaning
                missing_words_cleaned = list(set(missing_words_cleaned))
                n_unique_missing = len(missing_words_cleaned)

                # Count punctuation-only (empty after cleaning, no ?)
                n_punctuation_only = (
                    len(missing_words_raw) - n_unique_missing - corrupted_count
                )

                # Log the breakdown
                if n_punctuation_only > 0:
                    logger.info(
                        f"{n_punctuation_only} missing word instances were punctuation-only"
                    )

                if corrupted_count > 0:
                    logger.info(
                        f"{corrupted_count} missing word instances had encoding corruption (internal '?') and were excluded"
                    )

                if n_unique_missing > 0:
                    n_valid_missing = n_missing - corrupted_count - n_punctuation_only
                    logger.warning(
                        f"{n_valid_missing:,} word instances "
                        f"({(n_valid_missing/len(words))*100:.1f}%) "
                        f"not found in frequency list ({n_unique_missing:,} unique valid words)"
                    )

                    # Log sample
                    if n_unique_missing <= 50:
                        logger.info(
                            f"Missing valid words: {sorted(missing_words_cleaned)}"
                        )
                    else:
                        sample = sorted(missing_words_cleaned[:50])
                        logger.info(
                            f"Sample of {n_unique_missing} missing valid words: {sample}... "
                            f"({n_unique_missing - 50} more)"
                        )
                elif corrupted_count > 0 or n_punctuation_only > 0:
                    logger.info(
                        f"All {n_missing:,} missing word instances were either punctuation-only "
                        f"or encoding-corrupted (as expected)"
                    )

        # Compute Zipf frequencies
        # Ensure no zeros before log
        frequencies_clipped = frequencies.clip(lower=1e-10)

        if self.use_proportions:
            zipf_freq = np.log10(frequencies_clipped * 1e9) + 3
        else:
            zipf_freq = (
                np.log10((frequencies_clipped / self.total_corpus_tokens) * 1e9) + 3
            )

        return frequencies, zipf_freq

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
        exclude_missing_frequencies: bool = True,
    ) -> pd.DataFrame:
        """
        Add all linguistic features to dataframe

        Args:
            data: Input dataframe with word-level data
            compute_surprisal: Whether to compute surprisal (slow!)
            text_col: Column containing word text
            exclude_missing_frequencies: If True, remove words not found in frequency list

        Returns:
            DataFrame with added columns:
                - word_length: character count (if not present)
                - word_frequency_raw: raw proportion or frequency
                - word_frequency_zipf: Zipf-transformed frequency (1-7 scale)
                - surprisal: -log2 probability (if compute_surprisal=True)
        """

        data = data.copy()

        # 1. Word length (if not already present)
        if "word_length" not in data.columns:
            logger.info("Computing word length...")
            data["word_length"] = data[text_col].str.len()

        # 2. Word frequency - compute both raw and zipf together
        logger.info("Computing word frequency...")
        if self.strip_punctuation:
            logger.info("  Stripping punctuation from words before frequency lookup")

        # Compute both frequencies at once to avoid duplicate logging
        data["word_frequency_raw"], data["word_frequency_zipf"] = (
            self.compute_word_frequency(data[text_col], log_missing=True)
        )

        # Exclude words with missing frequencies if requested
        if exclude_missing_frequencies:
            missing_mask = data["word_frequency_raw"] == self.min_freq
            n_excluded = missing_mask.sum()

            if n_excluded > 0:
                logger.info(
                    f"Excluding {n_excluded:,} word instances ({(n_excluded/len(data))*100:.1f}%) "
                    f"with missing frequencies from analysis"
                )
                data = data[~missing_mask].copy()
                logger.info(f"Remaining dataset: {len(data):,} words")

        # Log statistics about frequency coverage
        logger.info(f"Final dataset: {len(data):,} words with valid frequencies")

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
            f"std={data['word_frequency_zipf'].std():.2f}, "
            f"range=[{data['word_frequency_zipf'].min():.2f}, {data['word_frequency_zipf'].max():.2f}]"
        )
        if compute_surprisal and "surprisal" in data.columns:
            logger.info(
                f"  Surprisal: mean={data['surprisal'].mean():.2f}, "
                f"std={data['surprisal'].std():.2f}"
            )

        return data
