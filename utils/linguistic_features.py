#!/usr/bin/env python3
"""
Linguistic Feature Computation for Danish Eye-Tracking Data
Implements word frequency (Zipf scale) and surprisal computation
Updated to use Leipzig Corpora frequency data
"""

import logging
from pathlib import Path
from typing import Dict

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
    ):
        """
        Initialize with Danish frequency lists

        Args:
            lemma_file: Path to the frequency file
                       Default: Leipzig Corpora processed file (word\tproportion format)
            use_proportions: If True, expects file to contain proportions
                           If False, expects raw frequency counts
        """
        self.lemma_file = Path(lemma_file)
        self.use_proportions = use_proportions
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

        if self.use_proportions:
            # File contains proportions (value between 0 and 1)
            self.freq_dict = dict(zip(freq_df["word"].str.lower(), freq_df["value"]))
            # For proportions, we don't need total_corpus_tokens
            self.total_corpus_tokens = None
            logger.info(
                f"Loaded {len(self.freq_dict):,} word proportions from Leipzig Corpora"
            )
        else:
            # File contains raw frequencies
            self.freq_dict = dict(zip(freq_df["word"].str.lower(), freq_df["value"]))
            self.total_corpus_tokens = freq_df["value"].sum()
            logger.info(
                f"Loaded {len(self.freq_dict):,} word frequencies "
                f"({self.total_corpus_tokens:,} total corpus tokens)"
            )

    def compute_word_frequency(
        self, words: pd.Series, return_zipf: bool = True
    ) -> pd.Series:
        """
        Compute word frequency from Danish frequency lists

        Args:
            words: Series of word strings
            return_zipf: If True, return Zipf-transformed frequencies (recommended)
                        If False, return raw frequencies or proportions

        Returns:
            Series of frequency values (Zipf scale 1-7 or raw values)
        """

        # Match words to frequency dictionary (case-insensitive)
        frequencies = words.str.lower().map(self.freq_dict)

        # Handle missing frequencies
        n_missing = frequencies.isna().sum()
        if n_missing > 0:
            pct_missing = (n_missing / len(words)) * 100
            logger.warning(
                f"{n_missing:,} words ({pct_missing:.1f}%) not found in frequency list. "
                f"Assigning minimum frequency."
            )

            if self.use_proportions:
                # For proportions, assign a very small value (1 occurrence per billion words)
                min_proportion = 1e-9
                frequencies = frequencies.fillna(min_proportion)
            else:
                # For raw frequencies, assign frequency of 1
                frequencies = frequencies.fillna(1)

        if return_zipf:
            # Transform to Zipf scale: log10(freq per billion) + 3
            # This gives interpretable scores from ~1 (rare) to ~7 (very frequent)

            if self.use_proportions:
                # proportions are already normalized (freq / total_tokens)
                # So we just need: log10(proportion * 1e9) + 3
                zipf_freq = np.log10(frequencies * 1e9) + 3
            else:
                # For raw frequencies: log10((freq / total) * 1e9) + 3
                zipf_freq = np.log10((frequencies / self.total_corpus_tokens) * 1e9) + 3

            # Handle any -inf values (from log of 0)
            zipf_freq = zipf_freq.replace([np.inf, -np.inf], np.nan)

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
                - word_frequency_raw: raw proportion or frequency
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

        # Log statistics about frequency coverage
        valid_freq = data["word_frequency_zipf"].notna()
        logger.info(
            f"Frequency coverage: {valid_freq.sum():,} / {len(data):,} words ({valid_freq.mean()*100:.1f}%)"
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
            f"std={data['word_frequency_zipf'].std():.2f}, "
            f"range=[{data['word_frequency_zipf'].min():.2f}, {data['word_frequency_zipf'].max():.2f}]"
        )
        if compute_surprisal and "surprisal" in data.columns:
            logger.info(
                f"  Surprisal: mean={data['surprisal'].mean():.2f}, "
                f"std={data['surprisal'].std():.2f}"
            )

        return data
