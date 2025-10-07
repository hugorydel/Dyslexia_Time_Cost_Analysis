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
    - Surprisal (using transformer models in pseudo-autoregressive mode)
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
        max_length: int = 512,
        cache_file: str = None,
        checkpoint_every: int = 100,  # Checkpoint every N sentences
        batch_size: int = 8,
    ) -> pd.Series:
        """
        Compute surprisal using BERT in pseudo-autoregressive mode with batched processing

        Args:
            batch_size: Number of word positions to process simultaneously (default: 20)
                       Increase for speed (may hit memory limits around 32-50)
            checkpoint_every: Save checkpoint every N sentences (default: 10 for testing)
        """

        self._load_surprisal_model(model_name)

        # Setup cache paths
        surprisal_dir = Path("word_surprisal")
        surprisal_dir.mkdir(exist_ok=True)
        if cache_file is None:
            cache_file = surprisal_dir / f"surprisal_autoregressive_{sentence_col}.pkl"
        else:
            cache_file = surprisal_dir / cache_file

        cache_path = Path(cache_file)
        checkpoint_path = Path(str(cache_path).replace(".pkl", "_checkpoint.pkl"))

        # Try to load cache
        if cache_path.exists():
            logger.info(f"Loading cached surprisal values from {cache_path}")
            try:
                import pickle

                with open(cache_path, "rb") as f:
                    cached_surprisal = pickle.load(f)
                logger.info(f"Loaded {len(cached_surprisal):,} cached surprisal values")

                if len(cached_surprisal) == len(data):
                    return pd.Series(cached_surprisal, index=data.index)
                else:
                    logger.warning(f"Cached data length mismatch. Recomputing...")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        # Load checkpoint
        processed_sentences = {}
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            try:
                import pickle

                with open(checkpoint_path, "rb") as f:
                    checkpoint_data = pickle.load(f)
                    processed_sentences = checkpoint_data["processed_sentences"]
                logger.info(
                    f"Resuming from checkpoint: {len(processed_sentences):,} sentences processed"
                )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

        logger.info(
            f"Computing surprisal with batched processing (batch_size={batch_size})..."
        )
        logger.info("Using pseudo-autoregressive BERT (left-context only)")

        import pickle

        import torch
        from tqdm import tqdm

        surprisal_values = np.full(len(data), np.nan)

        # Restore from checkpoint
        if processed_sentences:
            for sent_id, sent_surprisals in processed_sentences.items():
                for idx, val in sent_surprisals.items():
                    surprisal_values[idx] = val

        grouped = data.groupby(sentence_col)
        sentence_list = list(grouped)

        n_failed = 0
        n_oom_errors = 0  # Track out-of-memory errors
        sentences_processed = len(processed_sentences)

        for sent_idx, (sent_id, sent_data) in enumerate(
            tqdm(sentence_list, desc="Processing sentences")
        ):
            if sent_id in processed_sentences:
                continue

            words = sent_data[text_col].tolist()
            indices = sent_data.index.tolist()
            sentence_surprisals = {}

            # Process words in batches for speed
            for batch_start in range(0, len(words), batch_size):
                batch_end = min(batch_start + batch_size, len(words))
                batch_positions = range(batch_start, batch_end)

                try:
                    # Create batch of masked sentences
                    batch_masked_sentences = []
                    for word_pos in batch_positions:
                        masked_words = (
                            words[:word_pos]
                            + ["[MASK]"]
                            + ["[MASK]"] * (len(words) - word_pos - 1)
                        )
                        batch_masked_sentences.append(" ".join(masked_words))

                    # Tokenize batch
                    inputs = self._surprisal_tokenizer(
                        batch_masked_sentences,
                        return_tensors="pt",
                        add_special_tokens=True,
                        padding=True,  # Pad to same length within batch
                        truncation=True,
                        max_length=max_length,
                    )

                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}

                    # Get predictions for entire batch at once
                    with torch.no_grad():
                        outputs = self._surprisal_model(**inputs)
                        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

                    # Extract surprisal for each word in batch
                    mask_token_id = self._surprisal_tokenizer.mask_token_id

                    for batch_idx, word_pos in enumerate(batch_positions):
                        word = words[word_pos]
                        word_idx = indices[word_pos]

                        # Find first [MASK] position
                        token_ids = inputs["input_ids"][batch_idx]
                        mask_positions = (token_ids == mask_token_id).nonzero(
                            as_tuple=True
                        )[0]

                        if len(mask_positions) == 0:
                            n_failed += 1
                            continue

                        mask_pos = mask_positions[0].item()

                        # Get target token ID
                        target_tokens = self._surprisal_tokenizer.encode(
                            word, add_special_tokens=False
                        )
                        if len(target_tokens) == 0:
                            n_failed += 1
                            continue

                        target_token_id = target_tokens[0]

                        # Get probability
                        probs = torch.softmax(logits[batch_idx, mask_pos], dim=-1)
                        prob = probs[target_token_id].item()

                        # Compute surprisal
                        surprisal = -np.log2(prob) if prob > 0 else 20.0
                        surprisal_values[word_idx] = surprisal
                        sentence_surprisals[word_idx] = surprisal

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        n_oom_errors += 1
                        logger.warning(
                            f"Out of memory at batch size {batch_size}. "
                            f"Consider reducing batch_size."
                        )
                        # Clear cache and continue
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        logger.debug(f"Batch error: {e}")
                    n_failed += len(batch_positions)
                except Exception as e:
                    logger.debug(f"Batch processing error: {e}")
                    n_failed += len(batch_positions)

            processed_sentences[sent_id] = sentence_surprisals
            sentences_processed += 1

            # Checkpoint (every 100 sentences for testing)
            if (sent_idx + 1) % checkpoint_every == 0:
                try:
                    checkpoint_data = {
                        "processed_sentences": processed_sentences,
                        "n_failed": n_failed,
                        "n_oom_errors": n_oom_errors,
                        "batch_size": batch_size,
                        "sentences_total": len(sentence_list),
                    }
                    with open(checkpoint_path, "wb") as f:
                        pickle.dump(checkpoint_data, f)
                    logger.info(
                        f"Checkpoint: {sentences_processed:,}/{len(sentence_list):,} sentences "
                        f"({(sentences_processed/len(sentence_list)*100):.1f}%)"
                    )
                except Exception as e:
                    logger.warning(f"Checkpoint save failed: {e}")

        # Save final cache
        logger.info(f"Saving surprisal cache to {cache_path}")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(surprisal_values, f)
            logger.info("Cache saved successfully")

            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info("Checkpoint file removed (computation complete)")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

        # Statistics
        computed_count = (~np.isnan(surprisal_values)).sum()
        valid_surprisal = surprisal_values[~np.isnan(surprisal_values)]

        logger.info(f"\nSurprisal Computation Results:")
        logger.info(f"  Computed: {computed_count:,} / {len(data):,} words")

        if n_failed > 0:
            logger.warning(f"  Failed: {n_failed:,} words")
        if n_oom_errors > 0:
            logger.warning(
                f"  Out-of-memory errors: {n_oom_errors} (reduce batch_size)"
            )

        if len(valid_surprisal) > 0:
            logger.info(f"  Mean: {valid_surprisal.mean():.2f} bits")
            logger.info(f"  Median: {np.median(valid_surprisal):.2f} bits")
            logger.info(
                f"  Range: [{valid_surprisal.min():.2f}, {valid_surprisal.max():.2f}] bits"
            )
            logger.info(f"  Std: {valid_surprisal.std():.2f} bits")

        return pd.Series(surprisal_values, index=data.index)

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
