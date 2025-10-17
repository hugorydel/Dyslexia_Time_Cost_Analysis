#!/usr/bin/env python3
"""
Linguistic Feature Computation for Danish Eye-Tracking Data
Implements word frequency (Zipf scale) and surprisal computation
Updated to use Leipzig Corpora frequency data
"""

import logging
import pickle
import unicodedata
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DanishLinguisticFeatures:
    """
    Compute linguistic features for Danish text:
    - Word frequency (from Leipzig Corpora, Zipf transformed)
    - Surprisal (using transformer models in causal (autoregressive) mode)
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

        # Compute Zipf frequencies per billion
        # Ensure no zeros before log
        frequencies_clipped = frequencies.clip(lower=1e-10)

        if self.use_proportions:
            zipf_freq = np.log10(frequencies_clipped * 1e9)
        else:
            zipf_freq = np.log10((frequencies_clipped / self.total_corpus_tokens) * 1e9)

        return frequencies, zipf_freq

    def _load_surprisal_model(self, model_name: str = "KennethTM/gpt2-medium-danish"):
        """
        Lazy load transformer model for surprisal computation

        Args:
            model_name: HuggingFace model identifier
        """
        if self._surprisal_model is not None:
            return  # Already loaded

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading causal language model: {model_name}")

            self._surprisal_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._surprisal_model = AutoModelForCausalLM.from_pretrained(model_name)
            self._surprisal_model.eval()  # Set to evaluation mode

            # Move to GPU if available
            if torch.cuda.is_available():
                self._surprisal_model = self._surprisal_model.cuda()
                logger.info("Using GPU for surprisal computation")
            else:
                logger.info("Using CPU for surprisal computation")

            logger.info("Causal language model loaded successfully")

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
        speech_col: str = "speech_id",
        participant_col: str = "subject_id",
        word_pos_col: str = "word_position",
        model_name: str = "KennethTM/gpt2-medium-danish",
        max_length: int = 1024,
        stride: int = 512,
        cache_file: str = None,
        checkpoint_every: int = 100,
    ) -> pd.Series:
        """
        Compute surprisal using causal GPT-2 with proper left-to-right scoring

        Args:
            data: DataFrame with word-level data
            text_col: Column containing word text
            sentence_col: Column with sentence IDs
            speech_col: Column with speech IDs
            participant_col: Column with participant IDs
            word_pos_col: Column with word positions
            model_name: HuggingFace model (GPT-2 for Danish)
            max_length: Maximum context window
            stride: Stride for sliding window
            cache_file: Cache file path
            checkpoint_every: Save checkpoint frequency

        Returns:
            Series of surprisal values (in bits)
        """

        self._load_surprisal_model(model_name)

        logger.info("Computing surprisal with causal GPT-2 (left-to-right)...")

        # === STEP 1: Identify unique sentence texts ===
        unique_sentence_groups = data.groupby([speech_col, sentence_col])

        total_participant_sentences = data.groupby(
            [participant_col, speech_col, sentence_col]
        ).ngroups

        unique_sentences = {}  # {(speech_id, sentence_id): [word1, word2, ...]}
        sentence_start_positions = {}  # {(speech_id, sentence_id): min_word_position}

        for (speech_id, sentence_id), group in unique_sentence_groups:
            sent_data = group.sort_values(word_pos_col)
            sent_data_unique = sent_data.drop_duplicates(
                subset=word_pos_col, keep="first"
            )
            words = sent_data_unique.sort_values(word_pos_col)[text_col].tolist()
            min_position = sent_data[word_pos_col].min()

            sent_key = (speech_id, sentence_id)
            unique_sentences[sent_key] = words
            sentence_start_positions[sent_key] = min_position

        logger.info(f"Dataset statistics:")
        logger.info(
            f"  Total participant-sentence instances: {total_participant_sentences:,}"
        )
        logger.info(f"  Unique sentence texts: {len(unique_sentences):,}")
        logger.info(
            f"  Average participants per sentence: {total_participant_sentences / len(unique_sentences):.1f}"
        )

        # === STEP 2: Cache setup ===
        surprisal_dir = Path("word_surprisal")
        surprisal_dir.mkdir(exist_ok=True)

        if cache_file is None:
            cache_file = surprisal_dir / "surprisal_cache_gpt2.pkl"
        else:
            cache_file = surprisal_dir / cache_file

        cache_path = Path(cache_file)
        checkpoint_path = Path(str(cache_path).replace(".pkl", "_checkpoint.pkl"))

        sentence_surprisals = {}

        if cache_path.exists():
            logger.info(f"Loading cached surprisal values from {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                    if (
                        isinstance(cached_data, dict)
                        and "sentence_surprisals" in cached_data
                    ):
                        sentence_surprisals = cached_data["sentence_surprisals"]
                        sentence_start_positions_cached = cached_data.get(
                            "sentence_start_positions", {}
                        )
                        for key, pos in sentence_start_positions_cached.items():
                            if key not in sentence_start_positions:
                                sentence_start_positions[key] = pos
                    else:
                        sentence_surprisals = cached_data
                logger.info(
                    f"Loaded surprisal for {len(sentence_surprisals):,} unique sentences"
                )
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            try:
                with open(checkpoint_path, "rb") as f:
                    checkpoint_data = pickle.load(f)
                    sentence_surprisals.update(checkpoint_data["sentence_surprisals"])
                    if "sentence_start_positions" in checkpoint_data:
                        for key, pos in checkpoint_data[
                            "sentence_start_positions"
                        ].items():
                            if key not in sentence_start_positions:
                                sentence_start_positions[key] = pos
                logger.info(
                    f"Loaded checkpoint with {len(sentence_surprisals):,} sentences"
                )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

        # === STEP 3: Compute surprisal for uncached sentences ===
        sentences_to_process = [
            (key, words)
            for key, words in unique_sentences.items()
            if key not in sentence_surprisals
        ]

        if sentences_to_process:
            logger.info(
                f"Computing surprisal for {len(sentences_to_process):,} new sentences..."
            )
            logger.info(
                "Using causal GPT-2 with sliding window (left-to-right, no future leakage)"
            )

            for sent_idx, (sent_key, words) in enumerate(
                tqdm(sentences_to_process, desc="Processing unique sentences")
            ):
                try:
                    word_surprisals = self._compute_sentence_surprisal_gpt2(
                        words, max_length, stride
                    )
                    sentence_surprisals[sent_key] = word_surprisals
                except Exception as e:
                    logger.warning(
                        f"Failed to compute surprisal for sentence {sent_key}: {e}"
                    )
                    sentence_surprisals[sent_key] = [np.nan] * len(words)

                # Checkpoint periodically
                if (sent_idx + 1) % checkpoint_every == 0:
                    try:
                        checkpoint_data = {
                            "sentence_surprisals": sentence_surprisals,
                            "sentence_start_positions": sentence_start_positions,
                            "processed": len(sentence_surprisals),
                            "total": len(unique_sentences),
                        }
                        with open(checkpoint_path, "wb") as f:
                            pickle.dump(checkpoint_data, f)
                        logger.info(
                            f"Checkpoint: {len(sentence_surprisals):,} / {len(unique_sentences):,} unique sentences processed"
                        )
                    except Exception as e:
                        logger.warning(f"Checkpoint failed: {e}")

            # Save final cache
            logger.info(f"Saving surprisal cache to {cache_path}")
            try:
                cache_data = {
                    "sentence_surprisals": sentence_surprisals,
                    "sentence_start_positions": sentence_start_positions,
                }
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                logger.info("Cache saved successfully")

                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    logger.info("Checkpoint file removed")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

        # === STEP 4: Broadcast to all participant instances ===
        logger.info("Broadcasting surprisal values to all participant instances...")

        def map_surprisal(row):
            """Map surprisal value for a single row"""
            sent_key = (row[speech_col], row[sentence_col])
            global_word_pos = row[word_pos_col]

            if sent_key in sentence_surprisals and sent_key in sentence_start_positions:
                sent_start_pos = sentence_start_positions[sent_key]
                sentence_relative_pos = global_word_pos - sent_start_pos

                sent_surp = sentence_surprisals[sent_key]
                if 0 <= sentence_relative_pos < len(sent_surp):
                    return sent_surp[sentence_relative_pos]
            return np.nan

        logger.info("  Mapping surprisal to words...")
        surprisal_series = data.apply(map_surprisal, axis=1)
        surprisal_values = surprisal_series.values

        logger.info(f"  Mapped {(~pd.isna(surprisal_series)).sum():,} words")

        # === STEP 5: Statistics ===
        computed_count = (~np.isnan(surprisal_values)).sum()
        valid_surprisal = surprisal_values[~np.isnan(surprisal_values)]

        logger.info(f"\nSurprisal Computation Results:")
        logger.info(f"  Unique sentences processed: {len(sentence_surprisals):,}")
        logger.info(f"  Total words with surprisal: {computed_count:,} / {len(data):,}")

        if len(valid_surprisal) > 0:
            logger.info(f"  Mean: {valid_surprisal.mean():.2f} bits")
            logger.info(f"  Median: {np.median(valid_surprisal):.2f} bits")
            logger.info(f"  Std: {valid_surprisal.std():.2f} bits")
            logger.info(
                f"  Range: [{valid_surprisal.min():.2f}, {valid_surprisal.max():.2f}] bits"
            )

        return pd.Series(surprisal_values, index=data.index)

    def _compute_sentence_surprisal_gpt2(
        self, words: list, max_length: int = 1024, stride: int = 512
    ) -> list:
        """
        Compute word-level surprisal for a sentence using causal GPT-2

        Args:
            words: List of words in sentence
            max_length: Maximum context window
            stride: Stride for sliding window

        Returns:
            List of surprisal values (one per word, in bits)
        """
        import torch
        from torch.nn import CrossEntropyLoss

        # Tokenize with word-level mapping
        encoding = self._surprisal_tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        )

        input_ids = encoding["input_ids"][0]  # Shape: [seq_len]

        # Use word_ids() for direct token→word mapping
        word_ids_mapping = encoding.word_ids(batch_index=0)  # List[Optional[int]]

        # Move to GPU if available
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        # === Handle long sequences with sliding window ===
        seq_len = len(input_ids)

        if seq_len <= max_length:
            # Short sequence: single forward pass
            token_nlls = self._compute_token_nlls(input_ids.unsqueeze(0))
        else:
            # Long sequence: sliding window with overlap
            token_nlls = torch.full((seq_len,), float("nan"))

            # Process windows
            for start_idx in range(0, seq_len, stride):
                end_idx = min(start_idx + max_length, seq_len)
                window_ids = input_ids[start_idx:end_idx].unsqueeze(0)

                # Compute NLLs for this window
                window_nlls = self._compute_token_nlls(window_ids)

                # Decide which tokens to keep from this window
                # Skip first `stride` tokens to ensure full left context
                overlap = stride if start_idx > 0 else 0
                keep_start = overlap
                keep_end = len(window_nlls)

                # Store NLLs (only if not already filled from previous window)
                for i in range(keep_start, keep_end):
                    global_idx = start_idx + i
                    if global_idx < seq_len and torch.isnan(token_nlls[global_idx]):
                        token_nlls[global_idx] = window_nlls[i]

                if end_idx >= seq_len:
                    break

        # === Map token-level NLLs to word-level surprisal ===
        token_nlls = token_nlls.cpu().numpy()

        # Aggregate token NLLs by word
        word_nlls = [[] for _ in range(len(words))]

        for token_idx, word_idx in enumerate(word_ids_mapping):
            # word_idx is None for special tokens (shouldn't happen with add_special_tokens=False)
            if word_idx is not None and token_idx < len(token_nlls):
                nll = token_nlls[token_idx]
                if not np.isnan(nll):
                    word_nlls[word_idx].append(nll)

        # Sum NLLs per word and convert to bits
        word_surprisals = []
        for word_idx, nlls in enumerate(word_nlls):
            if len(nlls) > 0:
                total_nll = np.sum(nlls)
                surprisal_bits = total_nll / np.log(2)  # Convert nats to bits
                word_surprisals.append(surprisal_bits)
            else:
                # No tokens for this word (shouldn't happen, but handle gracefully)
                logger.warning(
                    f"No tokens found for word {word_idx}: '{words[word_idx]}'"
                )
                word_surprisals.append(np.nan)

        return word_surprisals

    def _compute_token_nlls(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token negative log-likelihoods (NLLs) using causal language model

        Args:
            input_ids: Token IDs [batch_size, seq_len]

        Returns:
            NLLs for each token [seq_len] (position i = NLL of token i given tokens < i)
        """
        import torch
        from torch.nn import CrossEntropyLoss

        # Forward pass
        with torch.no_grad():
            outputs = self._surprisal_model(input_ids)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # Compute cross-entropy loss per position
        # Shift: predict token i from tokens < i
        shift_logits = logits[:, :-1, :].contiguous()  # [batch, seq_len-1, vocab]
        shift_labels = input_ids[:, 1:].contiguous()  # [batch, seq_len-1]

        # Per-token NLL
        loss_fct = CrossEntropyLoss(reduction="none")
        token_nlls = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # Reshape and add NaN for first token (no NLL for position 0)
        token_nlls = token_nlls.view(input_ids.size(0), -1)  # [batch, seq_len-1]

        # Prepend NaN for first token (no context for position 0)
        first_token_nll = torch.full(
            (input_ids.size(0), 1), float("nan"), device=token_nlls.device
        )
        token_nlls = torch.cat([first_token_nll, token_nlls], dim=1)  # [batch, seq_len]

        return token_nlls[0]  # Return single sequence [seq_len]

    def add_all_features(
        self,
        data: pd.DataFrame,
        text_col: str = "word_text",
        exclude_missing_frequencies: bool = True,
    ) -> pd.DataFrame:
        """
        Add all linguistic features to dataframe

        Args:
            data: Input dataframe with word-level data
            text_col: Column containing word text
            exclude_missing_frequencies: If True, remove words not found in frequency list

        Returns:
            DataFrame with added columns:
                - word_length: character count (if not present)
                - word_frequency_raw: raw proportion or frequency
                - word_frequency_zipf: Zipf-transformed frequency (1-7 scale)
                - surprisal: -log2 probability
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

        # 3. Surprisal
        if "sentence_id" not in data.columns:
            logger.warning("sentence_id column required for surprisal. Skipping.")
        else:
            data["surprisal"] = self.compute_surprisal(
                data,
                text_col="word_text",
                sentence_col="sentence_id",
                speech_col="speech_id",
                participant_col="subject_id",
                word_pos_col="word_position",
            )
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
        logger.info(
            f"  Surprisal: mean={data['surprisal'].mean():.2f}, "
            f"std={data['surprisal'].std():.2f}"
        )

        return data
