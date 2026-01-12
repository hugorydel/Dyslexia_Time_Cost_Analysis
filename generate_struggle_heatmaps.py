#!/usr/bin/env python3
"""
Generate Dyslexic Struggle Heatmap Visualizations

This module creates page-like visualizations showing where dyslexic readers
disproportionately struggle compared to control readers, based on GAM model predictions.
"""

import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")


class StruggleHeatmapGenerator:
    """Generate heatmap visualizations of reading difficulty for dyslexic readers"""

    def __init__(
        self, model_path: str, data_path: str, output_dir: str = "struggle_heatmaps"
    ):
        """
        Initialize the heatmap generator

        Args:
            model_path: Path to cached GAM models pickle file
            data_path: Path to preprocessed data CSV
            output_dir: Directory to save output JPG files
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load models and data
        print("Loading GAM models...")
        self.models = self._load_models()

        print("Loading preprocessed data...")
        self.data = self._load_data()

        # Define color scale thresholds (in ms)
        self.thresholds = [0, 25, 50, 100, 150, 300]  # Extended upper bound

        # Create custom colormap: white -> green -> yellow -> orange -> red
        self.colormap = self._create_colormap()

        print(f"Initialization complete. Output directory: {self.output_dir}")

    def _load_models(self):
        """Load the cached GAM models from joblib cache file"""
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required. Install with: pip install joblib")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load using joblib (which is what caching_utils uses)
        print(f"  Loading from: {self.model_path}")
        skip_meta, duration_meta, gam_models = joblib.load(self.model_path)

        print(f"  ✓ Models loaded successfully")
        print(f"  ✓ Model type: {type(gam_models).__name__}")

        # Store metadata as well (might be useful)
        self.skip_meta = skip_meta
        self.duration_meta = duration_meta

        return gam_models

    def _load_data(self):
        """Load the preprocessed data with computed features"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        data = pd.read_csv(self.data_path)

        # Verify required columns
        required_cols = [
            "word_text",
            "word_length",
            "word_frequency_zipf",
            "surprisal",
            "speech_id",
            "paragraph_id",
            "sentence_id",
            "word_position",
        ]
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        print(
            f"  ✓ Loaded {len(data):,} words from {data['speech_id'].nunique()} speeches"
        )
        return data

    def _create_colormap(self):
        """Create custom colormap: white -> green -> yellow -> orange -> red"""
        colors = [
            (1.0, 1.0, 1.0),  # white (0ms)
            (0.7, 1.0, 0.7),  # light green (25ms)
            (1.0, 1.0, 0.4),  # yellow (50ms)
            (1.0, 0.7, 0.3),  # orange (100ms)
            (1.0, 0.3, 0.3),  # light red (150ms)
            (0.8, 0.0, 0.0),  # dark red (300ms+)
        ]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list("struggle", colors, N=n_bins)
        return cmap

    def predict_reading_time(self, word_features, dyslexic):
        """
        Predict reading time for a word using GAM models

        Args:
            word_features: Dict with keys 'word_length', 'word_frequency_zipf', 'surprisal'
            dyslexic: Boolean, True for dyslexic group, False for control

        Returns:
            Predicted reading time in milliseconds
        """
        # Create feature array for prediction
        features = pd.DataFrame(
            {
                "word_length": [word_features["word_length"]],
                "word_frequency_zipf": [word_features["word_frequency_zipf"]],
                "surprisal": [word_features["surprisal"]],
            }
        )

        # Determine group label
        group = "dyslexic" if dyslexic else "control"

        # Use the DyslexiaGAMModels methods with group parameter
        skip_prob = self.models.predict_skip(features, group=group)[0]
        trt = self.models.predict_trt(features, group=group)[0]

        # If trt is log-transformed, exponentiate it
        if trt < 10:  # Heuristic: if < 10, it's probably log-scale
            trt = np.exp(trt)

        # Expected reading time = (1 - skip_prob) * trt
        ert = (1 - skip_prob) * trt

        return ert

    def select_random_passages(
        self, n_passages=4, min_words=50, max_words=200, paragraphs_per_passage=1
    ):
        """
        Select passages from the dataset with maximum participant coverage

        Args:
            n_passages: Number of passages to select (default 4)
            min_words: Minimum words per passage
            max_words: Maximum words per passage
            paragraphs_per_passage: Number of consecutive paragraphs per passage

        Returns:
            List of passage DataFrames
        """
        speeches = self.data["speech_id"].unique()
        valid_passages = []

        for speech_id in speeches:
            speech_data = self.data[self.data["speech_id"] == speech_id]
            paragraph_ids = sorted(speech_data["paragraph_id"].unique())

            for i in range(len(paragraph_ids) - paragraphs_per_passage + 1):
                para_group = paragraph_ids[i : i + paragraphs_per_passage]

                # Check consecutiveness
                is_consecutive = all(
                    para_group[j + 1] - para_group[j] == 1
                    for j in range(len(para_group) - 1)
                )

                if not is_consecutive:
                    continue

                # Get words from ONE participant for display
                available_subjects = speech_data["subject_id"].unique()
                if len(available_subjects) > 0:
                    selected_subject = available_subjects[0]
                    subject_speech_data = speech_data[
                        speech_data["subject_id"] == selected_subject
                    ]

                    group_data = subject_speech_data[
                        subject_speech_data["paragraph_id"].isin(para_group)
                    ]
                    word_count = len(group_data)

                    if min_words <= word_count <= max_words:
                        # Count participants for ranking
                        all_passage_data = speech_data[
                            speech_data["paragraph_id"].isin(para_group)
                        ]
                        n_dyslexic = all_passage_data[
                            all_passage_data["dyslexic"] == True
                        ]["subject_id"].nunique()
                        n_control = all_passage_data[
                            all_passage_data["dyslexic"] == False
                        ]["subject_id"].nunique()

                        valid_passages.append(
                            {
                                "speech_id": speech_id,
                                "para_group": para_group,
                                "group_data": group_data,
                                "n_dyslexic": n_dyslexic,
                                "n_control": n_control,
                                "min_group_size": min(n_dyslexic, n_control),
                            }
                        )

        print(f"Found {len(valid_passages)} valid passages")

        # Sort by participant coverage
        valid_passages.sort(key=lambda x: x["min_group_size"], reverse=True)

        if len(valid_passages) > 0:
            print(
                f"  Best coverage: {valid_passages[0]['n_dyslexic']} dyslexic + {valid_passages[0]['n_control']} control"
            )

        # Select top N passages
        selected_passages = []
        for i in range(min(n_passages, len(valid_passages))):
            passage_info = valid_passages[i]
            passage = (
                passage_info["group_data"]
                .sort_values(["paragraph_id", "sentence_id", "word_position"])
                .copy()
            )
            selected_passages.append(
                {
                    "speech_id": passage_info["speech_id"],
                    "paragraph_ids": passage_info["para_group"],
                    "data": passage,
                    "n_dyslexic": passage_info["n_dyslexic"],
                    "n_control": passage_info["n_control"],
                }
            )

        print(f"Selected {len(selected_passages)} passages")
        return selected_passages

    def compute_struggle_scores(self, passage_df):
        """
        Compute both REAL and MODELED struggle scores

        Args:
            passage_df: DataFrame with passage words and features

        Returns:
            Tuple of (words_with_modeled_scores, words_with_real_scores, words_with_sample_sizes)
        """
        modeled_scores = []
        real_scores = []
        sample_sizes = []  # Track sample size for shrinkage
        skipped_words = 0

        for _, row in passage_df.iterrows():
            word_text = row["word_text"]

            # MODELED: Use GAM predictions
            word_features = {
                "word_length": row["word_length"],
                "word_frequency_zipf": row["word_frequency_zipf"],
                "surprisal": row["surprisal"],
            }

            # Check for invalid values
            if any(not np.isfinite(v) for v in word_features.values()):
                skipped_words += 1
                continue

            # Predict for both groups
            control_time_pred = self.predict_reading_time(word_features, dyslexic=False)
            dyslexic_time_pred = self.predict_reading_time(word_features, dyslexic=True)
            modeled_struggle = dyslexic_time_pred - control_time_pred

            # REAL: Get actual reading times
            word_id = (
                row["speech_id"],
                row["paragraph_id"],
                row["sentence_id"],
                row["word_position"],
            )

            word_instances = self.data[
                (self.data["speech_id"] == word_id[0])
                & (self.data["paragraph_id"] == word_id[1])
                & (self.data["sentence_id"] == word_id[2])
                & (self.data["word_position"] == word_id[3])
            ]

            n_total = 0  # Track sample size

            if (
                len(word_instances) > 0
                and "total_reading_time" in word_instances.columns
                and "dyslexic" in word_instances.columns
            ):
                # Get reading times (>50ms to filter skips)
                dyslexic_times = word_instances[
                    (word_instances["dyslexic"] == True)
                    & (word_instances["total_reading_time"] > 50)
                ]["total_reading_time"].dropna()

                control_times = word_instances[
                    (word_instances["dyslexic"] == False)
                    & (word_instances["total_reading_time"] > 50)
                ]["total_reading_time"].dropna()

                n_total = len(dyslexic_times) + len(control_times)

                # Compute average struggle
                if len(dyslexic_times) > 0 and len(control_times) > 0:
                    real_struggle = dyslexic_times.mean() - control_times.mean()
                else:
                    real_struggle = modeled_struggle
            else:
                real_struggle = modeled_struggle

            modeled_scores.append((word_text, modeled_struggle))
            real_scores.append((word_text, real_struggle))
            sample_sizes.append(n_total)

        if skipped_words > 0:
            print(f"  Note: Skipped {skipped_words} words with invalid features")

        return modeled_scores, real_scores, sample_sizes

    def _get_color_for_score(self, score):
        """Get RGB color for a given struggle score"""
        max_threshold = self.thresholds[-1]
        normalized = min(score / max_threshold, 1.0)
        return self.colormap(normalized)

    def create_stacked_heatmap(self, passages, score_type="modeled"):
        """
        Create a single heatmap with all paragraphs stacked vertically

        Args:
            passages: List of passage dicts
            score_type: 'modeled' (GAM) or 'real' (empirical data)
        """
        if score_type == "modeled":
            title_main = "GAM Model Predictions"
            filename = "struggle_heatmap_modeled.jpg"
        else:  # real
            title_main = "Empirical Data"
            filename = "struggle_heatmap_real.jpg"

        print(f"\n{'='*70}")
        print(
            f"Creating {score_type.upper()} visualization with {len(passages)} paragraphs"
        )
        print(f"{'='*70}")

        # Create figure - taller to accommodate all paragraphs
        fig, (ax_text, ax_colorbar) = plt.subplots(
            1, 2, figsize=(14, 16), gridspec_kw={"width_ratios": [10, 1]}
        )

        ax_text.set_xlim(0, 100)
        ax_text.set_ylim(0, 100)
        ax_text.axis("off")

        # Layout parameters
        line_height = 3.0
        char_width = 0.65
        max_line_width = 95
        paragraph_gap = 6

        # Add main title at top
        ax_text.text(
            50, 97, title_main, fontsize=16, fontweight="bold", ha="center", va="top"
        )

        # Start rendering paragraphs
        current_y = 92

        for para_idx, passage_info in enumerate(passages):
            passage_df = passage_info["data"]

            # Compute scores for this paragraph
            modeled_scores, real_scores, sample_sizes = self.compute_struggle_scores(
                passage_df
            )

            if score_type == "modeled":
                scores = modeled_scores
            else:  # real - apply shrinkage based on sample size
                scores_with_shrinkage = []
                for (word_text, score), n_samples in zip(real_scores, sample_sizes):
                    # Reliability weight: 0 when n=0, 1 when n>=20
                    # This shrinks uncertain estimates toward zero
                    reliability = min(n_samples / 20.0, 1.0)
                    shrunk_score = score * reliability
                    scores_with_shrinkage.append((word_text, shrunk_score))
                scores = scores_with_shrinkage

                # Print shrinkage statistics
                avg_samples = np.mean(sample_sizes)
                avg_reliability = np.mean([min(n / 20.0, 1.0) for n in sample_sizes])
                print(
                    f"  Para {para_idx + 1}: avg sample size={avg_samples:.1f}, avg reliability={avg_reliability:.2f}"
                )

            # Render words for this paragraph
            current_x = 2

            for i, (word_text, score) in enumerate(scores):
                word_width = len(word_text) * char_width
                space_width = char_width * 0.8

                # Line wrap
                if current_x + word_width > max_line_width and current_x > 2:
                    current_y -= line_height
                    current_x = 2

                color = self._get_color_for_score(score)

                # Draw word rectangle
                rect = mpatches.Rectangle(
                    (current_x, current_y - 2.2),
                    word_width,
                    2.8,
                    facecolor=color,
                    edgecolor="none",
                    zorder=1,
                )
                ax_text.add_patch(rect)

                # Draw word text
                ax_text.text(
                    current_x + 0.1,
                    current_y,
                    word_text,
                    fontsize=10,
                    fontfamily="monospace",
                    verticalalignment="center",
                    zorder=2,
                    color="black",
                )

                current_x += word_width

                # Add colored space
                if i < len(scores) - 1:
                    space_rect = mpatches.Rectangle(
                        (current_x, current_y - 2.2),
                        space_width,
                        2.8,
                        facecolor=color,
                        edgecolor="none",
                        zorder=1,
                    )
                    ax_text.add_patch(space_rect)
                    current_x += space_width

            # Add gap before next paragraph
            current_y -= line_height + paragraph_gap

            # Print stats for this paragraph
            score_vals = [score for _, score in scores]
            print(
                f"  Para {para_idx + 1}: {len(scores)} words, "
                f"mean={np.mean(score_vals):.1f}ms, "
                f"range=[{min(score_vals):.1f}, {max(score_vals):.1f}]ms"
            )

        # Create colorbar
        ax_colorbar.axis("off")

        gradient = np.linspace(1, 0, 256).reshape(256, 1)
        ax_colorbar.imshow(
            gradient,
            aspect="auto",
            cmap=self.colormap,
            extent=[0, 1, 0, self.thresholds[-1]],
            vmin=0,
            vmax=1,
        )

        for threshold in self.thresholds:
            y_pos = threshold
            ax_colorbar.text(
                1.5, y_pos, f"{int(threshold)}ms", fontsize=10, va="center"
            )
            ax_colorbar.plot([0, 1], [y_pos, y_pos], "k-", linewidth=0.5)

        ax_colorbar.set_xlim(0, 1)
        ax_colorbar.set_ylim(0, self.thresholds[-1])

        ax_colorbar.text(
            0.5,
            self.thresholds[-1] + 15,
            "Extra Time\n(Dyslexic vs Control)",
            fontsize=10,
            fontweight="bold",
            ha="center",
        )

        # Save
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight", format="jpg")
        plt.close()

        print(f"\n✓ Saved: {output_path}")
        print(f"{'='*70}\n")

    def create_residual_heatmap(self, passages):
        """
        Create a heatmap showing residuals (model - real)
        Blue = model under-predicts, Red = model over-predicts

        Args:
            passages: List of passage dicts
        """
        print(f"\n{'='*70}")
        print(f"Creating RESIDUAL visualization with {len(passages)} paragraphs")
        print(f"{'='*70}")

        # Create diverging colormap: blue (negative) -> white (0) -> red (positive)
        colors_diverging = [
            (0.0, 0.0, 0.8),  # dark blue (-150ms or less)
            (0.3, 0.5, 1.0),  # light blue (-75ms)
            (1.0, 1.0, 1.0),  # white (0ms - perfect prediction)
            (1.0, 0.5, 0.3),  # light red (+75ms)
            (0.8, 0.0, 0.0),  # dark red (+150ms or more)
        ]
        residual_cmap = LinearSegmentedColormap.from_list(
            "residual", colors_diverging, N=100
        )

        # Create figure
        fig, (ax_text, ax_colorbar) = plt.subplots(
            1, 2, figsize=(14, 16), gridspec_kw={"width_ratios": [10, 1]}
        )

        ax_text.set_xlim(0, 100)
        ax_text.set_ylim(0, 100)
        ax_text.axis("off")

        # Layout parameters
        line_height = 3.0
        char_width = 0.65
        max_line_width = 95
        paragraph_gap = 6

        # Add main title
        ax_text.text(
            50,
            97,
            "Model Residuals",
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="top",
        )
        ax_text.text(
            50,
            93,
            "Blue = under-predicts, Red = over-predicts",
            fontsize=10,
            style="italic",
            ha="center",
            va="top",
            color="gray",
        )

        # Start rendering paragraphs
        current_y = 89

        all_residuals = []

        for para_idx, passage_info in enumerate(passages):
            passage_df = passage_info["data"]

            # Compute scores
            modeled_scores, real_scores, sample_sizes = self.compute_struggle_scores(
                passage_df
            )

            # Apply shrinkage to real scores for fair comparison
            real_scores_shrunk = []
            for (word_text, score), n_samples in zip(real_scores, sample_sizes):
                reliability = min(n_samples / 20.0, 1.0)
                shrunk_score = score * reliability
                real_scores_shrunk.append((word_text, shrunk_score))

            # Compute residuals: model - real
            residuals = []
            for (word_text_m, model_score), (word_text_r, real_score) in zip(
                modeled_scores, real_scores_shrunk
            ):
                residual = model_score - real_score
                residuals.append((word_text_m, residual))
                all_residuals.append(residual)

            # Render words
            current_x = 2

            for i, (word_text, residual) in enumerate(residuals):
                word_width = len(word_text) * char_width
                space_width = char_width * 0.8

                # Line wrap
                if current_x + word_width > max_line_width and current_x > 2:
                    current_y -= line_height
                    current_x = 2

                # Get color for residual (-150 to +150 range)
                color = self._get_color_for_residual(residual, residual_cmap)

                # Draw word rectangle
                rect = mpatches.Rectangle(
                    (current_x, current_y - 2.2),
                    word_width,
                    2.8,
                    facecolor=color,
                    edgecolor="none",
                    zorder=1,
                )
                ax_text.add_patch(rect)

                # Draw word text
                ax_text.text(
                    current_x + 0.1,
                    current_y,
                    word_text,
                    fontsize=10,
                    fontfamily="monospace",
                    verticalalignment="center",
                    zorder=2,
                    color="black",
                )

                current_x += word_width

                # Add colored space
                if i < len(residuals) - 1:
                    space_rect = mpatches.Rectangle(
                        (current_x, current_y - 2.2),
                        space_width,
                        2.8,
                        facecolor=color,
                        edgecolor="none",
                        zorder=1,
                    )
                    ax_text.add_patch(space_rect)
                    current_x += space_width

            # Add gap before next paragraph
            current_y -= line_height + paragraph_gap

            # Print stats
            resid_vals = [r for _, r in residuals]
            print(
                f"  Para {para_idx + 1}: mean residual={np.mean(resid_vals):.1f}ms, "
                f"range=[{min(resid_vals):.1f}, {max(resid_vals):.1f}]ms"
            )

        # Print overall statistics
        print(
            f"\n  Overall: mean residual={np.mean(all_residuals):.1f}ms, "
            f"SD={np.std(all_residuals):.1f}ms"
        )

        # Create colorbar
        ax_colorbar.axis("off")

        # Symmetric gradient around zero
        thresholds = [-150, -75, 0, 75, 150]
        gradient_positions = np.linspace(0, 1, 256).reshape(256, 1)

        ax_colorbar.imshow(
            gradient_positions,
            aspect="auto",
            cmap=residual_cmap,
            extent=[0, 1, -150, 150],
            vmin=0,
            vmax=1,
        )

        # Add colorbar labels
        for threshold in thresholds:
            ax_colorbar.text(
                1.5, threshold, f"{threshold:+d}ms", fontsize=10, va="center"
            )
            ax_colorbar.plot([0, 1], [threshold, threshold], "k-", linewidth=0.5)

        ax_colorbar.set_xlim(0, 1)
        ax_colorbar.set_ylim(-150, 150)

        ax_colorbar.text(
            0.5,
            165,
            "Residual\n(Model - Real)",
            fontsize=10,
            fontweight="bold",
            ha="center",
        )

        # Save
        output_path = self.output_dir / "struggle_heatmap_residual.jpg"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight", format="jpg")
        plt.close()

        print(f"\n✓ Saved: {output_path}")
        print(f"{'='*70}\n")

    def _get_color_for_residual(self, residual, cmap):
        """Get RGB color for a residual value"""
        # Map -150 to +150 range to 0-1 for colormap
        # 0 (blue) for -150, 0.5 (white) for 0, 1.0 (red) for +150
        normalized = (residual + 150) / 300.0
        normalized = np.clip(normalized, 0, 1)
        return cmap(normalized)

    def create_feature_heatmap(self, passages, feature_name):
        """
        Create a heatmap colored by a specific linguistic feature

        Args:
            passages: List of passage dicts
            feature_name: 'word_length', 'word_frequency_zipf', or 'surprisal'
        """
        # Define feature-specific parameters
        if feature_name == "word_length":
            title_main = "Word Length"
            filename = "struggle_heatmap_length.jpg"
            thresholds = [0, 3, 6, 9, 12, 18]  # character count
            invert = False  # longer = more red
        elif feature_name == "word_frequency_zipf":
            title_main = "Word Frequency"
            filename = "struggle_heatmap_frequency.jpg"
            thresholds = [0, 2, 3, 4, 5, 7]  # zipf scale
            invert = True  # LOWER frequency (lower zipf) = more red
        elif feature_name == "surprisal":
            title_main = "Surprisal"
            filename = "struggle_heatmap_surprisal.jpg"
            thresholds = [0, 3, 6, 9, 12, 20]  # surprisal units
            invert = False  # higher surprisal = more red
        else:
            raise ValueError(f"Unknown feature: {feature_name}")

        print(f"\n{'='*70}")
        print(
            f"Creating {feature_name.upper()} visualization with {len(passages)} paragraphs"
        )
        print(f"{'='*70}")

        # Create figure
        fig, (ax_text, ax_colorbar) = plt.subplots(
            1, 2, figsize=(14, 16), gridspec_kw={"width_ratios": [10, 1]}
        )

        ax_text.set_xlim(0, 100)
        ax_text.set_ylim(0, 100)
        ax_text.axis("off")

        # Layout parameters
        line_height = 3.0
        char_width = 0.65
        max_line_width = 95
        paragraph_gap = 6

        # Add main title at top
        ax_text.text(
            50, 97, title_main, fontsize=16, fontweight="bold", ha="center", va="top"
        )

        # Start rendering paragraphs
        current_y = 92

        for para_idx, passage_info in enumerate(passages):
            passage_df = passage_info["data"]

            # Extract feature values for this paragraph
            feature_values = []
            for _, row in passage_df.iterrows():
                word_text = row["word_text"]
                feature_val = row[feature_name]

                # Skip invalid values
                if not np.isfinite(feature_val):
                    feature_val = np.median(
                        [
                            r[feature_name]
                            for _, r in passage_df.iterrows()
                            if np.isfinite(r[feature_name])
                        ]
                    )

                feature_values.append((word_text, feature_val))

            # Render words for this paragraph
            current_x = 2

            for i, (word_text, feature_val) in enumerate(feature_values):
                word_width = len(word_text) * char_width
                space_width = char_width * 0.8

                # Line wrap
                if current_x + word_width > max_line_width and current_x > 2:
                    current_y -= line_height
                    current_x = 2

                # Get color based on feature value
                color = self._get_color_for_feature(feature_val, thresholds, invert)

                # Draw word rectangle
                rect = mpatches.Rectangle(
                    (current_x, current_y - 2.2),
                    word_width,
                    2.8,
                    facecolor=color,
                    edgecolor="none",
                    zorder=1,
                )
                ax_text.add_patch(rect)

                # Draw word text
                ax_text.text(
                    current_x + 0.1,
                    current_y,
                    word_text,
                    fontsize=10,
                    fontfamily="monospace",
                    verticalalignment="center",
                    zorder=2,
                    color="black",
                )

                current_x += word_width

                # Add colored space
                if i < len(feature_values) - 1:
                    space_rect = mpatches.Rectangle(
                        (current_x, current_y - 2.2),
                        space_width,
                        2.8,
                        facecolor=color,
                        edgecolor="none",
                        zorder=1,
                    )
                    ax_text.add_patch(space_rect)
                    current_x += space_width

            # Add gap before next paragraph
            current_y -= line_height + paragraph_gap

            # Print stats
            vals = [val for _, val in feature_values]
            print(
                f"  Para {para_idx + 1}: mean={np.mean(vals):.2f}, "
                f"range=[{min(vals):.2f}, {max(vals):.2f}]"
            )

        # Create colorbar
        ax_colorbar.axis("off")

        gradient = np.linspace(1, 0, 256).reshape(256, 1)
        if invert:
            gradient = np.linspace(0, 1, 256).reshape(256, 1)  # Flip for frequency

        ax_colorbar.imshow(
            gradient,
            aspect="auto",
            cmap=self.colormap,
            extent=[0, 1, 0, thresholds[-1]],
            vmin=0,
            vmax=1,
        )

        # Add colorbar labels with appropriate units
        if feature_name == "word_length":
            unit = " chars"
        elif feature_name == "word_frequency_zipf":
            unit = ""
            # For frequency, add labels to clarify
        elif feature_name == "surprisal":
            unit = " bits"

        for threshold in thresholds:
            y_pos = threshold
            ax_colorbar.text(
                1.5, y_pos, f"{int(threshold)}{unit}", fontsize=10, va="center"
            )
            ax_colorbar.plot([0, 1], [y_pos, y_pos], "k-", linewidth=0.5)

        ax_colorbar.set_xlim(0, 1)
        ax_colorbar.set_ylim(0, thresholds[-1])

        # Add colorbar title
        if feature_name == "word_frequency_zipf":
            colorbar_title = "Frequency\n(lower = rarer)"
        elif feature_name == "word_length":
            colorbar_title = "Length\n(characters)"
        else:
            colorbar_title = "Surprisal\n(bits)"

        ax_colorbar.text(
            0.5,
            thresholds[-1] + 0.8,
            colorbar_title,
            fontsize=10,
            fontweight="bold",
            ha="center",
        )

        # Save
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight", format="jpg")
        plt.close()

        print(f"\n✓ Saved: {output_path}")
        print(f"{'='*70}\n")

    def create_scanpath_visualization(self, passages):
        """
        Create a scanpath visualization showing actual eye movements over text

        Args:
            passages: List of passage dicts
        """
        print(f"\n{'='*70}")
        print(f"Creating SCANPATH visualization with {len(passages)} paragraphs")
        print(f"{'='*70}")

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 16))

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")

        # Layout parameters
        line_height = 3.0
        char_width = 0.65
        max_line_width = 95
        paragraph_gap = 6

        # Add main title
        ax.text(
            50,
            97,
            "Eye Movement Scanpath",
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="top",
        )
        ax.text(
            50,
            93,
            "Circles = fixations (size = duration), Lines = saccades",
            fontsize=10,
            style="italic",
            ha="center",
            va="top",
            color="gray",
        )

        # Start rendering
        current_y = 89

        # For each paragraph, we'll pick ONE participant and show their scanpath
        for para_idx, passage_info in enumerate(passages):
            passage_df = passage_info["data"]
            speech_id = passage_info["speech_id"]
            para_id = passage_info["paragraph_ids"][0]

            # Get one dyslexic participant's data for this passage
            all_passage_data = self.data[
                (self.data["speech_id"] == speech_id)
                & (self.data["paragraph_id"] == para_id)
            ]

            # Find a dyslexic participant with complete data
            dyslexic_subjects = all_passage_data[all_passage_data["dyslexic"] == True][
                "subject_id"
            ].unique()

            if len(dyslexic_subjects) == 0:
                print(f"  Para {para_idx + 1}: No dyslexic data, skipping scanpath")
                continue

            selected_subject = dyslexic_subjects[0]
            subject_data = all_passage_data[
                all_passage_data["subject_id"] == selected_subject
            ].copy()
            subject_data = subject_data.sort_values(["sentence_id", "word_position"])

            # Build word position map
            word_positions = {}  # (sentence_id, word_position) -> (x, y)
            current_x = 2
            temp_y = current_y

            for _, row in passage_df.iterrows():
                word_text = row["word_text"]
                word_width = len(word_text) * char_width

                # Line wrap
                if current_x + word_width > max_line_width and current_x > 2:
                    temp_y -= line_height
                    current_x = 2

                # Store center position of this word
                word_center_x = current_x + word_width / 2
                word_center_y = temp_y
                word_positions[(row["sentence_id"], row["word_position"])] = (
                    word_center_x,
                    word_center_y,
                )

                # Draw word text in light gray
                ax.text(
                    current_x + 0.1,
                    temp_y,
                    word_text,
                    fontsize=10,
                    fontfamily="monospace",
                    verticalalignment="center",
                    color="lightgray",
                    zorder=1,
                )

                current_x += word_width + char_width * 0.8

            # Now draw fixations and saccades for this participant
            fixation_positions = []
            fixation_durations = []

            for _, row in subject_data.iterrows():
                key = (row["sentence_id"], row["word_position"])
                if (
                    key in word_positions
                    and "total_reading_time" in subject_data.columns
                ):
                    trt = row["total_reading_time"]
                    if pd.notna(trt) and trt > 50:  # Only fixated words
                        x, y = word_positions[key]
                        fixation_positions.append((x, y))
                        fixation_durations.append(trt)

            if len(fixation_positions) > 1:
                # Draw saccades (lines between fixations)
                for i in range(len(fixation_positions) - 1):
                    x1, y1 = fixation_positions[i]
                    x2, y2 = fixation_positions[i + 1]
                    ax.plot([x1, x2], [y1, y2], "b-", alpha=0.3, linewidth=1, zorder=2)

                # Draw fixations (circles sized by duration)
                for (x, y), duration in zip(fixation_positions, fixation_durations):
                    # Size circle by duration (50-500ms -> radius 0.2-1.0)
                    radius = 0.2 + (min(duration, 500) / 500.0) * 0.8
                    circle = plt.Circle(
                        (x, y), radius, color="red", alpha=0.6, zorder=3
                    )
                    ax.add_patch(circle)

                print(
                    f"  Para {para_idx + 1}: {len(fixation_positions)} fixations, subject {selected_subject}"
                )
            else:
                print(f"  Para {para_idx + 1}: Insufficient fixation data")

            # Update y position for next paragraph
            current_y = temp_y - (line_height + paragraph_gap)

        # Save
        output_path = self.output_dir / "struggle_heatmap_scanpath.jpg"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight", format="jpg")
        plt.close()

        print(f"\n✓ Saved: {output_path}")
        print(f"{'='*70}\n")

    def create_scanpath_visualization(self, passages):
        """
        Create a scanpath visualization showing eye movements over text

        Args:
            passages: List of passage dicts
        """
        print(f"\n{'='*70}")
        print(f"Creating SCANPATH visualization with {len(passages)} paragraphs")
        print(f"{'='*70}")

        # Create figure
        fig, ax_text = plt.subplots(1, 1, figsize=(14, 16))

        ax_text.set_xlim(0, 100)
        ax_text.set_ylim(0, 100)
        ax_text.axis("off")

        # Layout parameters
        line_height = 3.0
        char_width = 0.65
        max_line_width = 95
        paragraph_gap = 6

        # Add main title
        ax_text.text(
            50,
            97,
            "Scanpath Visualization",
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="top",
        )
        ax_text.text(
            50,
            93,
            "Eye movements from a single dyslexic reader",
            fontsize=10,
            style="italic",
            ha="center",
            va="top",
            color="gray",
        )

        # Start rendering
        current_y = 89

        # Track word positions for scanpath overlay
        word_positions = []  # List of (word_text, x, y, word_id)

        for para_idx, passage_info in enumerate(passages):
            passage_df = passage_info["data"]

            # Render text first to get positions
            current_x = 2

            for _, row in passage_df.iterrows():
                word_text = row["word_text"]
                word_width = len(word_text) * char_width
                space_width = char_width * 0.8

                # Line wrap
                if current_x + word_width > max_line_width and current_x > 2:
                    current_y -= line_height
                    current_x = 2

                # Draw word text (gray background)
                ax_text.text(
                    current_x + word_width / 2,
                    current_y,
                    word_text,
                    fontsize=10,
                    fontfamily="monospace",
                    verticalalignment="center",
                    ha="center",
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.1",
                        facecolor="lightgray",
                        edgecolor="none",
                        alpha=0.3,
                    ),
                )

                # Store position for scanpath
                word_id = (
                    row["speech_id"],
                    row["paragraph_id"],
                    row["sentence_id"],
                    row["word_position"],
                )
                word_center_x = current_x + word_width / 2
                word_positions.append((word_text, word_center_x, current_y, word_id))

                current_x += word_width + space_width

            # Add gap before next paragraph
            current_y -= line_height + paragraph_gap

        # Now overlay scanpath from ONE reader's fixations
        print(f"  Extracting fixation data for scanpath...")

        # Get fixation data for the passages
        all_fixations = []

        for word_text, x, y, word_id in word_positions:
            # Get all fixations on this word
            word_fixations = self.data[
                (self.data["speech_id"] == word_id[0])
                & (self.data["paragraph_id"] == word_id[1])
                & (self.data["sentence_id"] == word_id[2])
                & (self.data["word_position"] == word_id[3])
            ]

            # Filter for dyslexic readers only and valid fixations
            if (
                "dyslexic" in word_fixations.columns
                and "total_reading_time" in word_fixations.columns
            ):
                dyslexic_fixations = word_fixations[
                    (word_fixations["dyslexic"] == True)
                    & (word_fixations["total_reading_time"] > 50)
                ]

                if len(dyslexic_fixations) > 0:
                    # Pick first dyslexic reader for consistency
                    reader_id = dyslexic_fixations["subject_id"].iloc[0]
                    reader_fixation = dyslexic_fixations[
                        dyslexic_fixations["subject_id"] == reader_id
                    ].iloc[0]

                    duration = reader_fixation["total_reading_time"]

                    all_fixations.append(
                        {
                            "x": x,
                            "y": y,
                            "duration": duration,
                            "word": word_text,
                            "word_id": word_id,
                        }
                    )

        if len(all_fixations) == 0:
            print(f"  ⚠ No fixation data available for scanpath")
        else:
            print(f"  Found {len(all_fixations)} fixations")

            # Draw scanpath: lines connecting fixations
            for i in range(len(all_fixations) - 1):
                fix1 = all_fixations[i]
                fix2 = all_fixations[i + 1]

                # Draw line
                ax_text.plot(
                    [fix1["x"], fix2["x"]],
                    [fix1["y"], fix2["y"]],
                    color="blue",
                    alpha=0.3,
                    linewidth=1,
                    zorder=1,
                )

            # Draw fixation circles (sized by duration)
            durations = [f["duration"] for f in all_fixations]
            min_dur = min(durations)
            max_dur = max(durations)

            for i, fixation in enumerate(all_fixations):
                # Normalize duration to circle size
                if max_dur > min_dur:
                    norm_duration = (fixation["duration"] - min_dur) / (
                        max_dur - min_dur
                    )
                else:
                    norm_duration = 0.5

                # Circle radius: 0.3 to 1.2
                radius = 0.3 + norm_duration * 0.9

                # Color gradient: early fixations = green, late = red
                fixation_order = i / len(all_fixations)
                color = plt.cm.RdYlGn_r(fixation_order)

                circle = mpatches.Circle(
                    (fixation["x"], fixation["y"]),
                    radius,
                    facecolor=color,
                    edgecolor="darkblue",
                    linewidth=1.5,
                    alpha=0.6,
                    zorder=2,
                )
                ax_text.add_patch(circle)

            print(f"  Duration range: {min_dur:.0f}ms - {max_dur:.0f}ms")

        # Add legend
        legend_y = 5
        ax_text.text(
            50,
            legend_y,
            "Circle size = fixation duration | Color: green (early) → red (late)",
            fontsize=9,
            ha="center",
            style="italic",
            color="gray",
        )

        # Save
        output_path = self.output_dir / "struggle_heatmap_scanpath.jpg"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight", format="jpg")
        plt.close()

        print(f"\n✓ Saved: {output_path}")
        print(f"{'='*70}\n")

    def _get_color_for_feature(self, value, thresholds, invert=False):
        """Get RGB color for a feature value"""
        max_threshold = thresholds[-1]

        if invert:
            # For frequency: lower values should be more red
            normalized = 1.0 - min(value / max_threshold, 1.0)
        else:
            # For length/surprisal: higher values should be more red
            normalized = min(value / max_threshold, 1.0)

        return self.colormap(normalized)
        """
        Create a single heatmap with all paragraphs stacked vertically

        Args:
            passages: List of passage dicts
            score_type: 'modeled' (GAM) or 'real' (empirical data)
        """
        if score_type == "modeled":
            title_main = "GAM Model Predictions"
            filename = "struggle_heatmap_modeled.jpg"
        else:  # real
            # Get total participants across all passages
            total_dyslexic = sum(p.get("n_dyslexic", 0) for p in passages)
            total_control = sum(p.get("n_control", 0) for p in passages)
            avg_dyslexic = total_dyslexic // len(passages)
            avg_control = total_control // len(passages)

            title_main = "Empirical Data"
            filename = "struggle_heatmap_real.jpg"

        print(f"\n{'='*70}")
        print(
            f"Creating {score_type.upper()} visualization with {len(passages)} paragraphs"
        )
        print(f"{'='*70}")

        # Create figure - taller to accommodate all paragraphs
        fig, (ax_text, ax_colorbar) = plt.subplots(
            1, 2, figsize=(14, 16), gridspec_kw={"width_ratios": [10, 1]}
        )

        ax_text.set_xlim(0, 100)
        ax_text.set_ylim(0, 100)
        ax_text.axis("off")

        # Layout parameters
        line_height = 3.0
        char_width = 0.65
        max_line_width = 95
        paragraph_gap = 6  # Larger gap since we removed labels

        # Add main title at top
        ax_text.text(
            50, 97, title_main, fontsize=16, fontweight="bold", ha="center", va="top"
        )

        # Start rendering paragraphs
        current_y = 92

        for para_idx, passage_info in enumerate(passages):
            passage_df = passage_info["data"]
            speech_id = passage_info["speech_id"]
            para_id = passage_info["paragraph_ids"][0]

            # Compute scores for this paragraph
            modeled_scores, real_scores = self.compute_struggle_scores(passage_df)
            scores = modeled_scores if score_type == "modeled" else real_scores

            # Render words for this paragraph
            current_x = 2

            for i, (word_text, score) in enumerate(scores):
                word_width = len(word_text) * char_width
                space_width = char_width * 0.8

                # Line wrap
                if current_x + word_width > max_line_width and current_x > 2:
                    current_y -= line_height
                    current_x = 2

                color = self._get_color_for_score(score)

                # Draw word rectangle
                rect = mpatches.Rectangle(
                    (current_x, current_y - 2.2),
                    word_width,
                    2.8,
                    facecolor=color,
                    edgecolor="none",
                    zorder=1,
                )
                ax_text.add_patch(rect)

                # Draw word text
                ax_text.text(
                    current_x + 0.1,
                    current_y,
                    word_text,
                    fontsize=10,
                    fontfamily="monospace",
                    verticalalignment="center",
                    zorder=2,
                    color="black",
                )

                current_x += word_width

                # Add colored space
                if i < len(scores) - 1:
                    space_rect = mpatches.Rectangle(
                        (current_x, current_y - 2.2),
                        space_width,
                        2.8,
                        facecolor=color,
                        edgecolor="none",
                        zorder=1,
                    )
                    ax_text.add_patch(space_rect)
                    current_x += space_width

            # Add gap before next paragraph
            current_y -= line_height + paragraph_gap

            # Print stats for this paragraph
            score_vals = [score for _, score in scores]
            print(
                f"  Para {para_idx + 1}: {len(scores)} words, "
                f"mean={np.mean(score_vals):.1f}ms, "
                f"range=[{min(score_vals):.1f}, {max(score_vals):.1f}]ms"
            )

        # Create colorbar
        ax_colorbar.axis("off")

        gradient = np.linspace(1, 0, 256).reshape(256, 1)
        ax_colorbar.imshow(
            gradient,
            aspect="auto",
            cmap=self.colormap,
            extent=[0, 1, 0, self.thresholds[-1]],
            vmin=0,
            vmax=1,
        )

        for threshold in self.thresholds:
            y_pos = threshold
            ax_colorbar.text(
                1.5, y_pos, f"{int(threshold)}ms", fontsize=10, va="center"
            )
            ax_colorbar.plot([0, 1], [y_pos, y_pos], "k-", linewidth=0.5)

        ax_colorbar.set_xlim(0, 1)
        ax_colorbar.set_ylim(0, self.thresholds[-1])

        ax_colorbar.text(
            0.5,
            self.thresholds[-1] + 15,
            "Extra Time\n(Dyslexic vs Control)",
            fontsize=10,
            fontweight="bold",
            ha="center",
        )

        # Save
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight", format="jpg")
        plt.close()

        print(f"\n✓ Saved: {output_path}")
        print(f"{'='*70}\n")

    def generate_all_visualizations(self, n_passages=4):
        """
        Generate stacked visualizations for GAM, REAL, and linguistic features

        Args:
            n_passages: Number of passages to stack in each visualization (default 4)
        """
        print("=" * 70)
        print("DYSLEXIC STRUGGLE HEATMAP GENERATOR")
        print("=" * 70)

        # Select passages with maximum participant coverage
        passages = self.select_random_passages(n_passages=n_passages)

        # Generate GAM prediction visualization
        print("\nGenerating GAM prediction visualization...")
        self.create_stacked_heatmap(passages, score_type="modeled")

        # Generate empirical data visualization
        print("\nGenerating empirical data visualization...")
        self.create_stacked_heatmap(passages, score_type="real")

        # Generate residual visualization (model - real)
        print("\nGenerating residual visualization...")
        self.create_residual_heatmap(passages)

        # Generate linguistic feature visualizations
        print("\nGenerating word length visualization...")
        self.create_feature_heatmap(passages, feature_name="word_length")

        print("\nGenerating word frequency visualization...")
        self.create_feature_heatmap(passages, feature_name="word_frequency_zipf")

        print("\nGenerating surprisal visualization...")
        self.create_feature_heatmap(passages, feature_name="surprisal")

        print("\nGenerating scanpath visualization...")
        self.create_scanpath_visualization(passages)

        print("\n" + "=" * 70)
        print("✓ COMPLETE!")
        print("=" * 70)
        print(f"Generated 7 heatmap visualizations:")
        print(f"  - GAM predictions (dyslexic vs control time)")
        print(f"  - Empirical data (dyslexic vs control time, with shrinkage)")
        print(f"  - Residuals (model - real)")
        print(f"  - Word length (characters)")
        print(f"  - Word frequency (zipf scale)")
        print(f"  - Surprisal (bits)")
        print(f"  - Scanpath (eye movement trajectory)")
        print(f"Output directory: {self.output_dir.absolute()}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate dyslexic struggle heatmap visualizations"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="results_full_3k/cache_full/gam_models_quick0_logdur1.pkl",
        help="Path to cached GAM models",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="preprocessing_output/preprocessed_data.csv",
        help="Path to preprocessed data CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="struggle_heatmaps",
        help="Output directory for JPG files",
    )
    parser.add_argument(
        "--n-passages",
        type=int,
        default=4,
        help="Number of passages to stack in each visualization (creates 7 total images)",
    )

    args = parser.parse_args()

    try:
        generator = StruggleHeatmapGenerator(
            model_path=args.model_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
        )

        generator.generate_all_visualizations(n_passages=args.n_passages)

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
