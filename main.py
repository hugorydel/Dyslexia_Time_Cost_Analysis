#!/usr/bin/env python3
"""
Main Analysis Orchestrator for Dyslexia Time Cost Analysis
Uses ExtractedFeatures as primary data source
"""

import io
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Import configuration
import config
from utils.continuous_models import run_continuous_models

# Import utilities
from utils.data_utils import (
    clean_word_data,
    create_additional_measures,
    identify_dyslexic_subjects,
    load_extracted_features,
    load_participant_stats,
    merge_with_participant_stats,
)
from utils.gap_decomposition import run_gap_decomposition
from utils.hypothesis_testing_utils import prepare_hypothesis_testing_data
from utils.hypothesis_visualization import create_hypothesis_testing_visualizations

# Import linguistic features
from utils.linguistic_features import DanishLinguisticFeatures
from utils.misc_utils import (
    create_output_directories,
    save_json_results,
    setup_logging,
    validate_config,
)
from utils.quartile_analysis import run_quartile_analysis
from utils.results_reporting import generate_hypothesis_testing_report
from utils.sensitivity_analysis import run_sensitivity_analyses
from utils.stats_utils import calculate_basic_statistics, calculate_group_summary_stats
from utils.visualization_utils import (
    create_exploratory_plots,
    create_group_summary_plots,
)

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


class DyslexiaTimeAnalysisPipeline:
    """
    Main pipeline using ExtractedFeatures as primary data source
    """

    def __init__(self, config_obj=None):
        self.config = config_obj or config

        # Validate configuration
        validate_config(self.config)

        # Create output directories
        self.results_dir = Path("results")
        self.directories = create_output_directories(self.results_dir)

        # Initialize linguistic features processor
        self.linguistic_features = None

        logger.info("Dyslexia Time Analysis Pipeline initialized")

    def load_copco_data(self) -> pd.DataFrame:
        """
        Load CopCo dataset using ExtractedFeatures as primary source
        """
        logger.info("Loading CopCo dataset...")

        copco_path = Path(self.config.COPCO_PATH)
        if not copco_path.exists():
            raise FileNotFoundError(f"CopCo data not found at {copco_path}")

        # Load ExtractedFeatures
        extracted_features_path = copco_path / "ExtractedFeatures"
        dataset_stats_path = copco_path / "DatasetStatistics"

        if not extracted_features_path.exists():
            raise FileNotFoundError(
                f"ExtractedFeatures folder not found at {extracted_features_path}"
            )

        # Load the word-level eye-tracking data
        word_data = load_extracted_features(extracted_features_path)

        # Load participant statistics for group assignment
        if dataset_stats_path.exists():
            participant_stats = load_participant_stats(dataset_stats_path)
            word_data = merge_with_participant_stats(word_data, participant_stats)
            self._participant_stats = participant_stats
        else:
            self._participant_stats = pd.DataFrame()

        # Basic preprocessing
        word_data = self._preprocess_word_data(word_data)

        logger.info(
            f"Loaded: {len(word_data):,} words from {word_data['subject_id'].nunique()} participants"
        )
        return word_data

    def _preprocess_word_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the word-level data"""

        if data.empty:
            raise ValueError("Input data is empty")

        # Clean the data
        data = clean_word_data(data, self.config)

        # Assign dyslexic groups using participant stats
        if "dyslexic" not in data.columns:
            dyslexic_subjects, subject_lists = identify_dyslexic_subjects(
                data, self._participant_stats
            )
            data["dyslexic"] = data["subject_id"].isin(dyslexic_subjects)

            # Store group lists for JSON output
            self._dyslexic_subjects = subject_lists["dyslexic"]
            self._control_subjects = subject_lists["control"]

        # Create additional measures for analysis (includes skipping probability)
        data = create_additional_measures(data)

        # Calculate skipping probabilities
        logger.info("Calculating skipping probabilities...")
        try:
            from utils.skipping_utils import calculate_skipping_from_extracted_features

            skipping_results = calculate_skipping_from_extracted_features(data)

            if skipping_results:
                logger.info(
                    f"Skipping analysis: {skipping_results.get('overall_skipping_rate', 0)*100:.1f}% overall rate"
                )
                self._skipping_analysis = skipping_results
            else:
                logger.warning("Skipping analysis failed")
                self._skipping_analysis = {}

        except Exception as e:
            logger.warning(f"Skipping analysis failed: {e}")
            self._skipping_analysis = {}

        return data

    def compute_linguistic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute linguistic features for the dataset

        Args:
            data: Word-level dataframe

        Returns:
            DataFrame with added linguistic features:
                - word_length
                - word_frequency_zipf (Zipf scale 1-7)
                - surprisal
        """
        logger.info("=" * 60)
        logger.info("COMPUTING LINGUISTIC FEATURES")
        logger.info("=" * 60)

        # Initialize linguistic features processor
        try:
            self.linguistic_features = DanishLinguisticFeatures()
        except FileNotFoundError as e:
            logger.error(f"Failed to load frequency data: {e}")
            raise

        # Add all features
        data = self.linguistic_features.add_all_features(data, text_col="word_text")

        return data

    def analyze_linguistic_features(self, data: pd.DataFrame) -> dict:
        """
        Analyze linguistic features and their relationship with reading measures

        Args:
            data: DataFrame with linguistic features and eye-tracking measures

        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing linguistic feature effects...")

        results = {
            "linguistic_features_summary": {},
            "frequency_effects": {},
            "length_effects": {},
            "surprisal_effects": {},
        }

        # Overall feature distributions
        feature_cols = ["word_length", "word_frequency_zipf"]
        if "surprisal" in data.columns:
            feature_cols.append("surprisal")

        for col in feature_cols:
            results["linguistic_features_summary"][col] = {
                "mean": float(data[col].mean()),
                "std": float(data[col].std()),
                "min": float(data[col].min()),
                "max": float(data[col].max()),
                "median": float(data[col].median()),
                "missing_pct": float(data[col].isna().sum() / len(data) * 100),
            }

        # Frequency effects (by Zipf bins) - only for fixated words
        if "word_frequency_zipf" in data.columns:
            freq_bins = pd.cut(
                data["word_frequency_zipf"],
                bins=[0, 4, 5, 6, 10],
                labels=["Low (≤4)", "Medium (4-5)", "High (5-6)", "Very High (>6)"],
            )

            for measure in [
                "total_reading_time",
                "gaze_duration",
                "first_fixation_duration",
            ]:
                if measure in data.columns:
                    fixated = data[data["was_fixated"] == True]
                    freq_effects = fixated.groupby(freq_bins, observed=True)[
                        measure
                    ].agg(["mean", "std", "count"])

                    results["frequency_effects"][measure] = {
                        str(label): {
                            "mean": float(row["mean"]),
                            "std": float(row["std"]),
                            "count": int(row["count"]),
                        }
                        for label, row in freq_effects.iterrows()
                    }

        # Length effects (by word length bins)
        length_bins = pd.cut(
            data["word_length"],
            bins=[0, 3, 5, 7, 30],
            labels=["Short (1-3)", "Medium (4-5)", "Long (6-7)", "Very Long (8+)"],
        )

        for measure in ["total_reading_time", "gaze_duration", "skipping_probability"]:
            if measure in data.columns:
                if measure == "skipping_probability":
                    length_effects = data.groupby(length_bins, observed=True)[
                        measure
                    ].agg(["mean", "count"])
                else:
                    fixated = data[data["was_fixated"] == True]
                    length_effects = fixated.groupby(length_bins, observed=True)[
                        measure
                    ].agg(["mean", "std", "count"])

                results["length_effects"][measure] = {
                    str(label): {"mean": float(row["mean"]), "count": int(row["count"])}
                    for label, row in length_effects.iterrows()
                }

        # Surprisal effects (by surprisal bins) - only for fixated words
        if "surprisal" in data.columns:
            # Filter to fixated words with valid surprisal
            fixated = data[data["was_fixated"] == True]
            fixated_with_surprisal = fixated[fixated["surprisal"].notna()].copy()

            if len(fixated_with_surprisal) > 0:
                # Create bins ONCE on the fixated data
                fixated_with_surprisal["surprisal_bin"] = pd.qcut(
                    fixated_with_surprisal["surprisal"],
                    q=5,
                    labels=["Very Low", "Low", "Medium", "High", "Very High"],
                    duplicates="drop",
                )

                # Now compute effects for each measure using the same bins
                for measure in [
                    "total_reading_time",
                    "gaze_duration",
                    "first_fixation_duration",
                ]:
                    if measure in fixated_with_surprisal.columns:
                        surp_effects = fixated_with_surprisal.groupby(
                            "surprisal_bin", observed=True
                        )[measure].agg(["mean", "std", "count"])

                        results["surprisal_effects"][measure] = {
                            str(label): {
                                "mean": float(row["mean"]),
                                "std": float(row["std"]),
                                "count": int(row["count"]),
                            }
                            for label, row in surp_effects.iterrows()
                        }

        # Group differences in linguistic feature effects
        if "dyslexic" in data.columns:
            results["group_differences"] = {}

            for group_name, group_data in data.groupby("dyslexic"):
                group_label = "dyslexic" if group_name else "control"
                results["group_differences"][group_label] = {}

                fixated = group_data[group_data["was_fixated"] == True]

                for feature in ["word_frequency_zipf", "word_length", "surprisal"]:
                    if (
                        feature in fixated.columns
                        and "total_reading_time" in fixated.columns
                    ):
                        # For surprisal, drop NaN values before correlation
                        if feature == "surprisal":
                            valid_data = fixated[
                                [feature, "total_reading_time"]
                            ].dropna()
                            if len(valid_data) > 0:
                                corr = valid_data.corr().iloc[0, 1]
                            else:
                                corr = np.nan
                        else:
                            # Correlation with reading time
                            corr = (
                                fixated[[feature, "total_reading_time"]]
                                .corr()
                                .iloc[0, 1]
                            )

                        results["group_differences"][group_label][
                            f"{feature}_correlation"
                        ] = float(corr)

        logger.info("Linguistic feature analysis complete")
        return results

    def run_exploratory_analysis(self) -> dict:
        """
        Run exploratory analysis on the word-level data
        """

        logger.info("Running exploratory analysis...")

        # Load data
        data = self.load_copco_data()

        # Compute linguistic features
        data = self.compute_linguistic_features(data)

        self.save_processed_data(data, "processed_data_full.csv")

        # Get skipping analysis results if available
        skipping_analysis = getattr(self, "_skipping_analysis", {})

        # Calculate basic statistics
        summary = calculate_basic_statistics(data, skipping_analysis)

        # Add participant group details to summary
        if hasattr(self, "_dyslexic_subjects") and hasattr(self, "_control_subjects"):
            summary["participant_groups"] = {
                "dyslexic_subjects": self._dyslexic_subjects,
                "control_subjects": self._control_subjects,
            }

        # Calculate group statistics if dyslexic column exists
        if "dyslexic" in data.columns:
            group_stats = calculate_group_summary_stats(data, skipping_analysis)
            summary.update(group_stats)

        # Analyze linguistic features
        linguistic_analysis = self.analyze_linguistic_features(data)
        summary["linguistic_features"] = linguistic_analysis

        # Save results
        save_json_results(summary, self.results_dir / "exploratory_summary.json")

        # Create plots
        create_exploratory_plots(data, self.results_dir / "exploratory_plots.png")
        create_group_summary_plots(
            data, self.results_dir / "group_summary_plots.png", skipping_analysis
        )

        # Create linguistic feature plots
        self._create_linguistic_feature_plots(data)

        logger.info(f"Analysis complete. Results saved to {self.results_dir}")

        return summary

    def run_hypothesis_testing_analysis(self) -> dict:
        """
        Run comprehensive hypothesis testing analysis
        Tests all three hypotheses with rigorous statistical methods
        """
        logger.info("=" * 80)
        logger.info("HYPOTHESIS TESTING ANALYSIS")
        logger.info("=" * 80)

        # Load data
        data = self.load_copco_data()
        data = self.compute_linguistic_features(data)

        # PHASE 1: Data Preparation
        data, quartiles, scalers, vif = prepare_hypothesis_testing_data(data)

        # PHASE 2: Part A - Quartile Analysis
        quartile_results = run_quartile_analysis(data)

        # PHASE 3: Part B - Continuous Models
        continuous_results = run_continuous_models(data)

        # PHASE 4: Part C - Gap Decomposition
        gap_results = run_gap_decomposition(continuous_results["predictions"])

        # PHASE 5: Sensitivity Analyses
        sensitivity_results = run_sensitivity_analyses(data)

        # PHASE 6: Visualizations
        create_hypothesis_testing_visualizations(
            data, quartile_results, continuous_results, gap_results, self.results_dir
        )

        # PHASE 7: Generate Report
        report = generate_hypothesis_testing_report(
            quartile_results,
            continuous_results,
            gap_results,
            sensitivity_results,
            quartiles,
            scalers,
            vif,
        )

        # Save comprehensive results
        save_json_results(report, self.results_dir / "hypothesis_testing_results.json")

        logger.info(
            f"\nHypothesis testing complete! Results saved to {self.results_dir}"
        )

        return report

    def _create_linguistic_feature_plots(self, data: pd.DataFrame):
        """Create visualizations for linguistic features"""

        logger.info("Creating linguistic feature visualizations...")

        # Set style
        sns.set_style("whitegrid")

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # Only analyze fixated words for reading time measures
        fixated = data[data["was_fixated"] == True].copy()

        # 1. Frequency distribution
        ax1 = plt.subplot(3, 3, 1)
        sns.histplot(data=data, x="word_frequency_zipf", bins=30, ax=ax1)
        ax1.set_xlabel("Word Frequency (Zipf Scale)")
        ax1.set_ylabel("Count")
        ax1.set_title("Distribution of Word Frequencies")
        ax1.legend()

        # 2. Length distribution
        ax2 = plt.subplot(3, 3, 2)
        sns.histplot(data=data, x="word_length", bins=range(1, 20), ax=ax2)
        ax2.set_xlabel("Word Length (characters)")
        ax2.set_ylabel("Count")
        ax2.set_title("Distribution of Word Lengths")

        # 3. Frequency vs Length
        ax3 = plt.subplot(3, 3, 3)
        sns.scatterplot(
            data=data.sample(min(5000, len(data))),
            x="word_length",
            y="word_frequency_zipf",
            alpha=0.3,
            ax=ax3,
        )
        ax3.set_xlabel("Word Length")
        ax3.set_ylabel("Word Frequency (Zipf)")
        ax3.set_title("Frequency vs Length")

        # 4. Surprisal distribution
        ax4 = plt.subplot(3, 3, 4)
        if "surprisal" in data.columns:
            surprisal_data = data["surprisal"].dropna()
            sns.histplot(data=surprisal_data, bins=40, ax=ax4, color="purple")
            ax4.set_xlabel("Surprisal (bits)")
            ax4.set_ylabel("Count")
            ax4.set_title("Distribution of Word Surprisal")
            ax4.axvline(
                x=surprisal_data.median(),
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Median: {surprisal_data.median():.1f}",
            )
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, "Surprisal not computed", ha="center", va="center")
            ax4.set_title("Distribution of Word Surprisal")

        # 5. Length effect on reading time
        ax5 = plt.subplot(3, 3, 5)
        length_groups = fixated.groupby("word_length")["total_reading_time"].mean()
        length_groups[length_groups.index <= 15].plot(kind="line", ax=ax5, marker="o")
        ax5.set_xlabel("Word Length")
        ax5.set_ylabel("Mean Total Reading Time (ms)")
        ax5.set_title("Length Effect on Reading Time")
        ax5.grid(True, alpha=0.3)

        # 6. Surprisal effect by group
        ax6 = plt.subplot(3, 3, 6)
        if "surprisal" in data.columns and "dyslexic" in fixated.columns:
            fixated_with_surp = fixated[fixated["surprisal"].notna()].copy()
            surprisal_bins = pd.qcut(
                fixated_with_surp["surprisal"],
                q=5,
                labels=["Very Low", "Low", "Medium", "High", "Very High"],
                duplicates="drop",
            )
            fixated_with_surp["surprisal_bin"] = surprisal_bins

            for group_name, group_data in fixated_with_surp.groupby("dyslexic"):
                label = "Dyslexic" if group_name else "Control"
                surp_effect = group_data.groupby("surprisal_bin", observed=True)[
                    "total_reading_time"
                ].mean()
                surp_effect.plot(
                    kind="line", ax=ax6, marker="o", label=label, linewidth=2
                )

            ax6.set_xlabel("Surprisal Level")
            ax6.set_ylabel("Mean Total Reading Time (ms)")
            ax6.set_title("Surprisal Effect by Group")
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.tick_params(axis="x", rotation=45)
        else:
            ax6.text(0.5, 0.5, "Surprisal not available", ha="center", va="center")
            ax6.set_title("Surprisal Effect by Group")

        # 7. Group comparison - frequency effect
        if "dyslexic" in fixated.columns:
            ax7 = plt.subplot(3, 3, 7)
            for group_name, group_data in fixated.groupby("dyslexic"):
                label = "Dyslexic" if group_name else "Control"
                freq_bins = pd.cut(group_data["word_frequency_zipf"], bins=4)
                freq_effect = group_data.groupby(freq_bins, observed=True)[
                    "total_reading_time"
                ].mean()
                freq_effect.plot(kind="line", ax=ax7, marker="o", label=label)
            ax7.set_xlabel("Frequency Bin (Zipf)")
            ax7.set_ylabel("Mean Total Reading Time (ms)")
            ax7.set_title("Frequency Effect by Group")
            ax7.legend()
            ax7.grid(True, alpha=0.3)

            # 8. Group comparison - length effect
            ax8 = plt.subplot(3, 3, 8)
            for group_name, group_data in fixated.groupby("dyslexic"):
                label = "Dyslexic" if group_name else "Control"
                length_effect = group_data.groupby("word_length")[
                    "total_reading_time"
                ].mean()
                length_effect[length_effect.index <= 12].plot(
                    kind="line", ax=ax8, marker="o", label=label
                )
            ax8.set_xlabel("Word Length")
            ax8.set_ylabel("Mean Total Reading Time (ms)")
            ax8.set_title("Length Effect by Group")
            ax8.legend()
            ax8.grid(True, alpha=0.3)

            # 9. Group comparison - skipping by frequency
            ax9 = plt.subplot(3, 3, 9)
            for group_name, group_data in data.groupby("dyslexic"):
                label = "Dyslexic" if group_name else "Control"
                freq_bins = pd.cut(group_data["word_frequency_zipf"], bins=4)
                skip_effect = group_data.groupby(freq_bins, observed=True)[
                    "skipping_probability"
                ].mean()
                skip_effect.plot(kind="line", ax=ax9, marker="o", label=label)
            ax9.set_xlabel("Frequency Bin (Zipf)")
            ax9.set_ylabel("Skipping Probability")
            ax9.set_title("Frequency Effect on Skipping by Group")
            ax9.legend()
            ax9.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.results_dir / "linguistic_features_plots.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Linguistic feature plots saved to {output_path}")

    def save_processed_data(
        self, data: pd.DataFrame, filename: str = "processed_data.csv"
    ) -> None:
        """
        Save the fully processed dataset to CSV

        Args:
            data: Processed DataFrame with all features
            filename: Output filename
        """
        output_path = self.results_dir / filename

        logger.info(f"Saving processed data to {output_path}")
        logger.info(f"  Shape: {data.shape}")
        logger.info(f"  Columns: {len(data.columns)}")

        # Save to CSV
        data.to_csv(output_path, index=False, encoding="utf-8")

        logger.info(f"Processed data saved successfully")

        # Also save a data dictionary
        self._save_data_dictionary(data, output_path.with_suffix(".txt"))

    def _save_data_dictionary(self, data: pd.DataFrame, filepath: Path) -> None:
        """Save a data dictionary describing all columns"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("DATA DICTIONARY - Processed Dyslexia Dataset\n")
            f.write("=" * 80 + "\n\n")

            f.write(
                f"Dataset Shape: {data.shape[0]:,} rows × {data.shape[1]} columns\n"
            )
            f.write(
                f"Date Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write("COLUMN DESCRIPTIONS:\n")
            f.write("-" * 80 + "\n\n")

            # Column descriptions
            column_info = {
                "subject_id": "Participant identifier (P01-P58)",
                "trial_id": "Trial number",
                "speech_id": "Speech/text identifier",
                "paragraph_id": "Paragraph number within speech",
                "sentence_id": "Sentence number within speech",
                "word_position": "Word position in text (global)",
                "word_text": "The actual word text",
                "word_length": "Word length in characters",
                # Eye tracking measures
                "n_fixations": "Number of fixations on word (0 = skipped)",
                "first_fixation_duration": "Duration of first fixation (ms)",
                "gaze_duration": "Sum of all first-pass fixations (ms)",
                "word_go_past_time": "Go-past time / regression path duration (ms)",
                "total_reading_time": "Total time spent on word across all passes (ms)",
                "landing_position": "Landing position of first fixation",
                # Derived eye measures
                "skipped": "Boolean: word was skipped (n_fixations == 0)",
                "was_fixated": "Boolean: word was fixated (n_fixations > 0)",
                "skipping_probability": "Probability word was skipped (0 or 1)",
                "regression_probability": "Estimated regression probability",
                # Linguistic features
                "word_frequency_raw": "Raw word frequency/proportion from corpus",
                "word_frequency_zipf": "Zipf-transformed frequency (1-7 scale, higher = more frequent)",
                "surprisal": "Word surprisal in bits (from Danish BERT)",
                # Participant info
                "dyslexic": "Boolean: participant has dyslexia",
                "age": "Participant age",
                "sex": "Participant sex",
                "comprehension_accuracy": "Reading comprehension score",
                "words_per_minute": "Reading speed (words/minute)",
            }

            for col in data.columns:
                dtype = str(data[col].dtype)
                n_missing = data[col].isna().sum()
                pct_missing = (n_missing / len(data)) * 100

                description = column_info.get(col, "")

                f.write(f"{col}\n")
                f.write(f"  Type: {dtype}\n")
                if description:
                    f.write(f"  Description: {description}\n")
                f.write(f"  Missing: {n_missing:,} ({pct_missing:.2f}%)\n")

                # Add sample values for key columns
                if col in ["word_text", "subject_id"]:
                    sample = data[col].dropna().unique()[:5]
                    f.write(f"  Sample values: {', '.join(map(str, sample))}\n")
                elif data[col].dtype in ["float64", "int64"]:
                    f.write(
                        f"  Range: [{data[col].min():.2f}, {data[col].max():.2f}]\n"
                    )
                    f.write(
                        f"  Mean: {data[col].mean():.2f}, Median: {data[col].median():.2f}\n"
                    )

                f.write("\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("GROUP INFORMATION:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total participants: {data['subject_id'].nunique()}\n")
            if "dyslexic" in data.columns:
                dyslexic_counts = data.groupby("dyslexic")["subject_id"].nunique()
                f.write(f"  Control: {dyslexic_counts.get(False, 0)}\n")
                f.write(f"  Dyslexic: {dyslexic_counts.get(True, 0)}\n")

        logger.info(f"Data dictionary saved to {filepath}")


def display_menu() -> str:
    """Display analysis menu and get user choice"""
    print("\nDyslexia Time Cost Analysis Pipeline")
    print("=" * 50)
    print("1. Run exploratory analysis")
    print("2. Run hypothesis testing analysis")  # NEW
    print("3. Exit")

    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice in ["1", "2", "3"]:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")


def main():
    """Main entry point"""
    try:
        # Initialize pipeline
        pipeline = DyslexiaTimeAnalysisPipeline()
        logger.info("Dyslexia Time Analysis Pipeline initialized")

        while True:
            choice = display_menu()

            if choice == "1":
                logger.info("Running exploratory analysis...")
                pipeline.run_exploratory_analysis()
            elif choice == "2":
                logger.info("Running hypothesis testing analysis...")
                pipeline.run_hypothesis_testing_analysis()  # NEW
            elif choice == "3":
                logger.info("Exiting...")
                break

        logger.info("Analysis complete!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
