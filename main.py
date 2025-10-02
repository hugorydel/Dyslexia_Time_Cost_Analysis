#!/usr/bin/env python3
"""
Main Analysis Orchestrator for Dyslexia Time Cost Analysis
Uses ExtractedFeatures as primary data source
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import configuration
import config

# Import utilities
from utils.data_utils import (
    clean_word_data,
    create_additional_measures,
    identify_dyslexic_subjects,
    load_extracted_features,
    load_participant_stats,
    merge_with_participant_stats,
)

# NEW: Import linguistic features
from utils.linguistic_features import (
    DanishLinguisticFeatures,
    validate_linguistic_features,
)
from utils.misc_utils import (
    create_output_directories,
    save_json_results,
    setup_logging,
    validate_config,
)
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

        # Create additional measures for analysis (includes simple skipping)
        data = create_additional_measures(data)

        # Simple skipping analysis using n_fixations directly
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

    def compute_linguistic_features(
        self, data: pd.DataFrame, compute_surprisal: bool = False
    ) -> pd.DataFrame:
        """
        Compute linguistic features for the dataset

        Args:
            data: Word-level dataframe
            compute_surprisal: Whether to compute surprisal (computationally expensive!)

        Returns:
            DataFrame with added linguistic features:
                - word_length
                - word_frequency_zipf (Zipf scale 1-7)
                - surprisal (if requested)
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
        data = self.linguistic_features.add_all_features(
            data, compute_surprisal=compute_surprisal, text_col="word_text"
        )

        # Validate features
        validation = validate_linguistic_features(data)
        if not validation["valid"]:
            logger.warning("Linguistic feature validation warnings:")
            for warning in validation["warnings"]:
                logger.warning(f"  - {warning}")
        else:
            logger.info("All linguistic features validated successfully")

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

        # Frequency effects (by Zipf bins)
        if "word_frequency_zipf" in data.columns:
            # Bin into frequency categories
            freq_bins = pd.cut(
                data["word_frequency_zipf"],
                bins=[0, 3, 4, 5, 10],
                labels=["Low (≤3)", "Medium (3-4)", "High (4-5)", "Very High (>5)"],
            )

            for measure in [
                "total_reading_time",
                "gaze_duration",
                "first_fixation_duration",
            ]:
                if measure in data.columns:
                    # Only analyze fixated words
                    fixated = data[data["was_fixated"] == True]
                    freq_effects = fixated.groupby(freq_bins)[measure].agg(
                        ["mean", "std", "count"]
                    )

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
                    length_effects = data.groupby(length_bins)[measure].agg(
                        ["mean", "count"]
                    )
                else:
                    fixated = data[data["was_fixated"] == True]
                    length_effects = fixated.groupby(length_bins)[measure].agg(
                        ["mean", "std", "count"]
                    )

                results["length_effects"][measure] = {
                    str(label): {"mean": float(row["mean"]), "count": int(row["count"])}
                    for label, row in length_effects.iterrows()
                }

        # Group differences in linguistic feature effects
        if "dyslexic" in data.columns:
            results["group_differences"] = {}

            # Frequency effects by group
            for group_name, group_data in data.groupby("dyslexic"):
                group_label = "dyslexic" if group_name else "control"
                results["group_differences"][group_label] = {}

                fixated = group_data[group_data["was_fixated"] == True]

                for feature in ["word_frequency_zipf", "word_length"]:
                    if (
                        feature in fixated.columns
                        and "total_reading_time" in fixated.columns
                    ):
                        # Correlation with reading time
                        corr = (
                            fixated[[feature, "total_reading_time"]].corr().iloc[0, 1]
                        )
                        results["group_differences"][group_label][
                            f"{feature}_correlation"
                        ] = float(corr)

        logger.info("Linguistic feature analysis complete")
        return results

    def run_exploratory_analysis(self, compute_surprisal: bool = False) -> dict:
        """
        Run exploratory analysis on the word-level data

        Args:
            compute_surprisal: Whether to compute surprisal (slow!)
        """

        logger.info("Running exploratory analysis...")

        # Load data
        data = self.load_copco_data()

        # NEW: Compute linguistic features
        data = self.compute_linguistic_features(
            data, compute_surprisal=compute_surprisal
        )

        # Get skipping analysis results if available
        skipping_analysis = getattr(self, "_skipping_analysis", {})

        # Calculate basic statistics (now includes skipping stats)
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

        # NEW: Analyze linguistic features
        linguistic_analysis = self.analyze_linguistic_features(data)
        summary["linguistic_features"] = linguistic_analysis

        # Save results
        save_json_results(summary, self.results_dir / "exploratory_summary.json")

        # Create plots using abstracted visualization functions
        create_exploratory_plots(data, self.results_dir / "exploratory_plots.png")
        create_group_summary_plots(
            data, self.results_dir / "group_summary_plots.png", skipping_analysis
        )

        # NEW: Create linguistic feature plots
        self._create_linguistic_feature_plots(data)

        logger.info(f"Analysis complete. Results saved to {self.results_dir}")

        return summary

    def _create_linguistic_feature_plots(self, data: pd.DataFrame):
        """Create visualizations for linguistic features"""

        import matplotlib.pyplot as plt
        import seaborn as sns

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
        ax1.axvline(
            x=3, color="r", linestyle="--", alpha=0.5, label="Low freq threshold"
        )
        ax1.axvline(
            x=4, color="g", linestyle="--", alpha=0.5, label="High freq threshold"
        )
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

        # 4. Frequency effect on reading time
        ax4 = plt.subplot(3, 3, 4)
        freq_bins = pd.cut(fixated["word_frequency_zipf"], bins=5)
        sns.boxplot(data=fixated, x=freq_bins, y="total_reading_time", ax=ax4)
        ax4.set_xlabel("Frequency Bin (Zipf)")
        ax4.set_ylabel("Total Reading Time (ms)")
        ax4.set_title("Frequency Effect on Reading Time")
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha="right")

        # 5. Length effect on reading time
        ax5 = plt.subplot(3, 3, 5)
        length_groups = fixated.groupby("word_length")["total_reading_time"].mean()
        length_groups[length_groups.index <= 15].plot(kind="line", ax=ax5, marker="o")
        ax5.set_xlabel("Word Length")
        ax5.set_ylabel("Mean Total Reading Time (ms)")
        ax5.set_title("Length Effect on Reading Time")
        ax5.grid(True, alpha=0.3)

        # 6. Skipping by frequency
        ax6 = plt.subplot(3, 3, 6)
        freq_bins = pd.cut(data["word_frequency_zipf"], bins=5)
        skip_by_freq = data.groupby(freq_bins)["skipping_probability"].mean()
        skip_by_freq.plot(kind="bar", ax=ax6)
        ax6.set_xlabel("Frequency Bin (Zipf)")
        ax6.set_ylabel("Skipping Probability")
        ax6.set_title("Frequency Effect on Skipping")
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha="right")

        # 7. Group comparison - frequency effect
        if "dyslexic" in fixated.columns:
            ax7 = plt.subplot(3, 3, 7)
            for group_name, group_data in fixated.groupby("dyslexic"):
                label = "Dyslexic" if group_name else "Control"
                freq_bins = pd.cut(group_data["word_frequency_zipf"], bins=4)
                freq_effect = group_data.groupby(freq_bins)["total_reading_time"].mean()
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
                skip_effect = group_data.groupby(freq_bins)[
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


def main():
    """Main entry point for analysis"""

    parser = argparse.ArgumentParser(description="Dyslexia Time Cost Analysis")
    parser.add_argument(
        "--explore", action="store_true", help="Run exploratory analysis only"
    )
    parser.add_argument(
        "--surprisal",
        action="store_true",
        help="Compute surprisal (slow! requires transformers library)",
    )

    args = parser.parse_args()

    # Initialize pipeline
    try:
        pipeline = DyslexiaTimeAnalysisPipeline()
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)

    try:
        if args.explore:
            results = pipeline.run_exploratory_analysis(
                compute_surprisal=args.surprisal
            )
            print("\nExploratory Analysis Summary:")
            print(f"Data shape: {results['data_shape']}")
            print(f"Subjects: {results['subjects']}")
            if "dyslexic_subjects" in results:
                print(f"Dyslexic subjects: {results['dyslexic_subjects']}")
                print(f"Control subjects: {results['control_subjects']}")

            # Print linguistic feature summary
            if "linguistic_features" in results:
                print("\nLinguistic Features Summary:")
                for feature, stats in results["linguistic_features"][
                    "linguistic_features_summary"
                ].items():
                    print(f"  {feature}:")
                    print(f"    Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
                    print(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")

        else:
            print("Dyslexia Time Cost Analysis Pipeline")
            print("=" * 50)
            print("1. Run exploratory analysis (without surprisal)")
            print("2. Run exploratory analysis (WITH surprisal - slow!)")
            print("3. Exit")

            choice = input("\nEnter your choice (1-3): ")

            if choice == "1":
                results = pipeline.run_exploratory_analysis(compute_surprisal=False)
                print(f"Results saved to: {pipeline.results_dir}")
            elif choice == "2":
                results = pipeline.run_exploratory_analysis(compute_surprisal=True)
                print(f"Results saved to: {pipeline.results_dir}")
            else:
                print("Exiting...")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
