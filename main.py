#!/usr/bin/env python3
"""
Main Analysis Orchestrator for Dyslexia Time Cost Analysis
Uses ExtractedFeatures as primary data source
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Import configuration
import config

# Import custom modules
from feature_extraction import FeatureExtractor
from statistical_models import DyslexiaStatisticalAnalyzer

# Import utilities
from utils.data_utils import (
    clean_word_data,
    create_additional_measures,
    create_text_data_from_words,
    identify_dyslexic_subjects,
    load_extracted_features,
    load_participant_stats,
    merge_with_participant_stats,
)
from utils.misc_utils import (
    create_output_directories,
    save_json_results,
    setup_logging,
    validate_config,
)
from utils.stats_utils import (
    calculate_basic_statistics,
    calculate_group_summary_stats,
    generate_analysis_summary,
)
from visualization_module import DyslexiaVisualizationSuite

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

        self.feature_extractor = FeatureExtractor(self.config)
        self.statistical_analyzer = DyslexiaStatisticalAnalyzer(self.config)
        self.visualization_suite = DyslexiaVisualizationSuite(self.config)

        # Create output directories
        self.results_dir = Path("results")
        self.directories = create_output_directories(self.results_dir)

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
            dyslexic_subjects = identify_dyslexic_subjects(
                data, self._participant_stats
            )
            data["dyslexic"] = data["subject_id"].isin(dyslexic_subjects)

            # Store group lists for JSON output
            self._dyslexic_subjects = sorted(list(dyslexic_subjects))
            self._control_subjects = sorted(
                [s for s in data["subject_id"].unique() if s not in dyslexic_subjects]
            )

        # Create additional measures for analysis
        data = create_additional_measures(data)

        # Enhanced skipping analysis using RawData
        logger.info("Performing skipping analysis...")
        try:
            from utils.skipping_utils import enhanced_skipping_analysis

            copco_path = Path(self.config.COPCO_PATH)
            enhanced_data, skipping_results = enhanced_skipping_analysis(
                copco_path, data
            )

            if skipping_results:
                logger.info(
                    f"Skipping analysis: {skipping_results.get('overall_skipping_rate', 0)*100:.1f}% overall rate"
                )
                self._skipping_analysis = skipping_results
                data = enhanced_data
            else:
                logger.warning("Skipping analysis failed - using basic measures")
                self._skipping_analysis = {}

        except Exception as e:
            logger.warning(f"Skipping analysis failed: {e}")
            self._skipping_analysis = {}

        return data

    def run_exploratory_analysis(self) -> dict:
        """Run exploratory analysis on the word-level data"""

        logger.info("Running exploratory analysis...")

        # Load data
        data = self.load_copco_data()

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

        # Save results
        save_json_results(summary, self.results_dir / "exploratory_summary.json")

        # Create plots
        self._create_exploratory_plots(data)
        self._create_group_summary_plots(data)

        logger.info(f"Analysis complete. Results saved to {self.results_dir}")

        return summary

    def _create_exploratory_plots(self, data: pd.DataFrame):
        """Create exploratory plots for word-level data"""

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("Word-Level Exploratory Data Analysis")

            # Total reading time distribution
            if "total_reading_time" in data.columns:
                axes[0, 0].hist(data["total_reading_time"], bins=50, alpha=0.7)
                axes[0, 0].set_title("Total Reading Time Distribution")
                axes[0, 0].set_xlabel("Duration (ms)")

            # Group comparison with normalized histograms
            if "dyslexic" in data.columns and "total_reading_time" in data.columns:
                for group, label in [(True, "Dyslexic"), (False, "Control")]:
                    group_data = data[data["dyslexic"] == group]["total_reading_time"]
                    # Create normalized histogram
                    counts, bins, patches = axes[0, 1].hist(
                        group_data, alpha=0.7, label=label, bins=30, density=True
                    )

                axes[0, 1].set_title("Reading Time by Group (Normalized)")
                axes[0, 1].set_xlabel("Duration (ms)")
                axes[0, 1].set_ylabel("Density (normalized)")
                axes[0, 1].legend()

            # Word length distribution
            if "word_length" in data.columns:
                axes[1, 0].hist(data["word_length"], bins=range(1, 16), alpha=0.7)
                axes[1, 0].set_title("Word Length Distribution")
                axes[1, 0].set_xlabel("Characters")

            # Word frequency (top 20 words)
            if "word_text" in data.columns:
                word_counts = data["word_text"].value_counts().head(20)
                word_counts.plot(kind="bar", ax=axes[1, 1])
                axes[1, 1].set_title("Top 20 Most Frequent Words")
                axes[1, 1].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.savefig(
                self.results_dir / "exploratory_plots.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        except Exception as e:
            logger.warning(f"Could not create exploratory plots: {e}")

    def _create_group_summary_plots(self, data: pd.DataFrame):
        """Create comprehensive group summary plots"""

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style for better-looking plots
            plt.style.use("default")
            sns.set_palette("Set2")

            # Create figure with subplots for different measures
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle(
                "Group Summary Statistics: Dyslexic vs Control Readers",
                fontsize=16,
                fontweight="bold",
            )

            if "dyslexic" not in data.columns:
                logger.warning(
                    "No dyslexic column found - cannot create group comparison plots"
                )
                return

            # Calculate subject-level averages for cleaner group comparisons
            subject_averages = (
                data.groupby(["subject_id", "dyslexic"])
                .agg(
                    {
                        measure: "mean"
                        for measure in [
                            "total_reading_time",
                            "first_fixation_duration",
                            "gaze_duration",
                            "n_fixations",
                            "regression_probability",
                        ]
                        if measure in data.columns
                    }
                )
                .reset_index()
            )

            plot_idx = 0

            # 1. Total Reading Time by Group
            if "total_reading_time" in subject_averages.columns:
                ax = axes[plot_idx // 2, plot_idx % 2]
                groups = ["Control", "Dyslexic"]
                means = [
                    subject_averages[subject_averages["dyslexic"] == False][
                        "total_reading_time"
                    ].mean(),
                    subject_averages[subject_averages["dyslexic"] == True][
                        "total_reading_time"
                    ].mean(),
                ]
                stds = [
                    subject_averages[subject_averages["dyslexic"] == False][
                        "total_reading_time"
                    ].std(),
                    subject_averages[subject_averages["dyslexic"] == True][
                        "total_reading_time"
                    ].std(),
                ]

                bars = ax.bar(
                    groups,
                    means,
                    yerr=stds,
                    capsize=5,
                    alpha=0.7,
                    color=["skyblue", "lightcoral"],
                )
                ax.set_title("Total Reading Time by Group", fontweight="bold")
                ax.set_ylabel("Reading Time (ms)")

                # Add value labels on bars
                for bar, mean, std in zip(bars, means, stds):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + std + 10,
                        f"{mean:.1f}±{std:.1f}",
                        ha="center",
                        va="bottom",
                    )

                plot_idx += 1

            # 2. First Fixation Duration by Group
            if "first_fixation_duration" in subject_averages.columns:
                ax = axes[plot_idx // 2, plot_idx % 2]
                groups = ["Control", "Dyslexic"]
                means = [
                    subject_averages[subject_averages["dyslexic"] == False][
                        "first_fixation_duration"
                    ].mean(),
                    subject_averages[subject_averages["dyslexic"] == True][
                        "first_fixation_duration"
                    ].mean(),
                ]
                stds = [
                    subject_averages[subject_averages["dyslexic"] == False][
                        "first_fixation_duration"
                    ].std(),
                    subject_averages[subject_averages["dyslexic"] == True][
                        "first_fixation_duration"
                    ].std(),
                ]

                bars = ax.bar(
                    groups,
                    means,
                    yerr=stds,
                    capsize=5,
                    alpha=0.7,
                    color=["lightgreen", "orange"],
                )
                ax.set_title("First Fixation Duration by Group", fontweight="bold")
                ax.set_ylabel("Duration (ms)")

                for bar, mean, std in zip(bars, means, stds):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + std + 5,
                        f"{mean:.1f}±{std:.1f}",
                        ha="center",
                        va="bottom",
                    )

                plot_idx += 1

            # 3. Gaze Duration by Group
            if "gaze_duration" in subject_averages.columns:
                ax = axes[plot_idx // 2, plot_idx % 2]
                groups = ["Control", "Dyslexic"]
                means = [
                    subject_averages[subject_averages["dyslexic"] == False][
                        "gaze_duration"
                    ].mean(),
                    subject_averages[subject_averages["dyslexic"] == True][
                        "gaze_duration"
                    ].mean(),
                ]
                stds = [
                    subject_averages[subject_averages["dyslexic"] == False][
                        "gaze_duration"
                    ].std(),
                    subject_averages[subject_averages["dyslexic"] == True][
                        "gaze_duration"
                    ].std(),
                ]

                bars = ax.bar(
                    groups,
                    means,
                    yerr=stds,
                    capsize=5,
                    alpha=0.7,
                    color=["mediumpurple", "gold"],
                )
                ax.set_title("Gaze Duration by Group", fontweight="bold")
                ax.set_ylabel("Duration (ms)")

                for bar, mean, std in zip(bars, means, stds):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + std + 5,
                        f"{mean:.1f}±{std:.1f}",
                        ha="center",
                        va="bottom",
                    )

                plot_idx += 1

            # 4. Number of Fixations by Group
            if "n_fixations" in subject_averages.columns:
                ax = axes[plot_idx // 2, plot_idx % 2]
                groups = ["Control", "Dyslexic"]
                means = [
                    subject_averages[subject_averages["dyslexic"] == False][
                        "n_fixations"
                    ].mean(),
                    subject_averages[subject_averages["dyslexic"] == True][
                        "n_fixations"
                    ].mean(),
                ]
                stds = [
                    subject_averages[subject_averages["dyslexic"] == False][
                        "n_fixations"
                    ].std(),
                    subject_averages[subject_averages["dyslexic"] == True][
                        "n_fixations"
                    ].std(),
                ]

                bars = ax.bar(
                    groups,
                    means,
                    yerr=stds,
                    capsize=5,
                    alpha=0.7,
                    color=["lightsteelblue", "lightsalmon"],
                )
                ax.set_title("Number of Fixations per Word by Group", fontweight="bold")
                ax.set_ylabel("Number of Fixations")

                for bar, mean, std in zip(bars, means, stds):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + std + 0.05,
                        f"{mean:.2f}±{std:.2f}",
                        ha="center",
                        va="bottom",
                    )

                plot_idx += 1

            # 5. Skipping Probability by Group (if enhanced analysis available)
            if (
                hasattr(self, "_skipping_analysis")
                and self._skipping_analysis
                and "subject_level_skipping_by_group" in self._skipping_analysis
            ):

                ax = axes[plot_idx // 2, plot_idx % 2]
                groups = ["Control", "Dyslexic"]

                skipping_data = self._skipping_analysis[
                    "subject_level_skipping_by_group"
                ]

                means = [
                    skipping_data.get("control", {}).get("mean", 0),
                    skipping_data.get("dyslexic", {}).get("mean", 0),
                ]
                stds = [
                    skipping_data.get("control", {}).get("std", 0),
                    skipping_data.get("dyslexic", {}).get("std", 0),
                ]

                bars = ax.bar(
                    groups,
                    means,
                    yerr=stds,
                    capsize=5,
                    alpha=0.7,
                    color=["darkseagreen", "plum"],
                )
                ax.set_title(
                    "Word Skipping Probability by Group\n(Enhanced Analysis)",
                    fontweight="bold",
                )
                ax.set_ylabel("Skipping Probability")
                ax.set_ylim(0, max(means) * 1.3 if max(means) > 0 else 0.5)

                for bar, mean, std in zip(bars, means, stds):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + std + 0.01,
                        f"{mean:.3f}±{std:.3f}",
                        ha="center",
                        va="bottom",
                    )

                plot_idx += 1

            # Alternative: Show regression probability if no skipping data available
            elif "regression_probability" in subject_averages.columns:
                ax = axes[plot_idx // 2, plot_idx % 2]
                groups = ["Control", "Dyslexic"]
                means = [
                    subject_averages[subject_averages["dyslexic"] == False][
                        "regression_probability"
                    ].mean(),
                    subject_averages[subject_averages["dyslexic"] == True][
                        "regression_probability"
                    ].mean(),
                ]
                stds = [
                    subject_averages[subject_averages["dyslexic"] == False][
                        "regression_probability"
                    ].std(),
                    subject_averages[subject_averages["dyslexic"] == True][
                        "regression_probability"
                    ].std(),
                ]

                bars = ax.bar(
                    groups,
                    means,
                    yerr=stds,
                    capsize=5,
                    alpha=0.7,
                    color=["darkseagreen", "plum"],
                )
                ax.set_title(
                    "Regression Probability by Group\n(Go-past > Gaze Duration)",
                    fontweight="bold",
                )
                ax.set_ylabel("Regression Probability")
                ax.set_ylim(0, 1)

                for bar, mean, std in zip(bars, means, stds):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + std + 0.02,
                        f"{mean:.3f}±{std:.3f}",
                        ha="center",
                        va="bottom",
                    )

                plot_idx += 1

            # 6. Word Length Effects by Group (if remaining space)
            if (
                plot_idx < 6
                and "word_length" in data.columns
                and "total_reading_time" in data.columns
            ):
                ax = axes[plot_idx // 2, plot_idx % 2]

                # Create word length bins
                data["word_length_bin"] = pd.cut(
                    data["word_length"],
                    bins=[0, 3, 5, 7, float("inf")],
                    labels=[
                        "Short (1-3)",
                        "Medium (4-5)",
                        "Long (6-7)",
                        "Very Long (8+)",
                    ],
                )

                length_group_means = (
                    data.groupby(["word_length_bin", "dyslexic"], observed=True)[
                        "total_reading_time"
                    ]
                    .mean()
                    .unstack()
                )
                length_group_stds = (
                    data.groupby(["word_length_bin", "dyslexic"], observed=True)[
                        "total_reading_time"
                    ]
                    .std()
                    .unstack()
                )

                x = np.arange(len(length_group_means.index))
                width = 0.35

                bars1 = ax.bar(
                    x - width / 2,
                    length_group_means[False],
                    width,
                    yerr=length_group_stds[False],
                    label="Control",
                    alpha=0.7,
                    capsize=3,
                )
                bars2 = ax.bar(
                    x + width / 2,
                    length_group_means[True],
                    width,
                    yerr=length_group_stds[True],
                    label="Dyslexic",
                    alpha=0.7,
                    capsize=3,
                )

                ax.set_title("Reading Time by Word Length and Group", fontweight="bold")
                ax.set_ylabel("Reading Time (ms)")
                ax.set_xlabel("Word Length")
                ax.set_xticks(x)
                ax.set_xticklabels(length_group_means.index)
                ax.legend()

                plot_idx += 1

            # Hide any unused subplots
            for i in range(plot_idx, 6):
                axes[i // 2, i % 2].set_visible(False)

            plt.tight_layout()
            plt.savefig(
                self.results_dir / "group_summary_plots.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            logger.info("Group summary plots created successfully")

        except Exception as e:
            logger.warning(f"Could not create group summary plots: {e}")

    def run_full_analysis(self, save_intermediate=True) -> dict:
        """Run the complete analysis pipeline with word-level data"""

        logger.info("=" * 60)
        logger.info("STARTING FULL DYSLEXIA TIME COST ANALYSIS")
        logger.info("=" * 60)

        results = {
            "timestamp": datetime.now().isoformat(),
            "data_summary": {},
            "features": {},
            "statistical_results": {},
            "figures": {},
            "summary": {},
        }

        try:
            # Step 1: Load word-level data
            logger.info("Step 1: Loading word-level data...")
            word_data = self.load_copco_data()
            results["data_summary"] = {
                "n_words": len(word_data),
                "n_subjects": word_data["subject_id"].nunique(),
                "n_dyslexic_subjects": word_data.groupby("subject_id")["dyslexic"]
                .first()
                .sum(),
            }

            if save_intermediate:
                word_data.to_pickle(self.results_dir / "word_data.pkl")

            # Step 2: Feature extraction
            logger.info("Step 2: Extracting features...")
            text_data = create_text_data_from_words(word_data)
            featured_data = self.feature_extractor.extract_word_features(
                word_data, text_data
            )

            if save_intermediate:
                featured_data.to_pickle(self.results_dir / "featured_data.pkl")

            # Step 3: Statistical analysis
            logger.info("Step 3: Running statistical analysis...")
            statistical_results = self.statistical_analyzer.run_comprehensive_analysis(
                featured_data
            )
            results["statistical_results"] = statistical_results

            if save_intermediate:
                self.statistical_analyzer.save_results(str(self.results_dir / "models"))

            # Step 4: Visualization
            logger.info("Step 4: Creating visualizations...")
            figure_paths = self.visualization_suite.save_all_figures(
                featured_data, statistical_results
            )
            results["figures"] = figure_paths

            # Step 5: Generate summary
            logger.info("Step 5: Generating summary...")
            results["summary"] = generate_analysis_summary(results)

            # Save complete results
            save_json_results(results, self.results_dir / "complete_results.json")

            logger.info("=" * 60)
            logger.info("ANALYSIS COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)

            return results

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise


def main():
    """Main entry point for analysis"""

    parser = argparse.ArgumentParser(description="Dyslexia Time Cost Analysis")
    parser.add_argument(
        "--explore", action="store_true", help="Run exploratory analysis only"
    )
    parser.add_argument(
        "--full-analysis", action="store_true", help="Run complete analysis pipeline"
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
            results = pipeline.run_exploratory_analysis()
            print("\nExploratory Analysis Summary:")
            print(f"Data shape: {results['data_shape']}")
            print(f"Subjects: {results['subjects']}")
            if "dyslexic_subjects" in results:
                print(f"Dyslexic subjects: {results['dyslexic_subjects']}")
                print(f"Control subjects: {results['control_subjects']}")

        elif args.full_analysis:
            results = pipeline.run_full_analysis()
            print("\nAnalysis completed successfully!")
            print(f"Results saved to: {pipeline.results_dir}")

        else:
            print("Dyslexia Time Cost Analysis Pipeline")
            print("=" * 50)
            print("1. Run exploratory analysis")
            print("2. Run full analysis")
            print("3. Exit")

            choice = input("\nEnter your choice (1-3): ")

            if choice == "1":
                results = pipeline.run_exploratory_analysis()
                print(f"Results saved to: {pipeline.results_dir}")
            elif choice == "2":
                results = pipeline.run_full_analysis()
                print(f"Results saved to: {pipeline.results_dir}")
            else:
                print("Exiting...")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
