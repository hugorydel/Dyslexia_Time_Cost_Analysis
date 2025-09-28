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
from utils.visualization_utils import (
    create_exploratory_plots,
    create_group_summary_plots,
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

        # Create plots using abstracted visualization functions
        create_exploratory_plots(data, self.results_dir / "exploratory_plots.png")
        create_group_summary_plots(
            data, self.results_dir / "group_summary_plots.png", skipping_analysis
        )

        logger.info(f"Analysis complete. Results saved to {self.results_dir}")

        return summary

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
