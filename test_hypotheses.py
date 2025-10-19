#!/usr/bin/env python3
"""
Complete GAM-Based Dyslexia Analysis Pipeline - Pure Python
All imports fixed, using pygam instead of R
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hypothesis_testing_utils.data_preparation import prepare_data_pipeline
from hypothesis_testing_utils.ert_predictor import create_ert_predictor
from hypothesis_testing_utils.gam_models import fit_gam_models
from hypothesis_testing_utils.h1_feature_effects import test_hypothesis_1
from hypothesis_testing_utils.h2_amplification import test_hypothesis_2
from hypothesis_testing_utils.h3_gap_decomposition import test_hypothesis_3
from hypothesis_testing_utils.zipf_diagnostic import run_zipf_diagnostic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("hypothesis_testing_output.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
CI_ITERATIONS = 100  # Number of bootstrap iterations for CIs


class DyslexiaGAMPipeline:
    """
    Complete GAM-based analysis pipeline
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)

        logger.info("Dyslexia GAM Analysis Pipeline initialized")

    def load_and_prepare_data(self, data_path: str) -> tuple:
        """
        Load processed data and prepare for GAM

        Args:
            data_path: Path to processed CSV

        Returns:
            (train_data, test_data, metadata) tuple
        """

        logger.info("=" * 80)
        logger.info("LOADING AND PREPARING DATA")
        logger.info("=" * 80)

        # Load data
        data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(data):,} observations")

        # Prepare for GAM analysis
        train_data, test_data, metadata = prepare_data_pipeline(data)

        # Save metadata
        self.save_results(metadata, "data_preparation_metadata.json")

        return train_data, test_data, metadata

    def fit_models(self, train_data: pd.DataFrame) -> tuple:
        """
        Fit skip and duration GAMs

        Args:
            train_data: Training data

        Returns:
            (skip_model_dict, duration_model_dict, ert_predictor) tuple
        """

        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: MODEL FITTING")
        logger.info("=" * 80)

        # Fit both GAMs - returns metadata only, not model objects
        skip_metadata, duration_metadata, gam_instance = fit_gam_models(train_data)

        # Create ERT predictor
        ert_predictor = create_ert_predictor(gam_instance)

        # Save model metadata (now safe for JSON)
        model_metadata = {
            "skip_model": skip_metadata,
            "duration_model": duration_metadata,
        }

        self.save_results(model_metadata, "model_metadata.json")

        # Return metadata and predictor
        return skip_metadata, duration_metadata, ert_predictor

    def test_hypotheses(
        self,
        ert_predictor,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        metadata: dict,
    ) -> dict:
        """
        Test all three hypotheses

        Args:
            ert_predictor: ERTPredictor instance
            train_data: Training data
            test_data: Test data
            metadata: Data metadata including quartiles

        Returns:
            Dictionary with all hypothesis test results
        """

        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: HYPOTHESIS TESTING")
        logger.info("=" * 80)

        quartiles = metadata["quartiles"]

        # Hypothesis 1: Feature effects
        h1_results = test_hypothesis_1(ert_predictor, train_data, test_data, quartiles)
        self.save_results(h1_results, "h1_feature_effects.json")

        # Hypothesis 2: Dyslexic amplification
        h2_results = test_hypothesis_2(
            ert_predictor, train_data, quartiles, n_bootstrap=CI_ITERATIONS
        )
        self.save_results(h2_results, "h2_amplification.json")

        # Hypothesis 3: Gap decomposition
        # Combine train and test for full analysis
        full_data = pd.concat([train_data, test_data], ignore_index=True)
        h3_results = test_hypothesis_3(ert_predictor, full_data, quartiles)
        self.save_results(h3_results, "h3_gap_decomposition.json")

        # Compile summary
        hypothesis_summary = {
            "h1_feature_effects": h1_results,
            "h2_amplification": h2_results,
            "h3_gap_decomposition": h3_results,
            "overall_conclusions": self._generate_conclusions(
                h1_results, h2_results, h3_results
            ),
        }

        self.save_results(hypothesis_summary, "hypothesis_testing_results.json")

        return hypothesis_summary

    def _generate_conclusions(
        self, h1_results: dict, h2_results: dict, h3_results: dict
    ) -> dict:
        """Generate overall conclusions"""

        all_confirmed = (
            h1_results["status"] == "CONFIRMED"
            and h2_results["status"] in ["CONFIRMED", "PARTIALLY CONFIRMED"]
            and h3_results["status"] in ["CONFIRMED", "STRONGLY CONFIRMED"]
        )

        n_amplified = h2_results.get("n_amplified", 0)

        # Generate interpretation
        if h3_results["status"] == "STRONGLY CONFIRMED":
            gap_explanation = "both differential sensitivity and text difficulty"
        elif h3_results["status"] == "CONFIRMED":
            gap_explanation = "either differential sensitivity or text difficulty"
        else:
            gap_explanation = "factors beyond the measured features"

        interpretation = (
            f"Results show that word-level features (length, frequency, predictability) "
            f"significantly affect reading time in expected directions. "
            f"Dyslexic readers show amplified effects for {n_amplified}/3 features, "
            f"indicating greater sensitivity to word difficulty. "
            f"The dyslexic-control reading time gap is explained by {gap_explanation}."
        )

        conclusions = {
            "all_hypotheses_supported": all_confirmed,
            "h1_status": h1_results["status"],
            "h2_status": h2_results["status"],
            "h3_status": h3_results["status"],
            "key_findings": [
                h1_results.get("summary", ""),
                h2_results.get("summary", ""),
                h3_results.get("summary", ""),
            ],
            "interpretation": interpretation,
        }

        return conclusions

    def save_results(self, results: dict, filename: str):
        """Save results to JSON"""
        output_path = self.results_dir / filename

        # Convert numpy/pandas types
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict("records")
            return obj

        # Recursive conversion
        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [deep_convert(v) for v in data]
            else:
                return convert(data)

        results_clean = deep_convert(results)

        with open(output_path, "w") as f:
            json.dump(results_clean, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def run_complete_analysis(self):
        """
        Run complete analysis pipeline

        Args:
            data_path: Path to processed data CSV
        """
        logger.info("=" * 80)
        logger.info("DYSLEXIA GAM ANALYSIS - COMPLETE PIPELINE")
        logger.info("=" * 80)

        script_dir = Path(__file__).resolve().parent
        data_path = (
            script_dir / "input_data" / "preprocessed_data.csv"
        )  # Default path; can be parameterized

        try:
            # Phase 1: Data preparation
            train_data, test_data, metadata = self.load_and_prepare_data(data_path)

            # Phase 2: Model fitting
            skip_model, duration_model, ert_predictor = self.fit_models(train_data)

            # Phase 3: Hypothesis testing
            hypothesis_results = self.test_hypotheses(
                ert_predictor, train_data, test_data, metadata
            )

            # Phase 4: Zipf diagnostic
            # Combine train and test data
            logger.info("\n" + "=" * 80)
            logger.info("PREPARING DATA FOR ZIPF DIAGNOSTIC")
            logger.info("=" * 80)

            full_data = pd.concat([train_data, test_data], ignore_index=True)
            logger.info(f"Combined dataset: {len(full_data):,} observations")
            logger.info(f"  Train: {len(train_data):,}")
            logger.info(f"  Test:  {len(test_data):,}")

            # Run comprehensive zipf diagnostic
            logger.info("\n" + "=" * 80)
            logger.info("RUNNING ZIPF FREQUENCY DIAGNOSTIC")
            logger.info("=" * 80)

            diagnostic_results = run_zipf_diagnostic(
                data=full_data,
                ert_predictor=ert_predictor,
                quartiles=metadata["quartiles"],
                output_dir=self.results_dir / "zipf_diagnostic",
            )

            # Save diagnostic results
            self.save_results(diagnostic_results, "zipf_diagnostic_results.json")

            logger.info("✓ Zipf diagnostic complete")

            # Phase 5: Generate final report
            final_report = self._compile_final_report(metadata, hypothesis_results)
            self.save_results(final_report, "final_analysis_report.json")

            # Print summary
            self._print_summary(final_report)

            logger.info("\n" + "=" * 80)
            logger.info("ANALYSIS COMPLETE!")
            logger.info(f"All results saved to {self.results_dir}")
            logger.info("=" * 80)

            return final_report

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise

    def _compile_final_report(self, metadata: dict, hypothesis_results: dict) -> dict:
        """Compile final comprehensive report"""

        return {
            "metadata": {
                "n_total_observations": metadata["n_total"],
                "n_train": metadata["n_train"],
                "n_test": metadata["n_test"],
                "balance_stats": metadata["balance_stats"],
            },
            "hypothesis_testing": {
                "h1_feature_effects": {
                    "status": hypothesis_results["h1_feature_effects"]["status"],
                    "summary": hypothesis_results["h1_feature_effects"]["summary"],
                },
                "h2_amplification": {
                    "status": hypothesis_results["h2_amplification"]["status"],
                    "summary": hypothesis_results["h2_amplification"]["summary"],
                    "n_amplified": hypothesis_results["h2_amplification"][
                        "n_amplified"
                    ],
                },
                "h3_gap_decomposition": {
                    "status": hypothesis_results["h3_gap_decomposition"]["status"],
                    "summary": hypothesis_results["h3_gap_decomposition"]["summary"],
                },
            },
            "overall_conclusions": hypothesis_results["overall_conclusions"],
        }

    def _print_summary(self, report: dict):
        """Print summary to console"""

        logger.info("\n" + "=" * 80)
        logger.info("FINAL ANALYSIS SUMMARY")
        logger.info("=" * 80)

        logger.info("\nHYPOTHESIS TESTING RESULTS:")
        logger.info(
            f"  H1 (Feature Effects): {report['hypothesis_testing']['h1_feature_effects']['status']}"
        )
        logger.info(
            f"  H2 (Amplification): {report['hypothesis_testing']['h2_amplification']['status']}"
        )
        logger.info(
            f"  H3 (Gap Decomposition): {report['hypothesis_testing']['h3_gap_decomposition']['status']}"
        )

        logger.info("\nOVERALL INTERPRETATION:")
        logger.info(f"  {report['overall_conclusions']['interpretation']}")

        logger.info("\n" + "=" * 80)


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(
        description="Dyslexia GAM Analysis Pipeline (Pure Python)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = DyslexiaGAMPipeline(results_dir=args.results_dir)

    # Run analysis
    try:
        pipeline.run_complete_analysis()
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
