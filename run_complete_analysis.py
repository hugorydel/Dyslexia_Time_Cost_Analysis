#!/usr/bin/env python3
"""
COMPLETE Hypothesis Analysis Pipeline
"""

import argparse
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import pandas as pd

from hypothesis_testing_utils.caching_utils import load_or_recompute
from hypothesis_testing_utils.data_preparation import prepare_data_pipeline
from hypothesis_testing_utils.ert_predictor import create_ert_predictor
from hypothesis_testing_utils.gam_models import fit_gam_models
from hypothesis_testing_utils.h1_feature_effects import test_hypothesis_1
from hypothesis_testing_utils.h2_amplification import test_hypothesis_2
from hypothesis_testing_utils.h3_gap_decomposition import test_hypothesis_3
from hypothesis_testing_utils.visualization_utils import generate_all_figures
from hypothesis_testing_utils.zipf_diagnostic import run_zipf_diagnostic


def setup_logging(results_dir: Path):
    """
    Setup dual logging:
    1. hypothesis_testing_output.log (detailed, all INFO+)
    2. Console (WARNING+ only, keeps progress bars clean)
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = RotatingFileHandler(
        results_dir / "hypothesis_testing_output.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)  # Changed from WARNING to INFO for visibility
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = logging.getLogger(__name__)


class CompleteAnalysisPipeline:
    """Complete end-to-end analysis pipeline"""

    def __init__(
        self,
        results_dir: str = "results_final",
        n_bootstrap: int = 2000,
        use_cache: bool = True,
        quick_mode: bool = False,
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)

        setup_logging(self.results_dir)

        self.figures_dir = self.results_dir / "figures"

        self.quick_mode = quick_mode
        self.cache_dir = self.results_dir / (
            "cache_quick" if self.quick_mode else "cache_full"
        )

        self.figures_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

        self.n_bootstrap = n_bootstrap
        self.use_cache = use_cache

        logger.info("=" * 80)
        logger.info("COMPLETE DYSLEXIA GAM ANALYSIS PIPELINE - FULLY REVISED")
        logger.info("=" * 80)
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Bootstrap iterations (H1, H2, H3): {n_bootstrap}")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Quick mode: {quick_mode}")
        logger.info(f"Result caching: {'ENABLED' if use_cache else 'DISABLED'}")

    def run(self, data_path: str):
        """Execute complete analysis pipeline"""

        try:
            # ===== PHASE 1: DATA PREPARATION =====
            data = pd.read_csv(data_path)
            logger.info(f"Loaded data: {len(data):,} observations")

            prepared_data, metadata = prepare_data_pipeline(data)

            quartiles = metadata["quartiles"]
            bin_edges = np.array(metadata["pooled_bin_edges"])
            bin_weights = pd.Series(metadata["pooled_bin_weights"])

            self._save_json(metadata, "data_metadata.json")

            logger.info("✅ Phase 1 complete")

            # ===== DIAGNOSTIC: Zipf-Length Interaction =====
            run_zipf_diagnostic(prepared_data, n_bins=10, results_dir=self.results_dir)

            # ===== PHASE 2: MODEL FITTING (with caching) =====
            models_cache_path = self.cache_dir / (
                f"gam_models_quick{int(self.quick_mode)}_logdur1.pkl"
            )
            models_cache_existed = models_cache_path.exists()

            def compute_models():
                return fit_gam_models(
                    prepared_data, use_log_duration=True, quick_mode=self.quick_mode
                )

            logger.info(
                f"Model caching: {'ENABLED' if self.use_cache else 'DISABLED'} | "
                f"path={models_cache_path}"
            )

            skip_meta, duration_meta, gam_models = load_or_recompute(
                models_cache_path, compute_models, reuse=self.use_cache
            )

            model_cache_status = (
                "hit"
                if (self.use_cache and models_cache_existed)
                else ("write" if self.use_cache else "bypass")
            )
            logger.info(f"Model cache status: {model_cache_status}")

            # Build predictor (cheap) from cached or freshly fit models
            ert_predictor = create_ert_predictor(gam_models)

            model_metadata = {
                "skip_model": skip_meta,
                "duration_model": duration_meta,
                "cache_status": model_cache_status,
                "cache_path": str(models_cache_path),
            }
            self._save_json(model_metadata, "model_metadata.json")

            logger.info("✅ Phase 2 complete")

            # ===== PHASE 3: HYPOTHESIS TESTING =====

            # H1: Feature Effects (with p-values and CIs)
            h1_cache = self.cache_dir / f"h1_results_n{self.n_bootstrap}.pkl"

            def compute_h1():
                return test_hypothesis_1(
                    ert_predictor,
                    prepared_data,
                    quartiles,
                    bin_edges,
                    bin_weights,
                    n_bootstrap=self.n_bootstrap,
                )

            h1_results = load_or_recompute(h1_cache, compute_h1, reuse=self.use_cache)
            self._save_json(h1_results, "h1_results.json")

            # H2: Amplification (with p-values for SRs)
            h2_cache = self.cache_dir / f"h2_results_n{self.n_bootstrap}.pkl"

            def compute_h2():
                return test_hypothesis_2(
                    ert_predictor,
                    prepared_data,
                    quartiles,
                    bin_edges,
                    bin_weights,
                    n_bootstrap=self.n_bootstrap,
                )

            h2_results = load_or_recompute(h2_cache, compute_h2, reuse=self.use_cache)
            self._save_json(h2_results, "h2_results.json")

            # H3: Gap Decomposition (with p-values for all components)
            h3_cache = self.cache_dir / f"h3_results_n{self.n_bootstrap}.pkl"

            def compute_h3():
                return test_hypothesis_3(
                    ert_predictor,
                    prepared_data,
                    quartiles,
                    n_bootstrap=self.n_bootstrap,
                    bin_edges=bin_edges,
                )

            h3_results = load_or_recompute(h3_cache, compute_h3, reuse=self.use_cache)
            self._save_json(h3_results, "h3_results.json")

            logger.info("✅ Phase 3 complete")
            logger.info(
                "   All statistics include p-values (5 decimal places) and 95% CIs"
            )

            # ===== PHASE 4: FIGURE GENERATION =====
            generate_all_figures(
                ert_predictor=ert_predictor,
                data=prepared_data,
                h1_results=h1_results,
                h2_results=h2_results,
                h3_results=h3_results,
                quartiles=quartiles,
                gam_models=gam_models,
                skip_metadata=skip_meta,
                duration_metadata=duration_meta,
                output_dir=self.figures_dir,
            )

            logger.info("✅ Phase 4 complete")
            logger.info("   All figures have accompanying .json data files")

            # ===== PHASE 5: FINAL REPORT =====
            n_significant = h2_results.get("n_significant", 0)

            final_results = {
                "metadata": {
                    "n_total": len(prepared_data),
                    "n_bins": len(bin_weights),
                    "n_bootstrap": self.n_bootstrap,
                    "data_note": metadata.get("note", ""),
                    "used_cache": self.use_cache,
                    "model_cache": {
                        "enabled": self.use_cache,
                        "status": model_cache_status,
                        "path": str(models_cache_path),
                    },
                },
                "hypotheses": {
                    "h1": {
                        "status": h1_results["status"],
                        "summary": h1_results["summary"],
                    },
                    "h2": {
                        "status": h2_results["status"],
                        "summary": h2_results["summary"],
                        "n_significant": n_significant,
                        "amplified_features": h2_results.get("amplified_features", []),
                        "reduced_features": h2_results.get("reduced_features", []),
                    },
                    "h3": {
                        "status": h3_results["status"],
                        "summary": h3_results["summary"],
                    },
                },
            }

            self._save_json(final_results, "final_report.json")

            logger.info("✅ Phase 5 complete")

            # ===== COMPLETION =====
            self._print_completion_summary(final_results)

            return final_results

        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}", exc_info=True)
            raise

    def _save_json(self, data: dict, filename: str):
        """Save data to JSON with proper type conversion"""

        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict("records")
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            return obj

        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [deep_convert(v) for v in data]
            else:
                return convert(data)

        clean_data = deep_convert(data)

        output_path = self.results_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=2)

        logger.info(f"  Saved: {filename}")

    def _print_completion_summary(self, results: dict):
        """Print final completion summary"""

        logger.info("\n" + "=" * 80)
        logger.info("✅ ANALYSIS COMPLETE!")
        logger.info("=" * 80)

        logger.info("\nOUTPUTS:")
        logger.info(f"  📁 Results: {self.results_dir}/")
        logger.info(f"  📊 Figures: {self.figures_dir}/")
        logger.info(f"    - All figures include .json data files for reproducibility")
        logger.info(f"  📄 Summary: {self.results_dir}/RESULTS_SUMMARY.md")
        logger.info(f"  📝 Log: {self.results_dir}/hypothesis_testing_output.log")
        logger.info("\nNext steps:")
        logger.info("1. Review RESULTS_SUMMARY.md for comprehensive interpretation")
        logger.info("2. Check figures/ for publication-quality plots + JSON data")
        logger.info("3. Use JSON files to regenerate/restyle figures as needed")
        logger.info("4. Review hypothesis_testing_output.log for detailed output")
        logger.info("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Complete Dyslexia GAM Analysis Pipeline (FULLY REVISED)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="input_data/preprocessed_data.csv",
        help="Path to preprocessed data CSV",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Use cached results if available (models and results)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force recomputation of all results (and refit models)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_final",
        help="Directory for all output",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap iterations for H2 and H3 (default: 2000)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 100 bootstrap, 50 permutations for testing",
    )

    args = parser.parse_args()

    use_cache = args.use_cache and not args.no_cache
    n_bootstrap = 100 if args.quick else args.n_bootstrap

    if args.quick:
        print("⚡ QUICK MODE: Using 100 bootstrap + 50 permutation iterations")

    pipeline = CompleteAnalysisPipeline(
        results_dir=args.results_dir,
        n_bootstrap=n_bootstrap,
        use_cache=use_cache,
        quick_mode=args.quick,
    )

    try:
        pipeline.run(args.data_path)
        sys.exit(0)

    except Exception as e:
        print(f"❌ Fatal error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
