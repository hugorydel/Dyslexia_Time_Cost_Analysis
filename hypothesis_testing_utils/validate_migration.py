#!/usr/bin/env python3
"""
Validation Script - Check Code Migration
Verifies that the revised code is working correctly
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


class MigrationValidator:
    """Validates the revised code implementation"""

    def __init__(self):
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []

    def validate_data_preparation(self, data: pd.DataFrame, metadata: dict) -> bool:
        """Validate data preparation output"""
        print("\n" + "=" * 60)
        print("VALIDATING DATA PREPARATION")
        print("=" * 60)

        all_passed = True

        # Check 1: Raw zipf correlation
        if "length" in data.columns and "zipf" in data.columns:
            corr = data["length"].corr(data["zipf"])
            print(f"\n1. Zipf Correlation Check")
            print(f"   corr(length, zipf) = {corr:.3f}")

            if abs(corr) < 0.5:
                self.checks_failed.append(
                    "Zipf appears orthogonalized (correlation too low)"
                )
                print("   ❌ FAIL: Zipf appears orthogonalized!")
                print("   Expected: r ≈ -0.80 (strong negative correlation)")
                print(f"   Got: r = {corr:.3f}")
                all_passed = False
            elif abs(corr) > 0.7:
                self.checks_passed.append("Raw zipf with natural correlation")
                print("   ✅ PASS: Using raw zipf with natural correlation")
            else:
                self.warnings.append(f"Moderate zipf correlation (r={corr:.3f})")
                print(f"   ⚠️  WARNING: Correlation moderate (r={corr:.3f})")
                print("   Expected stronger correlation (r ≈ -0.80)")
        else:
            self.checks_failed.append("Missing length or zipf columns")
            print("   ❌ FAIL: Missing length or zipf columns")
            all_passed = False

        # Check 2: Pooled binning
        print(f"\n2. Pooled Binning Check")
        if "pooled_bin_edges" in metadata and "pooled_bin_weights" in metadata:
            n_bins = len(metadata["pooled_bin_weights"])
            print(f"   ✅ PASS: Pooled bins present ({n_bins} bins)")
            self.checks_passed.append("Pooled binning implemented")

            # Check bin balance
            weights = list(metadata["pooled_bin_weights"].values())
            max_weight = max(weights)
            min_weight = min(weights)

            if max_weight / min_weight > 2.0:
                self.warnings.append(
                    f"Unbalanced bins (ratio={max_weight/min_weight:.2f})"
                )
                print(f"   ⚠️  WARNING: Bins somewhat unbalanced")
                print(f"   Max/Min weight ratio: {max_weight/min_weight:.2f}")
            else:
                print(
                    f"   ✅ Bins reasonably balanced (ratio={max_weight/min_weight:.2f})"
                )
        else:
            self.checks_failed.append("Missing pooled binning metadata")
            print("   ❌ FAIL: Missing pooled_bin_edges or pooled_bin_weights")
            all_passed = False

        # Check 3: Length bins assigned to data
        print(f"\n3. Length Bin Assignment Check")
        if "length_bin" in data.columns:
            n_assigned = data["length_bin"].notna().sum()
            pct_assigned = n_assigned / len(data) * 100
            print(f"   ✅ PASS: Length bins assigned ({pct_assigned:.1f}% of data)")
            self.checks_passed.append("Length bins assigned to data")

            if pct_assigned < 95:
                self.warnings.append(
                    f"Only {pct_assigned:.1f}% of data has bins assigned"
                )
                print(f"   ⚠️  WARNING: {100-pct_assigned:.1f}% of data missing bins")
        else:
            self.checks_failed.append("Length_bin column missing")
            print("   ❌ FAIL: length_bin column not found")
            all_passed = False

        # Check 4: Orientation checks
        print(f"\n4. Feature Orientation Check")
        if "orientation_checks" in metadata:
            all_correct = True
            for feat, check in metadata["orientation_checks"].items():
                correct = check["correct_direction"]
                semantic = check["semantic"]

                if correct:
                    print(f"   ✅ {feat}: {semantic}")
                else:
                    print(f"   ❌ {feat}: {semantic} (REVERSED!)")
                    all_correct = False

            if all_correct:
                self.checks_passed.append("All feature orientations correct")
            else:
                self.checks_failed.append("Some features have reversed orientations")
                all_passed = False
        else:
            self.warnings.append("No orientation checks in metadata")
            print("   ⚠️  WARNING: No orientation checks performed")

        # Check 5: No orthogonalization flag
        print(f"\n5. Orthogonalization Flag Check")
        if "residualized" in metadata:
            if metadata["residualized"]:
                self.checks_failed.append("Residualization flag is True")
                print(
                    "   ❌ FAIL: residualized flag is True (should be False or absent)"
                )
                all_passed = False
            else:
                self.checks_passed.append("Residualization flag correctly False")
                print("   ✅ PASS: residualized flag is False")
        else:
            self.checks_passed.append("No residualization flag (good)")
            print("   ✅ PASS: No residualization flag (using raw zipf)")

        return all_passed

    def validate_gam_models(self, skip_metadata: dict, duration_metadata: dict) -> bool:
        """Validate GAM model specifications"""
        print("\n" + "=" * 60)
        print("VALIDATING GAM MODELS")
        print("=" * 60)

        all_passed = True

        # Check 1: Tensor product flag
        print(f"\n1. Tensor Product Check")
        if skip_metadata.get("has_tensor_product", False):
            self.checks_passed.append("Skip model has tensor product")
            print("   ✅ PASS: Skip model includes te(length, zipf)")
        else:
            self.checks_failed.append("Skip model missing tensor product")
            print("   ❌ FAIL: Skip model missing te(length, zipf)")
            all_passed = False

        if duration_metadata.get("has_tensor_product", False):
            self.checks_passed.append("Duration model has tensor product")
            print("   ✅ PASS: Duration model includes te(length, zipf)")
        else:
            self.checks_failed.append("Duration model missing tensor product")
            print("   ❌ FAIL: Duration model missing te(length, zipf)")
            all_passed = False

        # Check 2: Smearing factor for log-Gaussian
        print(f"\n2. Smearing Factor Check")
        if duration_metadata.get("uses_smearing_factor", False):
            self.checks_passed.append("Duration model uses smearing factor")
            print("   ✅ PASS: Smearing factor enabled for log-Gaussian model")
        else:
            self.warnings.append("No smearing factor (might be using Gamma)")
            print("   ⚠️  INFO: No smearing factor (using Gamma or not needed)")

        # Check 3: Model performance
        print(f"\n3. Model Performance Check")
        skip_auc_ctrl = skip_metadata.get("auc_control", 0)
        skip_auc_dys = skip_metadata.get("auc_dyslexic", 0)

        if skip_auc_ctrl > 0.6 and skip_auc_dys > 0.6:
            print(
                f"   ✅ Skip AUC reasonable: Control={skip_auc_ctrl:.3f}, Dyslexic={skip_auc_dys:.3f}"
            )
            self.checks_passed.append("Skip model has reasonable AUC")
        else:
            print(
                f"   ⚠️  WARNING: Low skip AUC: Control={skip_auc_ctrl:.3f}, Dyslexic={skip_auc_dys:.3f}"
            )
            self.warnings.append("Low skip model AUC")

        dur_r2_ctrl = duration_metadata.get("r2_control", 0)
        dur_r2_dys = duration_metadata.get("r2_dyslexic", 0)

        if dur_r2_ctrl > 0.1 and dur_r2_dys > 0.1:
            print(
                f"   ✅ Duration R² reasonable: Control={dur_r2_ctrl:.3f}, Dyslexic={dur_r2_dys:.3f}"
            )
            self.checks_passed.append("Duration model has reasonable R²")
        else:
            print(
                f"   ⚠️  WARNING: Low duration R²: Control={dur_r2_ctrl:.3f}, Dyslexic={dur_r2_dys:.3f}"
            )
            self.warnings.append("Low duration model R²")

        return all_passed

    def validate_h1_results(self, h1_results: dict) -> bool:
        """Validate H1 results structure"""
        print("\n" + "=" * 60)
        print("VALIDATING H1 RESULTS")
        print("=" * 60)

        all_passed = True

        # Check 1: Conditional zipf evaluation
        print(f"\n1. Conditional Zipf Evaluation Check")
        if "features" in h1_results and "zipf" in h1_results["features"]:
            zipf_results = h1_results["features"]["zipf"]

            if "amie_control" in zipf_results:
                method = zipf_results["amie_control"].get("method", "")

                if "conditional" in method:
                    self.checks_passed.append("Zipf uses conditional evaluation")
                    print(f"   ✅ PASS: Zipf uses conditional evaluation")
                    print(f"   Method: {method}")
                else:
                    self.checks_failed.append("Zipf not using conditional evaluation")
                    print(f"   ❌ FAIL: Zipf not using conditional evaluation")
                    print(f"   Method: {method}")
                    all_passed = False
            else:
                self.checks_failed.append("Missing AMIE results for zipf")
                print("   ❌ FAIL: Missing AMIE results")
                all_passed = False
        else:
            self.checks_failed.append("Missing zipf in H1 results")
            print("   ❌ FAIL: Missing zipf feature results")
            all_passed = False

        # Check 2: Pathway decomposition
        print(f"\n2. Pathway Decomposition Check")
        for feat in ["length", "zipf", "surprisal"]:
            if feat in h1_results.get("features", {}):
                feat_results = h1_results["features"][feat]

                has_pathways = (
                    "pathway_control" in feat_results
                    and "pathway_dyslexic" in feat_results
                )

                if has_pathways:
                    # Check for skip, duration, ERT deltas
                    ctrl_pathway = feat_results["pathway_control"]
                    has_all = (
                        "delta_p_skip" in ctrl_pathway
                        and "delta_trt_ms" in ctrl_pathway
                        and "delta_ert_ms" in ctrl_pathway
                        and "cohens_h" in ctrl_pathway
                    )

                    if has_all:
                        print(
                            f"   ✅ {feat}: Complete pathway decomposition + Cohen's h"
                        )
                    else:
                        print(f"   ⚠️  {feat}: Incomplete pathway decomposition")
                        self.warnings.append(f"{feat} missing some pathway metrics")
                else:
                    print(f"   ❌ {feat}: Missing pathway decomposition")
                    self.checks_failed.append(f"{feat} missing pathway results")
                    all_passed = False

        if len(self.checks_failed) == 0:
            self.checks_passed.append("All features have pathway decomposition")

        return all_passed

    def validate_h2_results(self, h2_results: dict) -> bool:
        """Validate H2 results structure"""
        print("\n" + "=" * 60)
        print("VALIDATING H2 RESULTS")
        print("=" * 60)

        all_passed = True

        # Check 1: Pathway-specific SRs
        print(f"\n1. Pathway-Specific SR Check")
        for feat in ["length", "zipf", "surprisal"]:
            if feat in h2_results.get("slope_ratios", {}):
                sr_data = h2_results["slope_ratios"][feat]

                has_all_pathways = all(
                    f"sr_{pathway}" in sr_data
                    for pathway in ["skip", "duration", "ert"]
                )

                if has_all_pathways:
                    print(f"   ✅ {feat}: All pathways (skip, duration, ERT)")
                else:
                    print(f"   ❌ {feat}: Missing pathway SRs")
                    self.checks_failed.append(f"{feat} missing pathway SRs")
                    all_passed = False
            else:
                print(f"   ❌ {feat}: No SR results")
                self.checks_failed.append(f"{feat} missing from H2 results")
                all_passed = False

        # Check 2: Bootstrap CIs
        print(f"\n2. Bootstrap Confidence Intervals Check")
        if "confidence_intervals" in h2_results:
            ci_data = h2_results["confidence_intervals"]

            # Check for SR CIs
            for pathway in ["skip", "duration", "ert"]:
                ci_key = f"sr_{pathway}"
                if ci_key in ci_data:
                    n_features_with_ci = len(ci_data[ci_key])
                    print(f"   ✅ SR({pathway}): CIs for {n_features_with_ci} features")
                else:
                    print(f"   ❌ SR({pathway}): Missing CIs")
                    self.checks_failed.append(f"Missing CIs for SR({pathway})")
                    all_passed = False

            self.checks_passed.append("Bootstrap CIs present")
        else:
            self.checks_failed.append("No bootstrap confidence intervals")
            print("   ❌ FAIL: No bootstrap CIs found")
            all_passed = False

        # Check 3: Bootstrap samples stored
        print(f"\n3. Bootstrap Samples Storage Check")
        if "bootstrap_samples" in h2_results:
            self.checks_passed.append("Bootstrap samples stored")
            print("   ✅ PASS: Bootstrap samples available for analysis")
        else:
            self.warnings.append("Bootstrap samples not stored")
            print("   ⚠️  INFO: Bootstrap samples not stored (optional)")

        return all_passed

    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        print(f"\n✅ Checks Passed: {len(self.checks_passed)}")
        for check in self.checks_passed:
            print(f"   - {check}")

        if self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"   - {warning}")

        if self.checks_failed:
            print(f"\n❌ Checks Failed: {len(self.checks_failed)}")
            for check in self.checks_failed:
                print(f"   - {check}")

        print("\n" + "=" * 60)

        if len(self.checks_failed) == 0:
            print("✅ ALL CRITICAL CHECKS PASSED!")
            print("Migration appears successful.")
            return True
        else:
            print("❌ SOME CHECKS FAILED")
            print("Please review failures above and fix issues.")
            return False


def main():
    """Run validation on existing results"""
    import json

    # Default paths
    results_dir = Path("results_v2")

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("Please run test_hypotheses_v2.py first")
        sys.exit(1)

    validator = MigrationValidator()

    # Load results
    try:
        with open(results_dir / "data_preparation_metadata.json") as f:
            data_metadata = json.load(f)

        with open(results_dir / "model_metadata.json") as f:
            model_metadata = json.load(f)

        with open(results_dir / "h1_feature_effects_v2.json") as f:
            h1_results = json.load(f)

        with open(results_dir / "h2_amplification_v2.json") as f:
            h2_results = json.load(f)

        print("✅ Loaded all result files")
    except FileNotFoundError as e:
        print(f"❌ Could not load results: {e}")
        print("Please run test_hypotheses_v2.py first")
        sys.exit(1)

    # Load data if available
    data_path = Path("input_data/preprocessed_data.csv")
    if data_path.exists():
        data = pd.read_csv(data_path)
        # Apply basic transformations
        data = data.rename(
            columns={"word_length": "length", "word_frequency_zipf": "zipf"}
        )
        print("✅ Loaded source data")
    else:
        print("⚠️  Source data not found, skipping data validation")
        data = None

    # Run validations
    data_ok = True
    if data is not None:
        data_ok = validator.validate_data_preparation(data, data_metadata)

    model_ok = validator.validate_gam_models(
        model_metadata["skip_model"], model_metadata["duration_model"]
    )

    h1_ok = validator.validate_h1_results(h1_results)
    h2_ok = validator.validate_h2_results(h2_results)

    # Print summary
    all_ok = validator.print_summary()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
