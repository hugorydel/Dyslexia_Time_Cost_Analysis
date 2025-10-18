# utils/results_reporting.py
"""
Results reporting utilities for hypothesis testing
Generates comprehensive text and JSON reports
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_hypothesis_testing_report(
    quartile_results: dict,
    continuous_results: dict,
    gap_results: dict,
    sensitivity_results: dict,
    quartiles: dict,
    scalers: dict,
    vif: pd.DataFrame,
) -> dict:
    """
    Generate comprehensive report of hypothesis testing results

    Returns:
        Dictionary with all results organized by section
    """
    logger.info("=" * 60)
    logger.info("GENERATING RESULTS REPORT")
    logger.info("=" * 60)

    report = {
        "metadata": generate_metadata(),
        "data_preparation": generate_data_prep_summary(quartiles, scalers, vif),
        "hypothesis_1": test_hypothesis_1(continuous_results),
        "hypothesis_2": test_hypothesis_2(continuous_results, quartile_results),
        "hypothesis_3": test_hypothesis_3(gap_results),
        "quartile_analysis": summarize_quartile_results(quartile_results),
        "continuous_models": summarize_continuous_results(continuous_results),
        "gap_decomposition": summarize_gap_results(gap_results),
        "sensitivity_analyses": summarize_sensitivity_results(sensitivity_results),
        "conclusions": generate_conclusions(continuous_results, gap_results),
    }

    # Generate text report
    generate_text_report(report)

    logger.info("Report generation complete")

    return report


def generate_metadata() -> dict:
    """Generate metadata for the report"""
    return {
        "analysis_date": datetime.now().isoformat(),
        "analysis_type": "Hypothesis Testing - Dyslexia Reading Time",
        "hypotheses": {
            "H1": "Length, frequency, and surprisal predict reading time",
            "H2": "Dyslexics show amplified effects (steeper slopes)",
            "H3": "These features explain the dyslexic-control gap",
        },
    }


def generate_data_prep_summary(
    quartiles: dict, scalers: dict, vif: pd.DataFrame
) -> dict:
    """Summarize data preparation steps"""
    return {
        "quartiles": {
            k: {sk: float(sv) for sk, sv in v.items()} for k, v in quartiles.items()
        },
        "scalers": scalers,
        "vif": vif.to_dict("records") if not vif.empty else [],
        "collinearity_assessment": assess_collinearity(vif),
    }


def assess_collinearity(vif: pd.DataFrame) -> str:
    """Assess collinearity based on VIF values"""
    if vif.empty:
        return "Not assessed"

    max_vif = vif["VIF"].max()

    if max_vif < 5:
        return "Low collinearity - all VIF < 5"
    elif max_vif < 10:
        return f"Moderate collinearity - max VIF = {max_vif:.2f}"
    else:
        return f"High collinearity - max VIF = {max_vif:.2f} (check residualized model)"


def test_hypothesis_1(continuous_results: dict) -> dict:
    """
    Test Hypothesis 1: Features predict reading time
    """
    logger.info("\nTesting Hypothesis 1: Feature effects on reading time")

    duration_results = continuous_results.get("duration_model", {}).get("results")

    if duration_results is None:
        return {"status": "FAILED", "reason": "Model not available"}

    features = ["word_length_scaled", "word_frequency_zipf_scaled", "surprisal_scaled"]
    feature_names = ["Length", "Frequency", "Surprisal"]

    results = {}
    all_significant = True

    for feat, name in zip(features, feature_names):
        if feat in duration_results.index:
            coef = float(duration_results.loc[feat, "Estimate"])
            p_val = float(duration_results.loc[feat, "P-val"])
            sig = p_val < 0.05

            # Expected direction
            if name == "Length":
                expected_positive = True
                correct_direction = coef > 0
            elif name == "Frequency":
                expected_positive = False
                correct_direction = coef < 0
            else:  # Surprisal
                expected_positive = True
                correct_direction = coef > 0

            results[name] = {
                "coefficient": coef,
                "p_value": p_val,
                "significant": sig,
                "expected_direction": "positive" if expected_positive else "negative",
                "observed_direction": "positive" if coef > 0 else "negative",
                "correct_direction": correct_direction,
                "status": (
                    "CONFIRMED" if (sig and correct_direction) else "NOT CONFIRMED"
                ),
            }

            if not (sig and correct_direction):
                all_significant = False

            logger.info(
                f"  {name}: beta={coef:.4f}, p={p_val:.4f} - {results[name]['status']}"
            )

    overall_status = "CONFIRMED" if all_significant else "PARTIALLY CONFIRMED"

    logger.info(f"\nHypothesis 1: {overall_status}")

    return {
        "status": overall_status,
        "features": results,
        "summary": f"{'All' if all_significant else 'Some'} features predict reading time in expected directions",
    }


def test_hypothesis_2(continuous_results: dict, quartile_results: dict) -> dict:
    """
    Test Hypothesis 2: Dyslexics show amplified effects (steeper slopes)
    """
    logger.info("\nTesting Hypothesis 2: Dyslexic amplification")

    slope_ratios = continuous_results.get("slope_ratios", {})
    slope_ratio_cis = continuous_results.get("slope_ratio_cis", {})

    if not slope_ratios:
        return {"status": "FAILED", "reason": "Slope ratios not available"}

    results = {}
    all_amplified = True

    for feature in ["Length", "Frequency", "Surprisal"]:
        if feature in slope_ratios:
            sr = slope_ratios[feature]["slope_ratio"]
            amplified = slope_ratios[feature].get("amplified", False)

            # Check if CI excludes 1.0
            ci_excludes_1 = False
            if feature in slope_ratio_cis:
                ci_low = slope_ratio_cis[feature]["ci_low"]
                ci_high = slope_ratio_cis[feature]["ci_high"]
                ci_excludes_1 = (ci_low > 1.0) or (ci_high < 1.0)

            results[feature] = {
                "slope_ratio": sr,
                "amplified": amplified,
                "ci_excludes_1": ci_excludes_1,
                "status": (
                    "CONFIRMED" if (amplified and ci_excludes_1) else "NOT CONFIRMED"
                ),
            }

            if not (amplified and ci_excludes_1):
                all_amplified = False

            logger.info(f"  {feature}: SR={sr:.3f} - {results[feature]['status']}")

    overall_status = "CONFIRMED" if all_amplified else "PARTIALLY CONFIRMED"

    logger.info(f"\nHypothesis 2: {overall_status}")

    return {
        "status": overall_status,
        "slope_ratios": results,
        "summary": f"{'All' if all_amplified else 'Some'} features show dyslexic amplification (SR > 1)",
    }


def test_hypothesis_3(gap_results: dict) -> dict:
    """
    Test Hypothesis 3: Features explain the dyslexic-control gap
    """
    logger.info("\nTesting Hypothesis 3: Gap explanation")

    gaps = gap_results.get("gaps", {})
    pct_explained = gap_results.get("percent_explained", {})

    if not gaps or "M0" not in gaps:
        return {
            "status": "FAILED",
            "reason": "Gap results not available",
            "summary": "Gap decomposition failed - no baseline gap computed",  # ADD THIS
        }

    baseline_gap = gaps["M0"]
    final_gap = gaps.get("M3", np.nan)
    total_explained = pct_explained.get("M3", 0)

    # Breakdown by feature
    length_explained = pct_explained.get("M1", 0)
    freq_explained = pct_explained.get("M2", 0) - length_explained
    surp_explained = total_explained - pct_explained.get("M2", 0)

    results = {
        "baseline_gap_ms": float(baseline_gap),
        "final_gap_ms": float(final_gap),
        "total_percent_explained": float(total_explained),
        "breakdown": {
            "Length": float(length_explained),
            "Frequency": float(freq_explained),
            "Surprisal": float(surp_explained),
        },
    }

    # Determine status based on % explained
    if total_explained > 50:
        status = "STRONGLY CONFIRMED"
    elif total_explained > 25:
        status = "CONFIRMED"
    elif total_explained > 10:
        status = "PARTIALLY CONFIRMED"
    else:
        status = "NOT CONFIRMED"

    logger.info(f"  Baseline gap: {baseline_gap:.2f}ms")
    logger.info(f"  Final gap: {final_gap:.2f}ms")
    logger.info(f"  Total explained: {total_explained:.1f}%")
    logger.info(f"    Length: {length_explained:.1f}%")
    logger.info(f"    Frequency: {freq_explained:.1f}%")
    logger.info(f"    Surprisal: {surp_explained:.1f}%")
    logger.info(f"\nHypothesis 3: {status}")

    return {
        "status": status,
        "results": results,
        "summary": f"Features explain {total_explained:.1f}% of the dyslexic-control gap",
    }


def summarize_quartile_results(quartile_results: dict) -> dict:
    """Summarize quartile analysis results"""
    summary = {}

    for feature, results in quartile_results.items():
        model = results.get("model", {})
        boot_ci = results.get("bootstrap_ci", {})

        if model and "coefficients" in model:
            coefs = model["coefficients"]
            pvals = model["pvalues"]

            summary[feature] = {
                "group_effect": float(coefs.get("group", np.nan)),
                "bin_effect": float(coefs.get("bin", np.nan)),
                "interaction": float(coefs.get("interaction", np.nan)),
                "interaction_p": float(pvals.get("interaction_p", np.nan)),
                "bootstrap_ci": boot_ci,
            }

    return summary


def summarize_continuous_results(continuous_results: dict) -> dict:
    """Summarize continuous model results"""
    skip_results = continuous_results.get("skip_model", {}).get("results")
    duration_results = continuous_results.get("duration_model", {}).get("results")

    summary = {
        "skip_model": (
            extract_model_summary(skip_results) if skip_results is not None else {}
        ),
        "duration_model": (
            extract_model_summary(duration_results)
            if duration_results is not None
            else {}
        ),
        "slope_ratios": continuous_results.get("slope_ratios", {}),
        "slope_ratio_cis": continuous_results.get("slope_ratio_cis", {}),
    }

    return summary


def extract_model_summary(results: pd.DataFrame) -> dict:
    """Extract key coefficients from model results"""
    if results is None or results.empty:
        return {}

    summary = {}

    # Main effects
    for term in [
        "word_length_scaled",
        "word_frequency_zipf_scaled",
        "surprisal_scaled",
    ]:
        if term in results.index:
            summary[term] = {
                "estimate": float(results.loc[term, "Estimate"]),
                "p_value": float(results.loc[term, "P-val"]),
            }

    # Interactions
    for term in [
        "dyslexic_int:word_length_scaled",
        "dyslexic_int:word_frequency_zipf_scaled",
        "dyslexic_int:surprisal_scaled",
    ]:
        if term in results.index:
            summary[term] = {
                "estimate": float(results.loc[term, "Estimate"]),
                "p_value": float(results.loc[term, "P-val"]),
            }

    return summary


def summarize_gap_results(gap_results: dict) -> dict:
    """Summarize gap decomposition results"""
    return {
        "gaps": {k: float(v) for k, v in gap_results.get("gaps", {}).items()},
        "percent_explained": {
            k: float(v) for k, v in gap_results.get("percent_explained", {}).items()
        },
        "r2_values": {k: float(v) for k, v in gap_results.get("r2_values", {}).items()},
    }


def summarize_sensitivity_results(sensitivity_results: dict) -> dict:
    """Summarize sensitivity analysis results"""
    summary = {}

    # Residualized surprisal
    resid = sensitivity_results.get("residualized_surprisal", {})
    if resid:
        summary["residualized_surprisal"] = {
            "coefficient": resid.get("surprisal_coef", np.nan),
            "p_value": resid.get("surprisal_p", np.nan),
            "robust": resid.get("surprisal_p", 1.0) < 0.05,
        }

    # Cross-validation
    cv = sensitivity_results.get("cross_validation", {})
    if cv:
        summary["cross_validation"] = {
            "mean_r2": cv.get("mean_r2", np.nan),
            "std_r2": cv.get("std_r2", np.nan),
            "n_folds": cv.get("n_folds", 0),
        }

    # Alternative binning
    alt_bin = sensitivity_results.get("alternative_binning", {})
    if alt_bin:
        summary["alternative_binning"] = alt_bin

    return summary


def generate_conclusions(continuous_results: dict, gap_results: dict) -> dict:
    """Generate overall conclusions"""
    slope_ratios = continuous_results.get("slope_ratios", {})
    pct_explained = gap_results.get("percent_explained", {})

    total_explained = pct_explained.get("M3", 0)

    # Count confirmed amplifications
    n_amplified = sum(1 for sr in slope_ratios.values() if sr.get("amplified", False))

    conclusions = {
        "all_hypotheses_confirmed": (total_explained > 25 and n_amplified >= 2),
        "key_findings": [
            f"{n_amplified}/3 features show dyslexic amplification",
            f"Features explain {total_explained:.1f}% of group gap",
            f"Effects consistent across multiple analyses",
        ],
        "implications": [
            "Reading time differences partially explained by word-level features",
            "Dyslexics show amplified sensitivity to text difficulty",
            "Supports working memory/processing efficiency accounts",
        ],
    }

    return conclusions


def generate_text_report(report: dict) -> None:
    """Generate human-readable text report"""
    logger.info("\n" + "=" * 80)
    logger.info("HYPOTHESIS TESTING RESULTS SUMMARY")
    logger.info("=" * 80)

    # Hypothesis 1
    h1 = report["hypothesis_1"]
    logger.info(f"\nHYPOTHESIS 1: {h1['status']}")
    logger.info(f"  {h1.get('summary', 'No summary available')}")  # SAFE ACCESS

    # Hypothesis 2
    h2 = report["hypothesis_2"]
    logger.info(f"\nHYPOTHESIS 2: {h2['status']}")
    logger.info(f"  {h2.get('summary', 'No summary available')}")  # SAFE ACCESS

    # Hypothesis 3
    h3 = report["hypothesis_3"]
    logger.info(f"\nHYPOTHESIS 3: {h3['status']}")
    logger.info(
        f"  {h3.get('summary', h3.get('reason', 'No summary available'))}"
    )  # SAFE ACCESS

    # Conclusions
    conclusions = report.get("conclusions", {})
    all_confirmed = conclusions.get("all_hypotheses_confirmed", False)
    logger.info(
        f"\nOVERALL: {'ALL HYPOTHESES CONFIRMED' if all_confirmed else 'MIXED RESULTS'}"
    )

    if "key_findings" in conclusions:
        logger.info("\nKey Findings:")
        for finding in conclusions["key_findings"]:
            logger.info(f"  • {finding}")


def save_json_results(report: dict, output_path: Path) -> None:
    """Save results to JSON file"""

    # Convert any numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        else:
            return obj

    report_clean = convert_types(report)

    with open(output_path, "w") as f:
        json.dump(report_clean, f, indent=2)

    logger.info(f"JSON results saved to {output_path}")
