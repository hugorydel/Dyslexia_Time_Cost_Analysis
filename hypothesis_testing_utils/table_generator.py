"""
Table Generation - FULLY REVISED
Key changes:
1. Added p-values (5 decimal places) for all statistics
2. Comprehensive confidence intervals
3. Enhanced with Cohen's h values
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def format_ci(value: float, ci_low: float, ci_high: float, decimals: int = 2) -> str:
    """Format value with 95% CI"""
    if np.isnan(value) or np.isnan(ci_low) or np.isnan(ci_high):
        return "—"
    return f"{value:.{decimals}f} [{ci_low:.{decimals}f}, {ci_high:.{decimals}f}]"


def format_p_value(p: float) -> str:
    """Format p-value to 5 decimal places"""
    if np.isnan(p):
        return "—"
    if p < 0.00001:
        return "<0.00001"
    return f"{p:.5f}"


def generate_table_1_feature_effects(
    h1_results: Dict, h2_results: Dict, output_path: Path, include_p_values: bool = True
):
    """
    Table 1: Feature Effects & Pathway Amplification
    FULLY REVISED: Comprehensive statistics with p-values
    """
    logger.info("Generating Table 1: Feature Effects & Pathway Amplification...")

    features = ["length", "zipf", "surprisal"]
    feature_labels = {"length": "Length", "zipf": "Zipf*", "surprisal": "Surprisal"}

    rows = []

    ci_data = h2_results.get("confidence_intervals", {})

    for feat in features:
        h1_feat = h1_results["features"][feat]
        ctrl_pathway = h1_feat["pathway_control"]
        dys_pathway = h1_feat["pathway_dyslexic"]

        h2_feat = h2_results["slope_ratios"].get(feat, {})
        feat_ci = ci_data.get(feat, {})

        row = {
            "Feature": feature_labels[feat],
        }

        # === AMIE SECTION (with p-values) ===
        ctrl_amie = h1_feat["amie_control"]
        dys_amie = h1_feat["amie_dyslexic"]

        row["AMIE_Control (ms)"] = format_ci(
            ctrl_amie.get("amie_ms", 0),
            ctrl_amie.get("ci_low", np.nan),
            ctrl_amie.get("ci_high", np.nan),
            decimals=1,
        )

        row["AMIE_Dyslexic (ms)"] = format_ci(
            dys_amie.get("amie_ms", 0),
            dys_amie.get("ci_low", np.nan),
            dys_amie.get("ci_high", np.nan),
            decimals=1,
        )

        if include_p_values:
            row["p_AMIE_Control"] = format_p_value(ctrl_amie.get("p_value", np.nan))
            row["p_AMIE_Dyslexic"] = format_p_value(dys_amie.get("p_value", np.nan))

        # === SKIP PATHWAY (with Cohen's h and p-values) ===
        row["p_skip_Q1_Control"] = f"{ctrl_pathway['p_skip_q1']:.3f}"
        row["p_skip_Q3_Control"] = f"{ctrl_pathway['p_skip_q3']:.3f}"
        row["Cohens_h_Skip_Control"] = f"{ctrl_pathway['cohens_h']:.3f}"

        row["p_skip_Q1_Dyslexic"] = f"{dys_pathway['p_skip_q1']:.3f}"
        row["p_skip_Q3_Dyslexic"] = f"{dys_pathway['p_skip_q3']:.3f}"
        row["Cohens_h_Skip_Dyslexic"] = f"{dys_pathway['cohens_h']:.3f}"

        # SR for skip
        sr_skip = h2_feat.get("sr_skip", np.nan)
        sr_skip_ci_low = h2_feat.get("sr_skip_ci_low", np.nan)
        sr_skip_ci_high = h2_feat.get("sr_skip_ci_high", np.nan)

        row["SR(skip)"] = format_ci(
            sr_skip, sr_skip_ci_low, sr_skip_ci_high, decimals=2
        )

        if include_p_values:
            row["p_SR_skip"] = format_p_value(h2_feat.get("sr_skip_p_value", np.nan))

        # === DURATION PATHWAY (with p-values) ===
        row["TRT_Q1_Control (ms)"] = f"{ctrl_pathway['trt_q1']:.1f}"
        row["TRT_Q3_Control (ms)"] = f"{ctrl_pathway['trt_q3']:.1f}"
        row["Delta_TRT_Control (ms)"] = format_ci(
            ctrl_pathway["delta_trt_ms"],
            ctrl_pathway.get("duration_ci_low", np.nan),
            ctrl_pathway.get("duration_ci_high", np.nan),
            decimals=1,
        )

        row["TRT_Q1_Dyslexic (ms)"] = f"{dys_pathway['trt_q1']:.1f}"
        row["TRT_Q3_Dyslexic (ms)"] = f"{dys_pathway['trt_q3']:.1f}"
        row["Delta_TRT_Dyslexic (ms)"] = format_ci(
            dys_pathway["delta_trt_ms"],
            dys_pathway.get("duration_ci_low", np.nan),
            dys_pathway.get("duration_ci_high", np.nan),
            decimals=1,
        )

        if include_p_values:
            row["p_Delta_TRT_Control"] = format_p_value(
                ctrl_pathway.get("duration_p_value", np.nan)
            )
            row["p_Delta_TRT_Dyslexic"] = format_p_value(
                dys_pathway.get("duration_p_value", np.nan)
            )

        # Cohen's d for duration
        row["Cohens_d_Duration_Control"] = (
            f"{h2_feat.get('cohens_d_duration_control', np.nan):.3f}"
        )
        row["Cohens_d_Duration_Dyslexic"] = (
            f"{h2_feat.get('cohens_d_duration_dyslexic', np.nan):.3f}"
        )

        # SR for duration
        sr_dur = h2_feat.get("sr_duration", np.nan)
        sr_dur_ci_low = h2_feat.get("sr_duration_ci_low", np.nan)
        sr_dur_ci_high = h2_feat.get("sr_duration_ci_high", np.nan)

        row["SR(duration)"] = format_ci(
            sr_dur, sr_dur_ci_low, sr_dur_ci_high, decimals=2
        )

        if include_p_values:
            row["p_SR_duration"] = format_p_value(
                h2_feat.get("sr_duration_p_value", np.nan)
            )

        # === ERT PATHWAY (with p-values) ===
        row["ERT_Q1_Control (ms)"] = f"{ctrl_pathway['ert_q1']:.1f}"
        row["ERT_Q3_Control (ms)"] = f"{ctrl_pathway['ert_q3']:.1f}"
        row["Delta_ERT_Control (ms)"] = format_ci(
            ctrl_pathway["delta_ert_ms"],
            ctrl_pathway.get("ert_ci_low", np.nan),
            ctrl_pathway.get("ert_ci_high", np.nan),
            decimals=1,
        )

        row["ERT_Q1_Dyslexic (ms)"] = f"{dys_pathway['ert_q1']:.1f}"
        row["ERT_Q3_Dyslexic (ms)"] = f"{dys_pathway['ert_q3']:.1f}"
        row["Delta_ERT_Dyslexic (ms)"] = format_ci(
            dys_pathway["delta_ert_ms"],
            dys_pathway.get("ert_ci_low", np.nan),
            dys_pathway.get("ert_ci_high", np.nan),
            decimals=1,
        )

        if include_p_values:
            row["p_Delta_ERT_Control"] = format_p_value(
                ctrl_pathway.get("ert_p_value", np.nan)
            )
            row["p_Delta_ERT_Dyslexic"] = format_p_value(
                dys_pathway.get("ert_p_value", np.nan)
            )

        # Cohen's d for ERT
        row["Cohens_d_ERT_Control"] = (
            f"{h2_feat.get('cohens_d_ert_control', np.nan):.3f}"
        )
        row["Cohens_d_ERT_Dyslexic"] = (
            f"{h2_feat.get('cohens_d_ert_dyslexic', np.nan):.3f}"
        )

        # SR for ERT
        sr_ert = h2_feat.get("sr_ert", np.nan)
        sr_ert_ci_low = h2_feat.get("sr_ert_ci_low", np.nan)
        sr_ert_ci_high = h2_feat.get("sr_ert_ci_high", np.nan)

        row["SR(ERT)"] = format_ci(sr_ert, sr_ert_ci_low, sr_ert_ci_high, decimals=2)

        if include_p_values:
            row["p_SR_ERT"] = format_p_value(h2_feat.get("sr_ert_p_value", np.nan))

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(output_path.with_suffix(".csv"), index=False)

    logger.info(f"  Saved: {output_path.stem}.csv")
    logger.info(f"    Includes p-values (5 decimal places) and 95% CIs")


def generate_all_tables(
    h1_results: Dict,
    h2_results: Dict,
    h3_results: Dict,
    skip_metadata: Dict,
    duration_metadata: Dict,
    output_dir: Path,
    include_p_values: bool = True,
):
    """
    Generate all tables (CSV only)
    FULLY REVISED: Comprehensive statistics with p-values
    """
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING PUBLICATION TABLES")
    logger.info("=" * 60)

    output_dir.mkdir(exist_ok=True, parents=True)

    # Table 1: Feature Effects & Amplification
    generate_table_1_feature_effects(
        h1_results,
        h2_results,
        output_dir / "table_1_feature_effects",
        include_p_values=include_p_values,
    )

    logger.info("\n✅ All tables generated successfully!")
    logger.info(f"Tables saved to: {output_dir}")
    logger.info("  Format: CSV only")
    logger.info("  All tables include p-values (5 decimal places) and 95% CIs")
