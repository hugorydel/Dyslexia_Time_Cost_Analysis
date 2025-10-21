"""
Table Generation - FULLY REVISED
Key changes:
1. Bootstrap CIs properly displayed for SRs
2. Added optional AMIE CIs display
3. Only CSV output (removed .txt and .tex)
4. Fixed n_amplified references to use n_significant
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


def generate_table_1_feature_effects(
    h1_results: Dict, h2_results: Dict, output_path: Path, include_amie_cis: bool = True
):
    """
    Table 1: Feature Effects & Pathway Amplification
    FULLY REVISED: Now includes both AMIE and SR bootstrap CIs

    Args:
        include_amie_cis: If True, add columns for AMIE bootstrap CIs
    """
    logger.info("Generating Table 1: Feature Effects & Pathway Amplification...")

    features = ["length", "zipf", "surprisal"]
    feature_labels = {"length": "Length", "zipf": "Zipf*", "surprisal": "Surprisal"}

    rows = []

    # Get CI data
    ci_data = h2_results.get("confidence_intervals", {})

    for feat in features:
        # Get H1 pathway results (point estimates)
        h1_feat = h1_results["features"][feat]
        ctrl_pathway = h1_feat["pathway_control"]
        dys_pathway = h1_feat["pathway_dyslexic"]

        # Get H2 SR results
        h2_feat = h2_results["slope_ratios"].get(feat, {})

        # Get CIs for each pathway and AMIEs
        feat_ci = ci_data.get(feat, {})

        # Build row
        row = {
            "Feature": feature_labels[feat],
        }

        # === AMIE SECTION ===
        row["AMIE_Control (ms)"] = f"{ctrl_pathway['delta_ert_ms']:.0f}"
        row["AMIE_Dyslexic (ms)"] = f"{dys_pathway['delta_ert_ms']:.0f}"

        # Add AMIE CIs if requested and available
        if include_amie_cis:
            # Control AMIE CI
            if "amie_control" in feat_ci:
                amie_ctrl_ci = feat_ci["amie_control"]
                row["AMIE_Control_CI"] = (
                    f"[{amie_ctrl_ci['ci_low']:.0f}, {amie_ctrl_ci['ci_high']:.0f}]"
                )
            else:
                row["AMIE_Control_CI"] = "—"

            # Dyslexic AMIE CI
            if "amie_dyslexic" in feat_ci:
                amie_dys_ci = feat_ci["amie_dyslexic"]
                row["AMIE_Dyslexic_CI"] = (
                    f"[{amie_dys_ci['ci_low']:.0f}, {amie_dys_ci['ci_high']:.0f}]"
                )
            else:
                row["AMIE_Dyslexic_CI"] = "—"

        # === SKIP PATHWAY ===
        row["Delta_p(skip) C/D"] = (
            f"{ctrl_pathway['delta_p_skip']:.3f} / {dys_pathway['delta_p_skip']:.3f}"
        )
        row["SR(skip)"] = (
            f"{h2_feat.get('sr_skip', np.nan):.2f}"
            if not np.isnan(h2_feat.get("sr_skip", np.nan))
            else "—"
        )
        row["95% CI(skip)"] = (
            f"[{feat_ci.get('skip', {}).get('ci_low', np.nan):.2f}, "
            f"{feat_ci.get('skip', {}).get('ci_high', np.nan):.2f}]"
            if "skip" in feat_ci and not np.isnan(feat_ci["skip"].get("ci_low", np.nan))
            else "—"
        )

        # === DURATION PATHWAY ===
        row["Delta_TRT (ms) C/D"] = (
            f"{ctrl_pathway['delta_trt_ms']:.0f} / {dys_pathway['delta_trt_ms']:.0f}"
        )
        row["SR(duration)"] = (
            f"{h2_feat.get('sr_duration', np.nan):.2f}"
            if not np.isnan(h2_feat.get("sr_duration", np.nan))
            else "—"
        )
        row["95% CI(duration)"] = (
            f"[{feat_ci.get('duration', {}).get('ci_low', np.nan):.2f}, "
            f"{feat_ci.get('duration', {}).get('ci_high', np.nan):.2f}]"
            if "duration" in feat_ci
            and not np.isnan(feat_ci["duration"].get("ci_low", np.nan))
            else "—"
        )

        # === ERT PATHWAY ===
        row["SR(ERT)"] = (
            f"{h2_feat.get('sr_ert', np.nan):.2f}"
            if not np.isnan(h2_feat.get("sr_ert", np.nan))
            else "—"
        )
        row["95% CI(ERT)"] = (
            f"[{feat_ci.get('ert', {}).get('ci_low', np.nan):.2f}, "
            f"{feat_ci.get('ert', {}).get('ci_high', np.nan):.2f}]"
            if "ert" in feat_ci and not np.isnan(feat_ci["ert"].get("ci_low", np.nan))
            else "—"
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save as CSV only
    df.to_csv(output_path.with_suffix(".csv"), index=False)

    logger.info(f"  Saved: {output_path.stem}.csv")
    if include_amie_cis:
        logger.info(f"    (includes AMIE bootstrap CIs)")


def generate_table_2_gap_decomposition(h3_results: Dict, output_path: Path):
    """
    Table 2: Gap Decomposition & Counterfactuals
    NOTE: No bootstrap CIs available for H3 analyses (not computed)
    """
    logger.info("Generating Table 2: Gap Decomposition & Counterfactuals...")

    shapley = h3_results["shapley_decomposition"]
    equal_ease = h3_results["equal_ease_counterfactual"]

    rows = [
        {
            "Component": "Total gap",
            "Gap (ms)": f"{h3_results['total_gap']:.1f}",
            "% of Total": "100%",
        },
        {"Component": "", "Gap (ms)": "", "% of Total": ""},
        {
            "Component": "Shapley decomposition:",
            "Gap (ms)": "",
            "% of Total": "",
        },
        {
            "Component": "  Skip contribution",
            "Gap (ms)": f"{shapley['skip_contribution']:.1f}",
            "% of Total": f"{shapley['skip_pct']:.0f}%",
        },
        {
            "Component": "  Duration contribution",
            "Gap (ms)": f"{shapley['duration_contribution']:.1f}",
            "% of Total": f"{shapley['duration_pct']:.0f}%",
        },
        {"Component": "", "Gap (ms)": "", "% of Total": ""},
        {
            "Component": "Equal-ease counterfactual:",
            "Gap (ms)": "",
            "% of Total": "",
        },
        {
            "Component": "  Dyslexic saved",
            "Gap (ms)": f"{equal_ease['dyslexic_saved']:.1f}",
            "% of Total": "—",
        },
        {
            "Component": "  Control saved",
            "Gap (ms)": f"{equal_ease['control_saved']:.1f}",
            "% of Total": "—",
        },
        {
            "Component": "  Gap shrink (ms)",
            "Gap (ms)": f"{equal_ease['gap_shrink_ms']:.1f}",
            "% of Total": "—",
        },
        {
            "Component": "  Gap shrink (%)",
            "Gap (ms)": f"{equal_ease['gap_shrink_pct']:.1f}%",
            "% of Total": "—",
        },
    ]

    df = pd.DataFrame(rows)

    # Save CSV only
    df.to_csv(output_path.with_suffix(".csv"), index=False)

    logger.info(f"  Saved: {output_path.stem}.csv")


def generate_all_tables(
    h1_results: Dict,
    h2_results: Dict,
    h3_results: Dict,
    skip_metadata: Dict,
    duration_metadata: Dict,
    output_dir: Path,
    include_amie_cis: bool = True,
):
    """
    Generate all tables (CSV only)
    FULLY REVISED: Fixed to handle n_significant correctly + AMIE CIs

    Args:
        include_amie_cis: If True, include AMIE bootstrap CIs in Table 1
    """
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING PUBLICATION TABLES")
    logger.info("=" * 60)

    output_dir.mkdir(exist_ok=True, parents=True)

    # Table 1: Feature Effects & Amplification (with AMIE CIs)
    generate_table_1_feature_effects(
        h1_results,
        h2_results,
        output_dir / "table_1_feature_effects",
        include_amie_cis=include_amie_cis,
    )

    # Table 2: Gap Decomposition
    generate_table_2_gap_decomposition(
        h3_results, output_dir / "table_2_gap_decomposition"
    )

    logger.info("\n✅ All tables generated successfully!")
    logger.info(f"Tables saved to: {output_dir}")
    logger.info("  Format: CSV only")
    if include_amie_cis:
        logger.info("  Table 1 includes AMIE bootstrap CIs")


def create_results_summary_markdown(
    h1_results: Dict, h2_results: Dict, h3_results: Dict, output_path: Path
):
    """
    Create human-readable markdown summary
    FULLY REVISED: Fixed n_amplified references + includes AMIE CIs
    """
    logger.info("Creating results summary (Markdown)...")

    # Fix reference to use n_significant
    n_significant = h2_results.get("n_significant", 0)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Analysis Results Summary\n\n")

        # H1
        f.write("## Hypothesis 1: Feature Effects\n\n")
        f.write(f"**Status:** {h1_results['status']}\n\n")
        f.write(f"{h1_results['summary']}\n\n")

        f.write("### Feature-Specific Results\n\n")
        for feat in ["length", "zipf", "surprisal"]:
            feat_data = h1_results["features"][feat]
            f.write(f"#### {feat.capitalize()}\n\n")

            ctrl_amie = feat_data["amie_control"].get("amie_ms", 0)
            dys_amie = feat_data["amie_dyslexic"].get("amie_ms", 0)

            f.write(f"- **AMIE (Control):** {ctrl_amie:.2f} ms\n")
            f.write(f"- **AMIE (Dyslexic):** {dys_amie:.2f} ms\n")

            # Add bootstrap CIs if available
            ci_data = h2_results.get("confidence_intervals", {}).get(feat, {})
            if "amie_control" in ci_data:
                ctrl_ci = ci_data["amie_control"]
                f.write(
                    f"  - *Bootstrap 95% CI:* [{ctrl_ci['ci_low']:.2f}, {ctrl_ci['ci_high']:.2f}]\n"
                )
            if "amie_dyslexic" in ci_data:
                dys_ci = ci_data["amie_dyslexic"]
                f.write(
                    f"  - *Bootstrap 95% CI:* [{dys_ci['ci_low']:.2f}, {dys_ci['ci_high']:.2f}]\n"
                )

            f.write(f"- **Expected Direction:** {feat_data['expected_direction']}\n")
            f.write(f"- **Status:** {feat_data['status']}\n")

            if feat_data.get("note"):
                f.write(f"\n*Note: {feat_data['note']}*\n")

            f.write("\n")

        # H2
        f.write("## Hypothesis 2: Dyslexic Amplification\n\n")
        f.write(f"**Status:** {h2_results['status']}\n\n")
        f.write(f"{h2_results['summary']}\n\n")

        f.write("### Slope Ratios by Feature and Pathway\n\n")
        f.write("| Feature | SR(skip) | SR(duration) | SR(ERT) |\n")
        f.write("|---------|----------|--------------|----------|\n")

        for feat in ["length", "zipf", "surprisal"]:
            sr_data = h2_results["slope_ratios"].get(feat, {})

            sr_skip = sr_data.get("sr_skip", np.nan)
            sr_dur = sr_data.get("sr_duration", np.nan)
            sr_ert = sr_data.get("sr_ert", np.nan)

            sr_skip_str = f"{sr_skip:.2f}" if not np.isnan(sr_skip) else "—"
            sr_dur_str = f"{sr_dur:.2f}" if not np.isnan(sr_dur) else "—"
            sr_ert_str = f"{sr_ert:.2f}" if not np.isnan(sr_ert) else "—"

            f.write(
                f"| {feat.capitalize()} | {sr_skip_str} | {sr_dur_str} | {sr_ert_str} |\n"
            )

        f.write("\n*SR > 1.0 indicates dyslexic amplification*\n\n")

        # Add feature status details
        f.write("### Feature-Level Decisions\n\n")
        feature_status = h2_results.get("feature_status", {})
        for feat, status_info in feature_status.items():
            f.write(f"- **{feat.capitalize()}:** {status_info['status']}")
            if status_info["pathway"]:
                f.write(f" (via {status_info['pathway']} pathway)")
            f.write("\n")
        f.write("\n")

        # H3
        f.write("## Hypothesis 3: Gap Decomposition\n\n")
        f.write(f"**Status:** {h3_results['status']}\n\n")
        f.write(f"{h3_results['summary']}\n\n")

        shapley = h3_results["shapley_decomposition"]
        equal_ease = h3_results["equal_ease_counterfactual"]

        f.write("### Shapley Decomposition\n\n")
        f.write(f"- **Total Gap:** {shapley['total_gap']:.2f} ms\n")
        f.write(
            f"- **Skip Contribution:** {shapley['skip_contribution']:.2f} ms ({shapley['skip_pct']:.0f}%)\n"
        )
        f.write(
            f"- **Duration Contribution:** {shapley['duration_contribution']:.2f} ms ({shapley['duration_pct']:.0f}%)\n\n"
        )

        f.write("### Equal-Ease Counterfactual\n\n")
        f.write(f"- **Baseline Gap:** {equal_ease['baseline_gap']:.2f} ms\n")
        f.write(
            f"- **Counterfactual Gap:** {equal_ease['counterfactual_gap']:.2f} ms\n"
        )
        f.write(
            f"- **Gap Shrink:** {equal_ease['gap_shrink_ms']:.2f} ms ({equal_ease['gap_shrink_pct']:.0f}%)\n"
        )
        f.write(f"- **Dyslexic Saved:** {equal_ease['dyslexic_saved']:.2f} ms\n")
        f.write(f"- **Control Saved:** {equal_ease['control_saved']:.2f} ms\n\n")

        # Add feature contributions if available
        if "equal_ease_feature_contributions" in h3_results:
            feat_contrib = h3_results["equal_ease_feature_contributions"]
            f.write("### Feature Contributions to Gap Reduction\n\n")
            for feat, contrib in feat_contrib["feature_contributions_ms"].items():
                pct = (
                    (contrib / feat_contrib["total_ms"] * 100)
                    if feat_contrib["total_ms"] > 0
                    else 0
                )
                f.write(f"- **{feat.capitalize()}:** {contrib:.2f} ms ({pct:.1f}%)\n")
            f.write(f"\n*Total reduction: {feat_contrib['total_ms']:.2f} ms*\n\n")

        # Overall
        f.write("## Overall Conclusion\n\n")

        f.write("The results demonstrate that:\n\n")
        f.write(
            "1. Word-level features (length, frequency, surprisal) significantly affect reading time\n"
        )
        f.write(
            f"2. {n_significant}/3 features show significant differential effects\n"
        )
        f.write(
            f"3. The dyslexic-control gap is primarily driven by {shapley['duration_pct']:.0f}% duration and {shapley['skip_pct']:.0f}% skip differences\n"
        )
        f.write(
            f"4. Text simplification (equal-ease) reduces the gap by {equal_ease['gap_shrink_pct']:.0f}%\n"
        )

    logger.info(f"  Saved: {output_path}")
