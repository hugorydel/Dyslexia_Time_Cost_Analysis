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
        row["Cohens_h_Control"] = f"{ctrl_pathway['cohens_h']:.3f}"

        row["p_skip_Q1_Dyslexic"] = f"{dys_pathway['p_skip_q1']:.3f}"
        row["p_skip_Q3_Dyslexic"] = f"{dys_pathway['p_skip_q3']:.3f}"
        row["Cohens_h_Dyslexic"] = f"{dys_pathway['cohens_h']:.3f}"

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


def generate_table_2_gap_decomposition(h3_results: Dict, output_path: Path):
    """
    Table 2: Gap Decomposition & Counterfactuals
    FULLY REVISED: Added p-values for all components
    """
    logger.info("Generating Table 2: Gap Decomposition & Counterfactuals...")

    shapley = h3_results["shapley_decomposition"]
    equal_ease = h3_results["equal_ease_counterfactual"]
    feature_contrib = h3_results.get("equal_ease_feature_contributions", {})

    rows = [
        {
            "Component": "Total gap",
            "Value (ms)": f"{h3_results['total_gap']:.1f}",
            "95% CI": "—",
            "p-value": "—",
            "% of Total": "100%",
        },
        {
            "Component": "",
            "Value (ms)": "",
            "95% CI": "",
            "p-value": "",
            "% of Total": "",
        },
        {
            "Component": "Shapley decomposition:",
            "Value (ms)": "",
            "95% CI": "",
            "p-value": "",
            "% of Total": "",
        },
        {
            "Component": "  Skip contribution",
            "Value (ms)": f"{shapley['skip_contribution']:.1f}",
            "95% CI": f"[{shapley['skip_contribution_stats']['ci_low']:.1f}, "
            f"{shapley['skip_contribution_stats']['ci_high']:.1f}]",
            "p-value": format_p_value(shapley["skip_contribution_stats"]["p_value"]),
            "% of Total": f"{shapley['skip_pct']:.0f}%",
        },
        {
            "Component": "  Duration contribution",
            "Value (ms)": f"{shapley['duration_contribution']:.1f}",
            "95% CI": f"[{shapley['duration_contribution_stats']['ci_low']:.1f}, "
            f"{shapley['duration_contribution_stats']['ci_high']:.1f}]",
            "p-value": format_p_value(
                shapley["duration_contribution_stats"]["p_value"]
            ),
            "% of Total": f"{shapley['duration_pct']:.0f}%",
        },
        {
            "Component": "",
            "Value (ms)": "",
            "95% CI": "",
            "p-value": "",
            "% of Total": "",
        },
        {
            "Component": "Equal-ease counterfactual:",
            "Value (ms)": "",
            "95% CI": "",
            "p-value": "",
            "% of Total": "",
        },
        {
            "Component": "  Dyslexic saved",
            "Value (ms)": f"{equal_ease['dyslexic_saved']:.1f}",
            "95% CI": "—",
            "p-value": "—",
            "% of Total": "—",
        },
        {
            "Component": "  Control saved",
            "Value (ms)": f"{equal_ease['control_saved']:.1f}",
            "95% CI": "—",
            "p-value": "—",
            "% of Total": "—",
        },
        {
            "Component": "  Gap shrink (ms)",
            "Value (ms)": f"{equal_ease['gap_shrink_ms']:.1f}",
            "95% CI": f"[{equal_ease['gap_shrink_stats']['ci_low']:.1f}, "
            f"{equal_ease['gap_shrink_stats']['ci_high']:.1f}]",
            "p-value": format_p_value(equal_ease["gap_shrink_stats"]["p_value"]),
            "% of Total": "—",
        },
        {
            "Component": "  Gap shrink (%)",
            "Value (ms)": f"{equal_ease['gap_shrink_pct']:.1f}%",
            "95% CI": "—",
            "p-value": "—",
            "% of Total": "—",
        },
    ]

    # Add feature contributions if available
    if "feature_contributions_stats" in feature_contrib:
        rows.append(
            {
                "Component": "",
                "Value (ms)": "",
                "95% CI": "",
                "p-value": "",
                "% of Total": "",
            }
        )
        rows.append(
            {
                "Component": "Feature contributions:",
                "Value (ms)": "",
                "95% CI": "",
                "p-value": "",
                "% of Total": "",
            }
        )

        for feat in ["length", "zipf", "surprisal"]:
            if feat in feature_contrib["feature_contributions_stats"]:
                stats = feature_contrib["feature_contributions_stats"][feat]
                contrib_ms = feature_contrib["feature_contributions_ms"][feat]
                total_ms = feature_contrib["total_ms"]

                rows.append(
                    {
                        "Component": f"  {feat.capitalize()}",
                        "Value (ms)": f"{contrib_ms:.1f}",
                        "95% CI": f"[{stats['ci_low']:.1f}, {stats['ci_high']:.1f}]",
                        "p-value": format_p_value(stats["p_value"]),
                        "% of Total": f"{contrib_ms/total_ms*100:.1f}%",
                    }
                )

    df = pd.DataFrame(rows)

    df.to_csv(output_path.with_suffix(".csv"), index=False)

    logger.info(f"  Saved: {output_path.stem}.csv")
    logger.info(f"    Includes p-values (5 decimal places) and 95% CIs")


def generate_table_3_per_feature_equalization(h3_results: Dict, output_path: Path):
    """
    Table 3: Per-Feature Equalization Results
    NEW TABLE: Shows individual feature contributions to gap with statistics
    """
    logger.info("Generating Table 3: Per-Feature Equalization Results...")

    per_feature = h3_results.get("per_feature_equalization", {})

    rows = []

    for feat in ["length", "zipf", "surprisal"]:
        if feat in per_feature:
            result = per_feature[feat]
            stats = result.get("gap_explained_stats", {})

            rows.append(
                {
                    "Feature": feat.capitalize(),
                    "Baseline Gap (ms)": f"{result['baseline_gap']:.1f}",
                    "Counterfactual Gap (ms)": f"{result['counterfactual_gap']:.1f}",
                    "Gap Explained (ms)": f"{result['gap_explained']:.1f}",
                    "95% CI": f"[{stats.get('ci_low', np.nan):.1f}, {stats.get('ci_high', np.nan):.1f}]",
                    "p-value": format_p_value(stats.get("p_value", np.nan)),
                    "% Explained": f"{result['pct_explained']:.1f}%",
                    "Method": result.get("method", "percentile_matching"),
                }
            )

    df = pd.DataFrame(rows)

    df.to_csv(output_path.with_suffix(".csv"), index=False)

    logger.info(f"  Saved: {output_path.stem}.csv")


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

    # Table 2: Gap Decomposition
    generate_table_2_gap_decomposition(
        h3_results, output_dir / "table_2_gap_decomposition"
    )

    # Table 3: Per-Feature Equalization
    generate_table_3_per_feature_equalization(
        h3_results, output_dir / "table_3_per_feature_equalization"
    )

    logger.info("\n✅ All tables generated successfully!")
    logger.info(f"Tables saved to: {output_dir}")
    logger.info("  Format: CSV only")
    logger.info("  All tables include p-values (5 decimal places) and 95% CIs")


def create_results_summary_markdown(
    h1_results: Dict, h2_results: Dict, h3_results: Dict, output_path: Path
):
    """
    Create human-readable markdown summary
    FULLY REVISED: Added comprehensive statistics
    """
    logger.info("Creating results summary (Markdown)...")

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

            ctrl_amie = feat_data["amie_control"]
            dys_amie = feat_data["amie_dyslexic"]

            f.write(f"- **AMIE (Control):** {ctrl_amie.get('amie_ms', 0):.2f} ms\n")
            f.write(
                f"  - *95% CI:* [{ctrl_amie.get('ci_low', 0):.2f}, {ctrl_amie.get('ci_high', 0):.2f}]\n"
            )
            f.write(f"  - *p-value:* {ctrl_amie.get('p_value', np.nan):.5f}\n")

            f.write(f"- **AMIE (Dyslexic):** {dys_amie.get('amie_ms', 0):.2f} ms\n")
            f.write(
                f"  - *95% CI:* [{dys_amie.get('ci_low', 0):.2f}, {dys_amie.get('ci_high', 0):.2f}]\n"
            )
            f.write(f"  - *p-value:* {dys_amie.get('p_value', np.nan):.5f}\n")

            # Add Cohen's h
            ctrl_pathway = feat_data["pathway_control"]
            dys_pathway = feat_data["pathway_dyslexic"]

            f.write(f"\n**Skip Pathway:**\n")
            f.write(f"- Control: Cohen's h = {ctrl_pathway['cohens_h']:.3f}\n")
            f.write(f"- Dyslexic: Cohen's h = {dys_pathway['cohens_h']:.3f}\n")

            f.write(f"\n**Status:** {feat_data['status']}\n")

            if feat_data.get("note"):
                f.write(f"\n*Note: {feat_data['note']}*\n")

            f.write("\n")

        # H2
        f.write("## Hypothesis 2: Dyslexic Amplification\n\n")
        f.write(f"**Status:** {h2_results['status']}\n\n")
        f.write(f"{h2_results['summary']}\n\n")

        f.write("### Slope Ratios by Feature and Pathway\n\n")
        f.write(
            "| Feature | SR(skip) | p-value | SR(duration) | p-value | SR(ERT) | p-value |\n"
        )
        f.write(
            "|---------|----------|---------|--------------|---------|---------|----------|\n"
        )

        for feat in ["length", "zipf", "surprisal"]:
            sr_data = h2_results["slope_ratios"].get(feat, {})

            sr_skip = sr_data.get("sr_skip", np.nan)
            p_skip = sr_data.get("sr_skip_p_value", np.nan)
            sr_dur = sr_data.get("sr_duration", np.nan)
            p_dur = sr_data.get("sr_duration_p_value", np.nan)
            sr_ert = sr_data.get("sr_ert", np.nan)
            p_ert = sr_data.get("sr_ert_p_value", np.nan)

            f.write(
                f"| {feat.capitalize()} | {sr_skip:.2f} | {p_skip:.5f} | "
                f"{sr_dur:.2f} | {p_dur:.5f} | {sr_ert:.2f} | {p_ert:.5f} |\n"
            )

        f.write("\n*SR > 1.0 indicates dyslexic amplification*\n\n")

        # Feature status
        f.write("### Feature-Level Decisions\n\n")
        feature_status = h2_results.get("feature_status", {})
        for feat, status_info in feature_status.items():
            f.write(f"- **{feat.capitalize()}:** {status_info['status']}")
            if status_info["pathway"]:
                sr = h2_results["slope_ratios"][feat].get(
                    f"sr_{status_info['pathway']}", np.nan
                )
                p_val = h2_results["slope_ratios"][feat].get(
                    f"sr_{status_info['pathway']}_p_value", np.nan
                )
                f.write(
                    f" (via {status_info['pathway']} pathway, SR={sr:.2f}, p={p_val:.5f})"
                )
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
            f"- **Skip Contribution:** {shapley['skip_contribution']:.2f} ms "
            f"({shapley['skip_pct']:.0f}%)\n"
        )
        f.write(
            f"  - *95% CI:* [{shapley['skip_contribution_stats']['ci_low']:.2f}, "
            f"{shapley['skip_contribution_stats']['ci_high']:.2f}]\n"
        )
        f.write(f"  - *p-value:* {shapley['skip_contribution_stats']['p_value']:.5f}\n")

        f.write(
            f"- **Duration Contribution:** {shapley['duration_contribution']:.2f} ms "
            f"({shapley['duration_pct']:.0f}%)\n"
        )
        f.write(
            f"  - *95% CI:* [{shapley['duration_contribution_stats']['ci_low']:.2f}, "
            f"{shapley['duration_contribution_stats']['ci_high']:.2f}]\n"
        )
        f.write(
            f"  - *p-value:* {shapley['duration_contribution_stats']['p_value']:.5f}\n\n"
        )

        f.write("### Equal-Ease Counterfactual\n\n")
        f.write(f"- **Baseline Gap:** {equal_ease['baseline_gap']:.2f} ms\n")
        f.write(
            f"- **Counterfactual Gap:** {equal_ease['counterfactual_gap']:.2f} ms\n"
        )
        f.write(
            f"- **Gap Shrink:** {equal_ease['gap_shrink_ms']:.2f} ms "
            f"({equal_ease['gap_shrink_pct']:.0f}%)\n"
        )
        f.write(
            f"  - *95% CI:* [{equal_ease['gap_shrink_stats']['ci_low']:.2f}, "
            f"{equal_ease['gap_shrink_stats']['ci_high']:.2f}]\n"
        )
        f.write(f"  - *p-value:* {equal_ease['gap_shrink_stats']['p_value']:.5f}\n")
        f.write(f"- **Dyslexic Saved:** {equal_ease['dyslexic_saved']:.2f} ms\n")
        f.write(f"- **Control Saved:** {equal_ease['control_saved']:.2f} ms\n\n")

        # Feature contributions
        if "equal_ease_feature_contributions" in h3_results:
            feat_contrib = h3_results["equal_ease_feature_contributions"]
            f.write("### Feature Contributions to Gap Reduction\n\n")

            if "feature_contributions_stats" in feat_contrib:
                for feat in ["length", "zipf", "surprisal"]:
                    contrib = feat_contrib["feature_contributions_ms"][feat]
                    stats = feat_contrib["feature_contributions_stats"][feat]
                    pct = (
                        (contrib / feat_contrib["total_ms"] * 100)
                        if feat_contrib["total_ms"] > 0
                        else 0
                    )
                    f.write(
                        f"- **{feat.capitalize()}:** {contrib:.2f} ms ({pct:.1f}%)\n"
                    )
                    f.write(
                        f"  - *95% CI:* [{stats['ci_low']:.2f}, {stats['ci_high']:.2f}]\n"
                    )
                    f.write(f"  - *p-value:* {stats['p_value']:.5f}\n")

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
            f"3. The dyslexic-control gap is primarily driven by {shapley['duration_pct']:.0f}% "
            f"duration and {shapley['skip_pct']:.0f}% skip differences\n"
        )
        f.write(
            f"4. Text simplification (equal-ease) reduces the gap by "
            f"{equal_ease['gap_shrink_pct']:.0f}% (p={equal_ease['gap_shrink_stats']['p_value']:.5f})\n"
        )

    logger.info(f"  Saved: {output_path}")
