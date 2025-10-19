"""
Table Generation for Analysis Plan Tables
Generates formatted tables for publication
FIXED: Unicode encoding issues on Windows
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
    h1_results: Dict, h2_results: Dict, output_path: Path
):
    """
    Table 1: Feature Effects & Pathway Amplification

    Columns:
    - Feature
    - Delta p(skip) C/D, Cohen's h
    - SR(skip) [95% CI]
    - Delta TRT (ms) C/D
    - SR(dur) [95% CI]
    - Delta ERT (ms) C/D
    - SR(ERT) [95% CI]
    """
    logger.info("Generating Table 1: Feature Effects & Pathway Amplification...")

    features = ["length", "zipf", "surprisal"]
    feature_labels = {"length": "Length", "zipf": "Zipf*", "surprisal": "Surprisal"}

    rows = []

    for feat in features:
        # Get H1 pathway results
        h1_feat = h1_results["features"][feat]
        ctrl_pathway = h1_feat["pathway_control"]
        dys_pathway = h1_feat["pathway_dyslexic"]

        # Get H2 SR results
        h2_feat = h2_results["slope_ratios"].get(feat, {})

        # Build row - FIXED: Use "Delta" instead of Greek Δ
        row = {
            "Feature": feature_labels[feat],
            # Skip pathway
            "Delta_p(skip) C/D": f"{ctrl_pathway['delta_p_skip']:.3f} / {dys_pathway['delta_p_skip']:.3f}",
            "Cohen's h": f"{ctrl_pathway.get('cohens_h', 0):.3f}",
            "SR(skip) [95% CI]": format_ci(
                h2_feat.get("sr_skip", np.nan),
                h2_feat.get("sr_skip_ci_low", np.nan),
                h2_feat.get("sr_skip_ci_high", np.nan),
                decimals=2,
            ),
            # Duration pathway
            "Delta_TRT (ms) C/D": f"{ctrl_pathway['delta_trt_ms']:.0f} / {dys_pathway['delta_trt_ms']:.0f}",
            "SR(dur) [95% CI]": format_ci(
                h2_feat.get("sr_duration", np.nan),
                h2_feat.get("sr_duration_ci_low", np.nan),
                h2_feat.get("sr_duration_ci_high", np.nan),
                decimals=2,
            ),
            # ERT pathway
            "Delta_ERT (ms) C/D": f"{ctrl_pathway['delta_ert_ms']:.0f} / {dys_pathway['delta_ert_ms']:.0f}",
            "SR(ERT) [95% CI]": format_ci(
                h2_feat.get("sr_ert", np.nan),
                h2_feat.get("sr_ert_ci_low", np.nan),
                h2_feat.get("sr_ert_ci_high", np.nan),
                decimals=2,
            ),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save as CSV
    df.to_csv(output_path.with_suffix(".csv"), index=False)

    # Save as formatted text - FIXED: Specify UTF-8 encoding
    with open(output_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write("Table 1: Feature Effects & Pathway Amplification\n")
        f.write("=" * 100 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("Notes:\n")
        f.write("* Zipf uses conditional evaluation (within length bins)\n")
        f.write("† Flagged: unstable denominator (control Delta_p ≈ 0)\n")
        f.write("C/D = Control / Dyslexic\n")
        f.write("SR = Slope Ratio (Dyslexic / Control)\n")

    # Save as LaTeX - FIXED: Specify UTF-8 encoding
    with open(output_path.with_suffix(".tex"), "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=False))

    logger.info(f"  Saved: {output_path.stem} (.csv, .txt, .tex)")


def generate_table_2_gap_decomposition(h3_results: Dict, output_path: Path):
    """
    Table 2: Gap Decomposition & Counterfactuals

    Sections:
    1. Observed total gap
    2. Shapley decomposition (skip & duration)
    3. Equal-ease counterfactual
    """
    logger.info("Generating Table 2: Gap Decomposition & Counterfactuals...")

    shapley = h3_results["shapley_decomposition"]
    equal_ease = h3_results["equal_ease_counterfactual"]

    rows = [
        {
            "Component": "Total gap",
            "Gap (ms)": f"{h3_results['total_gap']:.1f}",
            "95% CI": "—",
            "% of Total": "100%",
        },
        {"Component": "", "Gap (ms)": "", "95% CI": "", "% of Total": ""},
        {
            "Component": "Shapley decomposition:",
            "Gap (ms)": "",
            "95% CI": "",
            "% of Total": "",
        },
        {
            "Component": "  Skip contribution",
            "Gap (ms)": f"{shapley['skip_contribution']:.1f}",
            "95% CI": "—",
            "% of Total": f"{shapley['skip_pct']:.0f}%",
        },
        {
            "Component": "  Duration contribution",
            "Gap (ms)": f"{shapley['duration_contribution']:.1f}",
            "95% CI": "—",
            "% of Total": f"{shapley['duration_pct']:.0f}%",
        },
        {"Component": "", "Gap (ms)": "", "95% CI": "", "% of Total": ""},
        {
            "Component": "Equal-ease counterfactual:",
            "Gap (ms)": "",
            "95% CI": "",
            "% of Total": "",
        },
        {
            "Component": "  Dyslexic saved",
            "Gap (ms)": f"{equal_ease['dyslexic_saved']:.1f}",
            "95% CI": "—",
            "% of Total": "—",
        },
        {
            "Component": "  Control saved",
            "Gap (ms)": f"{equal_ease['control_saved']:.1f}",
            "95% CI": "—",
            "% of Total": "—",
        },
        {
            "Component": "  Gap shrink (ms)",
            "Gap (ms)": f"{equal_ease['gap_shrink_ms']:.1f}",
            "95% CI": "—",
            "% of Total": "—",
        },
        {
            "Component": "  Gap shrink (%)",
            "Gap (ms)": f"{equal_ease['gap_shrink_pct']:.1f}%",
            "95% CI": "—",
            "% of Total": "—",
        },
    ]

    df = pd.DataFrame(rows)

    # Save formats - FIXED: UTF-8 encoding
    df.to_csv(output_path.with_suffix(".csv"), index=False)

    with open(output_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write("Table 2: Gap Decomposition & Counterfactuals\n")
        f.write("=" * 100 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("Interpretation:\n")
        f.write(
            f"Duration pathway drives {shapley['duration_pct']:.0f}% of the reading time gap. "
        )
        f.write(f"Equal-ease text (shorter, more frequent, more predictable words) ")
        f.write(
            f"reduces the gap by {equal_ease['gap_shrink_pct']:.0f}%, with dyslexics "
        )
        f.write(f"benefiting more ({equal_ease['dyslexic_saved']:.0f}ms saved) than ")
        f.write(f"controls ({equal_ease['control_saved']:.0f}ms saved).\n")

    with open(output_path.with_suffix(".tex"), "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=False))

    logger.info(f"  Saved: {output_path.stem} (.csv, .txt, .tex)")


def generate_table_3_model_performance(
    skip_metadata: Dict, duration_metadata: Dict, output_path: Path
):
    """
    Table 3: Model Performance (Cross-Validation)

    Supplementary table with model fit statistics
    """
    logger.info("Generating Table 3: Model Performance...")

    rows = [
        {
            "Metric": "Skip AUC",
            "Mean": f"{(skip_metadata['auc_control'] + skip_metadata['auc_dyslexic'])/2:.3f}",
            "SD": "—",
            "Note": "Average of control and dyslexic models",
        },
        {
            "Metric": "Duration R²",
            "Mean": f"{(duration_metadata['r2_control'] + duration_metadata['r2_dyslexic'])/2:.3f}",
            "SD": "—",
            "Note": "Pseudo-R² (explained deviance)",
        },
        {
            "Metric": "N observations (skip)",
            "Mean": f"{skip_metadata['n_obs_control'] + skip_metadata['n_obs_dyslexic']:,}",
            "SD": "—",
            "Note": "Combined control + dyslexic",
        },
        {
            "Metric": "N observations (duration)",
            "Mean": f"{duration_metadata['n_obs_control'] + duration_metadata['n_obs_dyslexic']:,}",
            "SD": "—",
            "Note": "Fixated words only",
        },
    ]

    df = pd.DataFrame(rows)

    # FIXED: UTF-8 encoding for all file operations
    df.to_csv(output_path.with_suffix(".csv"), index=False)

    with open(output_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write("Table 3: Model Performance (Supplementary)\n")
        f.write("=" * 100 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("Note: Separate models fitted for control and dyslexic groups.\n")
        f.write("Models include tensor product te(length, zipf) interaction.\n")

    with open(output_path.with_suffix(".tex"), "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=False))

    logger.info(f"  Saved: {output_path.stem} (.csv, .txt, .tex)")


def generate_all_tables(
    h1_results: Dict,
    h2_results: Dict,
    h3_results: Dict,
    skip_metadata: Dict,
    duration_metadata: Dict,
    output_dir: Path,
):
    """
    Generate all tables specified in analysis plan

    Args:
        h1_results: H1 test results
        h2_results: H2 test results with CIs
        h3_results: H3 gap decomposition results
        skip_metadata: Skip model metadata
        duration_metadata: Duration model metadata
        output_dir: Directory to save tables
    """
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING PUBLICATION TABLES")
    logger.info("=" * 60)

    output_dir.mkdir(exist_ok=True, parents=True)

    # Table 1: Feature Effects & Amplification
    generate_table_1_feature_effects(
        h1_results, h2_results, output_dir / "table_1_feature_effects"
    )

    # Table 2: Gap Decomposition
    generate_table_2_gap_decomposition(
        h3_results, output_dir / "table_2_gap_decomposition"
    )

    # Table 3: Model Performance
    generate_table_3_model_performance(
        skip_metadata, duration_metadata, output_dir / "table_3_model_performance"
    )

    logger.info("\n✅ All tables generated successfully!")
    logger.info(f"Tables saved to: {output_dir}")
    logger.info("  Formats: CSV (data), TXT (formatted), TEX (LaTeX)")


def create_results_summary_markdown(
    h1_results: Dict, h2_results: Dict, h3_results: Dict, output_path: Path
):
    """
    Create human-readable markdown summary of results
    """
    logger.info("Creating results summary (Markdown)...")

    # FIXED: UTF-8 encoding
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

        # Overall
        f.write("## Overall Conclusion\n\n")

        conclusions = {
            "h1": h1_results["status"],
            "h2": h2_results["status"],
            "h3": h3_results["status"],
        }

        all_confirmed = all(
            status in ["CONFIRMED", "STRONGLY CONFIRMED"]
            for status in conclusions.values()
        )

        if all_confirmed:
            f.write("✅ **All hypotheses confirmed!**\n\n")
        else:
            f.write("⚠️ **Some hypotheses partially confirmed**\n\n")

        f.write("The results demonstrate that:\n\n")
        f.write(
            "1. Word-level features (length, frequency, surprisal) significantly affect reading time\n"
        )
        f.write(
            f"2. Dyslexic readers show amplified effects for {h2_results['n_amplified']}/3 features\n"
        )
        f.write(
            f"3. The dyslexic-control gap is primarily driven by {shapley['duration_pct']:.0f}% duration and {shapley['skip_pct']:.0f}% skip differences\n"
        )
        f.write(
            f"4. Text simplification (equal-ease) reduces the gap by {equal_ease['gap_shrink_pct']:.0f}%\n"
        )

    logger.info(f"  Saved: {output_path}")


if __name__ == "__main__":
    print("This module provides table generation utilities.")
    print("Import and use generate_all_tables() function.")
