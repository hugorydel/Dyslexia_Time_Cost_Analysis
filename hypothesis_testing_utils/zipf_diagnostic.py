"""
Diagnostic script to investigate the zipf frequency paradox
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    """Load the prepared data"""
    script_dir = Path(__file__).resolve().parent.parent
    data_path = script_dir / "preprocessing_results" / "preprocessed_data.csv"

    data = pd.read_csv(data_path)
    logger.info(f"Loaded {len(data):,} observations")
    return data


def check_zipf_distribution(data):
    """Check zipf distribution and compute basic stats"""
    logger.info("\n" + "=" * 60)
    logger.info("ZIPF DISTRIBUTION CHECK")
    logger.info("=" * 60)

    zipf_col = "word_frequency_zipf"

    if zipf_col not in data.columns:
        logger.error(f"Column {zipf_col} not found!")
        return

    # Basic stats
    logger.info(f"\nZipf statistics:")
    logger.info(f"  Mean: {data[zipf_col].mean():.3f}")
    logger.info(f"  Median: {data[zipf_col].median():.3f}")
    logger.info(f"  Std: {data[zipf_col].std():.3f}")
    logger.info(f"  Min: {data[zipf_col].min():.3f}")
    logger.info(f"  Max: {data[zipf_col].max():.3f}")
    logger.info(f"  Q1: {data[zipf_col].quantile(0.25):.3f}")
    logger.info(f"  Q3: {data[zipf_col].quantile(0.75):.3f}")

    # Check for any issues
    n_missing = data[zipf_col].isna().sum()
    n_negative = (data[zipf_col] < 0).sum()
    n_zero = (data[zipf_col] == 0).sum()

    logger.info(f"\nData quality:")
    logger.info(f"  Missing values: {n_missing:,}")
    logger.info(f"  Negative values: {n_negative:,}")
    logger.info(f"  Zero values: {n_zero:,}")

    # Check distribution by group
    logger.info(f"\nZipf by group:")
    for group in [True, False]:
        group_name = "Dyslexic" if group else "Control"
        group_data = data[data["dyslexic"] == group]
        logger.info(
            f"  {group_name}: mean={group_data[zipf_col].mean():.3f}, "
            f"median={group_data[zipf_col].median():.3f}"
        )


def check_correlations(data):
    """Check correlations between features"""
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE CORRELATIONS")
    logger.info("=" * 60)

    features = ["word_length", "word_frequency_zipf", "surprisal"]

    # Compute correlation matrix
    corr_matrix = data[features].corr()

    logger.info("\nPearson correlations:")
    logger.info(corr_matrix.to_string())

    # Check if zipf is highly correlated with others
    zipf_corr_length = corr_matrix.loc["word_frequency_zipf", "word_length"]
    zipf_corr_surprisal = corr_matrix.loc["word_frequency_zipf", "surprisal"]

    logger.info(f"\nZipf correlations:")
    logger.info(f"  with Length: {zipf_corr_length:.3f}")
    logger.info(f"  with Surprisal: {zipf_corr_surprisal:.3f}")

    if abs(zipf_corr_length) > 0.7 or abs(zipf_corr_surprisal) > 0.7:
        logger.warning(
            "⚠️  High correlation detected! This could cause multicollinearity."
        )


def analyze_raw_relationship(data):
    """Analyze raw relationship between zipf and reading time"""
    logger.info("\n" + "=" * 60)
    logger.info("RAW ZIPF-ERT RELATIONSHIP")
    logger.info("=" * 60)

    # Focus on fixated words
    fixated = data[data["n_fixations"] > 0].copy()

    # Bin zipf into quartiles
    fixated["zipf_bin"] = pd.qcut(
        fixated["word_frequency_zipf"],
        q=4,
        labels=["Q1 (rare)", "Q2", "Q3", "Q4 (freq)"],
    )

    logger.info("\nMean reading time by zipf quartile:")
    for group in [True, False]:
        group_name = "Dyslexic" if group else "Control"
        group_data = fixated[fixated["dyslexic"] == group]

        logger.info(f"\n{group_name}:")
        for bin_label in ["Q1 (rare)", "Q2", "Q3", "Q4 (freq)"]:
            bin_data = group_data[group_data["zipf_bin"] == bin_label]
            mean_trt = bin_data["total_reading_time"].mean()
            logger.info(f"  {bin_label}: {mean_trt:.1f} ms (n={len(bin_data):,})")

    # Compute raw correlations
    logger.info("\nRaw correlations (zipf vs TRT):")
    for group in [True, False]:
        group_name = "Dyslexic" if group else "Control"
        group_data = fixated[fixated["dyslexic"] == group]

        corr, pval = pearsonr(
            group_data["word_frequency_zipf"], group_data["total_reading_time"]
        )
        logger.info(f"  {group_name}: r={corr:.3f}, p={pval:.4f}")


def analyze_skip_relationship(data):
    """Analyze relationship between zipf and skipping"""
    logger.info("\n" + "=" * 60)
    logger.info("ZIPF-SKIP RELATIONSHIP")
    logger.info("=" * 60)

    # Create skip indicator
    data["skip"] = (data["n_fixations"] == 0).astype(int)

    # Bin zipf
    data["zipf_bin"] = pd.qcut(
        data["word_frequency_zipf"], q=4, labels=["Q1 (rare)", "Q2", "Q3", "Q4 (freq)"]
    )

    logger.info("\nSkip rate by zipf quartile:")
    for group in [True, False]:
        group_name = "Dyslexic" if group else "Control"
        group_data = data[data["dyslexic"] == group]

        logger.info(f"\n{group_name}:")
        for bin_label in ["Q1 (rare)", "Q2", "Q3", "Q4 (freq)"]:
            bin_data = group_data[group_data["zipf_bin"] == bin_label]
            skip_rate = bin_data["skip"].mean()
            logger.info(
                f"  {bin_label}: {skip_rate:.3f} skip rate (n={len(bin_data):,})"
            )

    # Expected: Higher frequency → higher skip rate
    # If this is violated, we have a problem


def check_zipf_computation(data):
    """Check if zipf values make sense"""
    logger.info("\n" + "=" * 60)
    logger.info("ZIPF COMPUTATION CHECK")
    logger.info("=" * 60)

    # Sample some words and their zipf values
    logger.info("\nSample words and their zipf values:")

    # Get unique words
    if "word_text" in data.columns:
        word_stats = (
            data.groupby("word_text")
            .agg(
                {
                    "word_frequency_zipf": "first",
                    "word_length": "first",
                }
            )
            .reset_index()
        )

        # Sort by zipf
        word_stats_sorted = word_stats.sort_values("word_frequency_zipf")

        logger.info("\n10 rarest words:")
        for _, row in word_stats_sorted.head(10).iterrows():
            logger.info(
                f"  '{row['word_text']}': zipf={row['word_frequency_zipf']:.2f}, "
                f"length={row['word_length']}"
            )

        logger.info("\n10 most frequent words:")
        for _, row in word_stats_sorted.tail(10).iterrows():
            logger.info(
                f"  '{row['word_text']}': zipf={row['word_frequency_zipf']:.2f}, "
                f"length={row['word_length']}"
            )
    else:
        logger.warning("word_text column not found - cannot check word examples")


def create_diagnostic_plots(data):
    """Create diagnostic plots"""
    logger.info("\n" + "=" * 60)
    logger.info("CREATING DIAGNOSTIC PLOTS")
    logger.info("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Zipf distribution by group
    ax = axes[0, 0]
    for group in [False, True]:
        group_name = "Control" if not group else "Dyslexic"
        color = "blue" if not group else "orange"
        group_data = data[data["dyslexic"] == group]
        ax.hist(
            group_data["word_frequency_zipf"],
            bins=50,
            alpha=0.5,
            label=group_name,
            color=color,
        )
    ax.set_xlabel("Zipf Frequency")
    ax.set_ylabel("Count")
    ax.set_title("Zipf Distribution by Group")
    ax.legend()

    # 2. Zipf vs Length
    ax = axes[0, 1]
    sample = data.sample(min(10000, len(data)))
    ax.scatter(sample["word_frequency_zipf"], sample["word_length"], alpha=0.1, s=1)
    ax.set_xlabel("Zipf Frequency")
    ax.set_ylabel("Word Length")
    ax.set_title("Zipf vs Length")

    # 3. Zipf vs Surprisal
    ax = axes[0, 2]
    ax.scatter(sample["word_frequency_zipf"], sample["surprisal"], alpha=0.1, s=1)
    ax.set_xlabel("Zipf Frequency")
    ax.set_ylabel("Surprisal")
    ax.set_title("Zipf vs Surprisal")

    # 4. Zipf vs TRT (fixated only)
    ax = axes[1, 0]
    fixated = data[data["n_fixations"] > 0]
    for group in [False, True]:
        group_name = "Control" if not group else "Dyslexic"
        color = "blue" if not group else "orange"
        group_data = fixated[fixated["dyslexic"] == group].sample(
            min(5000, len(fixated))
        )
        ax.scatter(
            group_data["word_frequency_zipf"],
            group_data["total_reading_time"],
            alpha=0.1,
            s=1,
            color=color,
            label=group_name,
        )
    ax.set_xlabel("Zipf Frequency")
    ax.set_ylabel("Total Reading Time (ms)")
    ax.set_title("Zipf vs TRT (Fixated Words)")
    ax.set_ylim(0, 1000)
    ax.legend()

    # 5. Binned relationship (Zipf vs TRT)
    ax = axes[1, 1]
    fixated["zipf_bin"] = pd.cut(fixated["word_frequency_zipf"], bins=10)
    binned = (
        fixated.groupby(["zipf_bin", "dyslexic"])["total_reading_time"]
        .mean()
        .reset_index()
    )

    for group in [False, True]:
        group_name = "Control" if not group else "Dyslexic"
        color = "blue" if not group else "orange"
        group_data = binned[binned["dyslexic"] == group]
        bin_centers = [interval.mid for interval in group_data["zipf_bin"]]
        ax.plot(
            bin_centers,
            group_data["total_reading_time"],
            marker="o",
            label=group_name,
            color=color,
        )
    ax.set_xlabel("Zipf Frequency (binned)")
    ax.set_ylabel("Mean TRT (ms)")
    ax.set_title("Binned: Zipf vs TRT")
    ax.legend()

    # 6. Skip rate by zipf
    ax = axes[1, 2]
    data["skip"] = (data["n_fixations"] == 0).astype(int)
    data["zipf_bin"] = pd.cut(data["word_frequency_zipf"], bins=10)
    skip_binned = data.groupby(["zipf_bin", "dyslexic"])["skip"].mean().reset_index()

    for group in [False, True]:
        group_name = "Control" if not group else "Dyslexic"
        color = "blue" if not group else "orange"
        group_data = skip_binned[skip_binned["dyslexic"] == group]
        bin_centers = [interval.mid for interval in group_data["zipf_bin"]]
        ax.plot(
            bin_centers, group_data["skip"], marker="o", label=group_name, color=color
        )
    ax.set_xlabel("Zipf Frequency (binned)")
    ax.set_ylabel("Skip Rate")
    ax.set_title("Binned: Zipf vs Skip Rate")
    ax.legend()

    plt.tight_layout()
    plt.savefig("zipf_diagnostic_plots.png", dpi=150, bbox_inches="tight")
    logger.info("Plots saved to zipf_diagnostic_plots.png")
    plt.close()


def main():
    """Run all diagnostics"""
    logger.info("=" * 80)
    logger.info("ZIPF FREQUENCY PARADOX DIAGNOSTIC")
    logger.info("=" * 80)

    # Load data
    data = load_data()

    # Run diagnostics
    check_zipf_distribution(data)
    check_correlations(data)
    check_zipf_computation(data)
    analyze_skip_relationship(data)
    analyze_raw_relationship(data)
    create_diagnostic_plots(data)

    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
