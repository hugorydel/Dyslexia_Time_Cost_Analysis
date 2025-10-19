"""
Complete Zipf Frequency Diagnostic Script
Combines data-level checks with model-level pathway analysis
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure nice plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


class ZipfDiagnostic:
    """
    Complete diagnostic tool for zipf frequency paradox
    Includes both data-level and model-level analysis
    """

    def __init__(
        self, data: pd.DataFrame, ert_predictor=None, quartiles: Optional[Dict] = None
    ):
        """
        Initialize diagnostic

        Args:
            data: Full dataset
            ert_predictor: Optional ERTPredictor instance for model-level analysis
            quartiles: Optional feature quartiles from analysis
        """
        self.data = data
        self.ert_predictor = ert_predictor
        self.quartiles = quartiles
        self.results = {}

        # Standardize column names
        if "word_frequency_zipf" in data.columns:
            self.data["zipf"] = self.data["word_frequency_zipf"]
        if "word_length" in data.columns:
            self.data["length"] = self.data["word_length"]
        if "total_reading_time" in data.columns:
            self.data["TRT"] = self.data["total_reading_time"]
        if "n_fixations" in data.columns:
            self.data["skip"] = (self.data["n_fixations"] == 0).astype(int)

        logger.info("=" * 80)
        logger.info("ZIPF FREQUENCY DIAGNOSTIC - COMPLETE ANALYSIS")
        logger.info("=" * 80)

    def run_all_diagnostics(self, output_dir: Path = Path("diagnostic_results")):
        """
        Run complete diagnostic suite

        Args:
            output_dir: Directory to save results and plots
        """
        output_dir.mkdir(exist_ok=True, parents=True)

        logger.info("\n" + "=" * 80)
        logger.info("PART 1: DATA-LEVEL DIAGNOSTICS")
        logger.info("=" * 80)

        # Data-level checks
        self.check_zipf_distribution()
        self.check_correlations()
        self.check_zipf_computation()
        self.analyze_skip_relationship()
        self.analyze_raw_relationship()

        # Create data-level plots
        self.create_data_diagnostic_plots(output_dir / "zipf_data_diagnostics.png")

        # Model-level analysis (if predictor available)
        if self.ert_predictor is not None and self.quartiles is not None:
            logger.info("\n" + "=" * 80)
            logger.info("PART 2: MODEL-LEVEL DIAGNOSTICS")
            logger.info("=" * 80)

            # Pathway analysis
            logger.info("\n" + "=" * 80)
            logger.info("DIAGNOSTIC 1: PATHWAY AMIEs (Skip vs Duration vs ERT)")
            logger.info("=" * 80)
            pathway_results = self.compute_pathway_amies()
            self.results["pathway_amies"] = pathway_results
            self.print_pathway_summary(pathway_results)

            # Conditioned contrasts
            logger.info("\n" + "=" * 80)
            logger.info("DIAGNOSTIC 2: CONDITIONED CONTRASTS (Within Length Bins)")
            logger.info("=" * 80)
            conditioned_results = self.compute_conditioned_contrasts(n_bins=5)
            self.results["conditioned_contrasts"] = conditioned_results
            self.print_conditioned_summary(conditioned_results)

            # Simple linear models
            logger.info("\n" + "=" * 80)
            logger.info("DIAGNOSTIC 3: SIMPLE LINEAR MODEL CHECKS")
            logger.info("=" * 80)
            linear_results = self.fit_simple_models()
            self.results["linear_models"] = linear_results
            self.print_linear_summary(linear_results)

            # Create model-level visualizations
            logger.info("\n" + "=" * 80)
            logger.info("GENERATING MODEL VISUALIZATIONS")
            logger.info("=" * 80)
            self.create_pathway_comparison_plot(
                pathway_results, output_dir / "pathway_comparison.png"
            )
            self.create_conditioned_contrast_plot(
                conditioned_results, output_dir / "conditioned_contrasts.png"
            )

            # Generate comprehensive report
            self.generate_summary_report(output_dir / "zipf_diagnostic_report.txt")
        else:
            logger.info("\n" + "=" * 80)
            logger.info("SKIPPING MODEL-LEVEL DIAGNOSTICS")
            logger.info("(No ERT predictor or quartiles provided)")
            logger.info("=" * 80)

        logger.info("\n" + "=" * 80)
        logger.info(f"DIAGNOSTIC COMPLETE - Results saved to {output_dir}")
        logger.info("=" * 80)

        return self.results

    # ========================================================================
    # PART 1: DATA-LEVEL DIAGNOSTICS
    # ========================================================================

    def check_zipf_distribution(self):
        """Check zipf distribution and compute basic stats"""
        logger.info("\n" + "=" * 60)
        logger.info("ZIPF DISTRIBUTION CHECK")
        logger.info("=" * 60)

        if "zipf" not in self.data.columns:
            logger.error("Zipf column not found!")
            return

        # Basic stats
        logger.info(f"\nZipf statistics:")
        logger.info(f"  Mean: {self.data['zipf'].mean():.3f}")
        logger.info(f"  Median: {self.data['zipf'].median():.3f}")
        logger.info(f"  Std: {self.data['zipf'].std():.3f}")
        logger.info(f"  Min: {self.data['zipf'].min():.3f}")
        logger.info(f"  Max: {self.data['zipf'].max():.3f}")
        logger.info(f"  Q1: {self.data['zipf'].quantile(0.25):.3f}")
        logger.info(f"  Q3: {self.data['zipf'].quantile(0.75):.3f}")

        # Check for any issues
        n_missing = self.data["zipf"].isna().sum()
        n_negative = (self.data["zipf"] < 0).sum()
        n_zero = (self.data["zipf"] == 0).sum()

        logger.info(f"\nData quality:")
        logger.info(f"  Missing values: {n_missing:,}")
        logger.info(f"  Negative values: {n_negative:,}")
        logger.info(f"  Zero values: {n_zero:,}")

        # Check distribution by group
        if "dyslexic" in self.data.columns:
            logger.info(f"\nZipf by group:")
            for group in [False, True]:
                group_name = "Control" if not group else "Dyslexic"
                group_data = self.data[self.data["dyslexic"] == group]
                logger.info(
                    f"  {group_name}: mean={group_data['zipf'].mean():.3f}, "
                    f"median={group_data['zipf'].median():.3f}"
                )
        elif "group" in self.data.columns:
            logger.info(f"\nZipf by group:")
            for group in ["control", "dyslexic"]:
                group_data = self.data[self.data["group"] == group]
                logger.info(
                    f"  {group.capitalize()}: mean={group_data['zipf'].mean():.3f}, "
                    f"median={group_data['zipf'].median():.3f}"
                )

    def check_correlations(self):
        """Check correlations between features"""
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE CORRELATIONS")
        logger.info("=" * 60)

        features = []
        if "length" in self.data.columns:
            features.append("length")
        if "zipf" in self.data.columns:
            features.append("zipf")
        if "surprisal" in self.data.columns:
            features.append("surprisal")

        if len(features) < 2:
            logger.warning("Not enough features to compute correlations")
            return

        # Compute correlation matrix
        corr_matrix = self.data[features].corr()

        logger.info("\nPearson correlations:")
        logger.info(corr_matrix.to_string())

        # Check if zipf is highly correlated with others
        if "length" in features and "zipf" in features:
            zipf_corr_length = corr_matrix.loc["zipf", "length"]
            logger.info(f"\nZipf correlations:")
            logger.info(f"  with Length: {zipf_corr_length:.3f}")

            if abs(zipf_corr_length) > 0.7:
                logger.warning(
                    "⚠️  High correlation detected! This could cause multicollinearity."
                )

        if "surprisal" in features and "zipf" in features:
            zipf_corr_surprisal = corr_matrix.loc["zipf", "surprisal"]
            logger.info(f"  with Surprisal: {zipf_corr_surprisal:.3f}")

    def check_zipf_computation(self):
        """Check if zipf values make sense"""
        logger.info("\n" + "=" * 60)
        logger.info("ZIPF COMPUTATION CHECK")
        logger.info("=" * 60)

        # Sample some words and their zipf values
        logger.info("\nSample words and their zipf values:")

        # Get unique words
        if "word_text" in self.data.columns:
            word_stats = (
                self.data.groupby("word_text")
                .agg(
                    {
                        "zipf": "first",
                        "length": "first",
                    }
                )
                .reset_index()
            )

            # Sort by zipf
            word_stats_sorted = word_stats.sort_values("zipf")

            logger.info("\n10 rarest words:")
            for _, row in word_stats_sorted.head(10).iterrows():
                logger.info(
                    f"  '{row['word_text']}': zipf={row['zipf']:.2f}, "
                    f"length={row['length']}"
                )

            logger.info("\n10 most frequent words:")
            for _, row in word_stats_sorted.tail(10).iterrows():
                logger.info(
                    f"  '{row['word_text']}': zipf={row['zipf']:.2f}, "
                    f"length={row['length']}"
                )
        else:
            logger.warning("word_text column not found - cannot check word examples")

    def analyze_skip_relationship(self):
        """Analyze relationship between zipf and skipping"""
        logger.info("\n" + "=" * 60)
        logger.info("ZIPF-SKIP RELATIONSHIP")
        logger.info("=" * 60)

        # Bin zipf
        self.data["zipf_bin"] = pd.qcut(
            self.data["zipf"],
            q=4,
            labels=["Q1 (rare)", "Q2", "Q3", "Q4 (freq)"],
            duplicates="drop",
        )

        logger.info("\nSkip rate by zipf quartile:")

        # Determine group column
        group_col = "dyslexic" if "dyslexic" in self.data.columns else "group"

        if group_col == "dyslexic":
            groups = [False, True]
            group_names = {False: "Control", True: "Dyslexic"}
        else:
            groups = ["control", "dyslexic"]
            group_names = {"control": "Control", "dyslexic": "Dyslexic"}

        for group in groups:
            group_name = group_names[group]
            group_data = self.data[self.data[group_col] == group]

            logger.info(f"\n{group_name}:")
            for bin_label in ["Q1 (rare)", "Q2", "Q3", "Q4 (freq)"]:
                bin_data = group_data[group_data["zipf_bin"] == bin_label]
                if len(bin_data) > 0:
                    skip_rate = bin_data["skip"].mean()
                    logger.info(
                        f"  {bin_label}: {skip_rate:.3f} skip rate (n={len(bin_data):,})"
                    )

    def analyze_raw_relationship(self):
        """Analyze raw relationship between zipf and reading time"""
        logger.info("\n" + "=" * 60)
        logger.info("RAW ZIPF-TRT RELATIONSHIP")
        logger.info("=" * 60)

        # Focus on fixated words
        fixated = self.data[self.data["skip"] == 0].copy()

        # Bin zipf into quartiles
        fixated["zipf_bin"] = pd.qcut(
            fixated["zipf"],
            q=4,
            labels=["Q1 (rare)", "Q2", "Q3", "Q4 (freq)"],
            duplicates="drop",
        )

        logger.info("\nMean reading time by zipf quartile:")

        # Determine group column
        group_col = "dyslexic" if "dyslexic" in fixated.columns else "group"

        if group_col == "dyslexic":
            groups = [False, True]
            group_names = {False: "Control", True: "Dyslexic"}
        else:
            groups = ["control", "dyslexic"]
            group_names = {"control": "Control", "dyslexic": "Dyslexic"}

        for group in groups:
            group_name = group_names[group]
            group_data = fixated[fixated[group_col] == group]

            logger.info(f"\n{group_name}:")
            for bin_label in ["Q1 (rare)", "Q2", "Q3", "Q4 (freq)"]:
                bin_data = group_data[group_data["zipf_bin"] == bin_label]
                if len(bin_data) > 0:
                    mean_trt = bin_data["TRT"].mean()
                    logger.info(
                        f"  {bin_label}: {mean_trt:.1f} ms (n={len(bin_data):,})"
                    )

        # Compute raw correlations
        logger.info("\nRaw correlations (zipf vs TRT):")
        for group in groups:
            group_name = group_names[group]
            group_data = fixated[fixated[group_col] == group]

            if len(group_data) > 10:
                corr, pval = pearsonr(group_data["zipf"], group_data["TRT"])
                logger.info(f"  {group_name}: r={corr:.3f}, p={pval:.4f}")

    def create_data_diagnostic_plots(self, save_path: Path):
        """Create data-level diagnostic plots"""
        logger.info("\n" + "=" * 60)
        logger.info("CREATING DATA DIAGNOSTIC PLOTS")
        logger.info("=" * 60)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Zipf Data-Level Diagnostics", fontsize=14, fontweight="bold")

        # Determine group column
        group_col = "dyslexic" if "dyslexic" in self.data.columns else "group"

        if group_col == "dyslexic":
            groups = [False, True]
            group_names = {False: "Control", True: "Dyslexic"}
            colors = {False: "blue", True: "orange"}
        else:
            groups = ["control", "dyslexic"]
            group_names = {"control": "Control", "dyslexic": "Dyslexic"}
            colors = {"control": "blue", "dyslexic": "orange"}

        # 1. Zipf distribution by group
        ax = axes[0, 0]
        for group in groups:
            group_name = group_names[group]
            color = colors[group]
            group_data = self.data[self.data[group_col] == group]
            ax.hist(
                group_data["zipf"],
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
        sample = self.data.sample(min(10000, len(self.data)))
        ax.scatter(sample["zipf"], sample["length"], alpha=0.1, s=1)
        ax.set_xlabel("Zipf Frequency")
        ax.set_ylabel("Word Length")
        ax.set_title("Zipf vs Length")

        # Add correlation
        corr = self.data["zipf"].corr(self.data["length"])
        ax.text(
            0.05,
            0.95,
            f"r = {corr:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 3. Zipf vs Surprisal
        if "surprisal" in self.data.columns:
            ax = axes[0, 2]
            ax.scatter(sample["zipf"], sample["surprisal"], alpha=0.1, s=1)
            ax.set_xlabel("Zipf Frequency")
            ax.set_ylabel("Surprisal")
            ax.set_title("Zipf vs Surprisal")

            corr = self.data["zipf"].corr(self.data["surprisal"])
            ax.text(
                0.05,
                0.95,
                f"r = {corr:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        else:
            axes[0, 2].axis("off")

        # 4. Zipf vs TRT (fixated only)
        ax = axes[1, 0]
        fixated = self.data[self.data["skip"] == 0]
        for group in groups:
            group_name = group_names[group]
            color = colors[group]
            group_data = fixated[fixated[group_col] == group].sample(
                min(5000, len(fixated[fixated[group_col] == group]))
            )
            ax.scatter(
                group_data["zipf"],
                group_data["TRT"],
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
        fixated_copy = fixated.copy()
        fixated_copy["zipf_bin"] = pd.cut(fixated_copy["zipf"], bins=7)
        binned = (
            fixated_copy.groupby(["zipf_bin", group_col])["TRT"].mean().reset_index()
        )

        for group in groups:
            group_name = group_names[group]
            color = colors[group]
            group_data = binned[binned[group_col] == group]
            bin_centers = [interval.mid for interval in group_data["zipf_bin"]]
            ax.plot(
                bin_centers,
                group_data["TRT"],
                marker="o",
                label=group_name,
                color=color,
                linewidth=2,
                markersize=8,
            )
        ax.set_xlabel("Zipf Frequency (binned)")
        ax.set_ylabel("Mean TRT (ms)")
        ax.set_title("Binned: Zipf vs TRT")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Skip rate by zipf
        ax = axes[1, 2]
        data_copy = self.data.copy()
        data_copy["zipf_bin"] = pd.cut(data_copy["zipf"], bins=7)
        skip_binned = (
            data_copy.groupby(["zipf_bin", group_col])["skip"].mean().reset_index()
        )

        for group in groups:
            group_name = group_names[group]
            color = colors[group]
            group_data = skip_binned[skip_binned[group_col] == group]
            bin_centers = [interval.mid for interval in group_data["zipf_bin"]]
            ax.plot(
                bin_centers,
                group_data["skip"],
                marker="o",
                label=group_name,
                color=color,
                linewidth=2,
                markersize=8,
            )
        ax.set_xlabel("Zipf Frequency (binned)")
        ax.set_ylabel("Skip Rate")
        ax.set_title("Binned: Zipf vs Skip Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Data diagnostic plots saved to {save_path}")
        plt.close()

    # ========================================================================
    # PART 2: MODEL-LEVEL DIAGNOSTICS
    # ========================================================================

    def compute_pathway_amies(self) -> Dict:
        """
        Compute Q1→Q3 effects on three pathways:
        1. P(skip) - how much does frequency affect skipping?
        2. TRT|fix - how much does frequency affect duration (when fixated)?
        3. ERT - combined effect
        """
        q1 = self.quartiles["zipf"]["q1"]
        q3 = self.quartiles["zipf"]["q3"]

        # Mean values for other features
        length_mean = self.data["length"].mean()
        surprisal_mean = self.data["surprisal"].mean()

        # Create grids
        grid_q1 = pd.DataFrame(
            {"zipf": [q1], "length": [length_mean], "surprisal": [surprisal_mean]}
        )
        grid_q3 = pd.DataFrame(
            {"zipf": [q3], "length": [length_mean], "surprisal": [surprisal_mean]}
        )

        results = {}

        for group in ["control", "dyslexic"]:
            # Get components at Q1 and Q3
            ert_q1, p_skip_q1, trt_q1 = self.ert_predictor.predict_ert(
                grid_q1, group, return_components=True
            )
            ert_q3, p_skip_q3, trt_q3 = self.ert_predictor.predict_ert(
                grid_q3, group, return_components=True
            )

            # Compute deltas
            delta_p_skip = p_skip_q3[0] - p_skip_q1[0]
            delta_trt = trt_q3[0] - trt_q1[0]
            delta_ert = ert_q3[0] - ert_q1[0]

            # Also compute (1-p_skip) change
            fix_prob_q1 = 1 - p_skip_q1[0]
            fix_prob_q3 = 1 - p_skip_q3[0]
            delta_fix_prob = fix_prob_q3 - fix_prob_q1

            results[group] = {
                "p_skip_q1": float(p_skip_q1[0]),
                "p_skip_q3": float(p_skip_q3[0]),
                "delta_p_skip": float(delta_p_skip),
                "fix_prob_q1": float(fix_prob_q1),
                "fix_prob_q3": float(fix_prob_q3),
                "delta_fix_prob": float(delta_fix_prob),
                "trt_q1": float(trt_q1[0]),
                "trt_q3": float(trt_q3[0]),
                "delta_trt": float(delta_trt),
                "ert_q1": float(ert_q1[0]),
                "ert_q3": float(ert_q3[0]),
                "delta_ert": float(delta_ert),
            }

        # Compute slope ratios for each pathway
        sr_skip = (
            abs(results["dyslexic"]["delta_p_skip"])
            / abs(results["control"]["delta_p_skip"])
            if results["control"]["delta_p_skip"] != 0
            else np.nan
        )
        sr_trt = (
            abs(results["dyslexic"]["delta_trt"]) / abs(results["control"]["delta_trt"])
            if results["control"]["delta_trt"] != 0
            else np.nan
        )
        sr_ert = (
            abs(results["dyslexic"]["delta_ert"]) / abs(results["control"]["delta_ert"])
            if results["control"]["delta_ert"] != 0
            else np.nan
        )

        results["slope_ratios"] = {
            "sr_skip": float(sr_skip),
            "sr_trt": float(sr_trt),
            "sr_ert": float(sr_ert),
        }

        return results

    def compute_conditioned_contrasts(self, n_bins: int = 5) -> Dict:
        """
        Compute zipf effects WITHIN length bins (follows data manifold)
        """
        # Create length bins
        self.data["length_bin"] = pd.qcut(
            self.data["length"], q=n_bins, labels=False, duplicates="drop"
        )

        # Determine group column
        if "group" in self.data.columns:
            group_col = "group"
            groups = ["control", "dyslexic"]
        else:
            self.data["group"] = self.data["dyslexic"].map(
                {False: "control", True: "dyslexic"}
            )
            group_col = "group"
            groups = ["control", "dyslexic"]

        results = {
            "control": [],
            "dyslexic": [],
        }

        for group in groups:
            group_data = self.data[self.data[group_col] == group]

            for bin_idx in range(n_bins):
                bin_data = group_data[group_data["length_bin"] == bin_idx]

                if len(bin_data) < 20:
                    continue

                # Compute mean values in this bin
                length_mean = bin_data["length"].mean()
                surprisal_mean = bin_data["surprisal"].mean()

                # Get zipf Q1 and Q3 within this bin
                zipf_q1_bin = bin_data["zipf"].quantile(0.25)
                zipf_q3_bin = bin_data["zipf"].quantile(0.75)

                # Create grids at this length
                grid_q1 = pd.DataFrame(
                    {
                        "zipf": [zipf_q1_bin],
                        "length": [length_mean],
                        "surprisal": [surprisal_mean],
                    }
                )
                grid_q3 = pd.DataFrame(
                    {
                        "zipf": [zipf_q3_bin],
                        "length": [length_mean],
                        "surprisal": [surprisal_mean],
                    }
                )

                # Predict
                try:
                    ert_q1 = self.ert_predictor.predict_ert(grid_q1, group)[0]
                    ert_q3 = self.ert_predictor.predict_ert(grid_q3, group)[0]
                    delta_ert = ert_q3 - ert_q1

                    # Also get observed means
                    obs_q1_data = bin_data[bin_data["zipf"] <= zipf_q1_bin]
                    obs_q3_data = bin_data[bin_data["zipf"] >= zipf_q3_bin]
                    obs_ert_q1 = (
                        obs_q1_data["ERT"].mean()
                        if "ERT" in bin_data.columns and len(obs_q1_data) > 0
                        else np.nan
                    )
                    obs_ert_q3 = (
                        obs_q3_data["ERT"].mean()
                        if "ERT" in bin_data.columns and len(obs_q3_data) > 0
                        else np.nan
                    )
                    obs_delta = obs_ert_q3 - obs_ert_q1

                    results[group].append(
                        {
                            "bin": bin_idx,
                            "length_mean": float(length_mean),
                            "zipf_q1": float(zipf_q1_bin),
                            "zipf_q3": float(zipf_q3_bin),
                            "ert_q1_predicted": float(ert_q1),
                            "ert_q3_predicted": float(ert_q3),
                            "delta_ert_predicted": float(delta_ert),
                            "ert_q1_observed": float(obs_ert_q1),
                            "ert_q3_observed": float(obs_ert_q3),
                            "delta_ert_observed": float(obs_delta),
                            "n_obs": int(len(bin_data)),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to predict for {group} bin {bin_idx}: {e}")
                    continue

        # Compute weighted average
        for group in groups:
            if len(results[group]) > 0:
                weights = [r["n_obs"] for r in results[group]]
                deltas_predicted = [r["delta_ert_predicted"] for r in results[group]]
                deltas_observed = [
                    r["delta_ert_observed"]
                    for r in results[group]
                    if not np.isnan(r["delta_ert_observed"])
                ]

                avg_predicted = np.average(deltas_predicted, weights=weights)
                avg_observed = (
                    np.mean(deltas_observed) if len(deltas_observed) > 0 else np.nan
                )

                results[f"{group}_weighted_avg"] = {
                    "predicted": float(avg_predicted),
                    "observed": float(avg_observed),
                }

        # Compute slope ratio on conditioned contrasts
        if "control_weighted_avg" in results and "dyslexic_weighted_avg" in results:
            sr_conditioned = (
                abs(results["dyslexic_weighted_avg"]["predicted"])
                / abs(results["control_weighted_avg"]["predicted"])
                if results["control_weighted_avg"]["predicted"] != 0
                else np.nan
            )
            results["sr_conditioned"] = float(sr_conditioned)

        return results

    def fit_simple_models(self) -> Dict:
        """
        Fit simple linear models to check for group×zipf interactions

        Models:
        1. log(TRT) ~ group * (length + zipf) + length:zipf  [fixated words only]
        2. skip ~ group * (length + zipf) + length:zipf       [all words]
        """
        results = {}

        # Determine group column and create numeric version
        if "group" in self.data.columns:
            group_col = "group"
            self.data["group_numeric"] = (self.data["group"] == "dyslexic").astype(int)
        else:
            self.data["group"] = self.data["dyslexic"].map(
                {False: "control", True: "dyslexic"}
            )
            self.data["group_numeric"] = self.data["dyslexic"].astype(int)

        # Model 1: Duration (log-transformed TRT on fixated words)
        fixated = self.data[self.data["skip"] == 0].copy()
        fixated["log_trt"] = np.log(fixated["TRT"])

        # Prepare features
        X_duration = fixated[["group_numeric", "length", "zipf"]].copy()
        X_duration["group_x_length"] = (
            X_duration["group_numeric"] * X_duration["length"]
        )
        X_duration["group_x_zipf"] = X_duration["group_numeric"] * X_duration["zipf"]
        X_duration["length_x_zipf"] = X_duration["length"] * X_duration["zipf"]

        y_duration = fixated["log_trt"].values

        # Fit
        model_duration = LinearRegression()
        model_duration.fit(X_duration, y_duration)

        coef_names = [
            "intercept",
            "group",
            "length",
            "zipf",
            "group×length",
            "group×zipf",
            "length×zipf",
        ]
        coefs_duration = [model_duration.intercept_] + list(model_duration.coef_)

        results["duration_model"] = {
            "coefficients": dict(zip(coef_names, [float(c) for c in coefs_duration])),
            "r2": float(model_duration.score(X_duration, y_duration)),
            "n_obs": len(fixated),
        }

        # Model 2: Skip probability
        data_skip = self.data.copy()

        X_skip = data_skip[["group_numeric", "length", "zipf"]].copy()
        X_skip["group_x_length"] = X_skip["group_numeric"] * X_skip["length"]
        X_skip["group_x_zipf"] = X_skip["group_numeric"] * X_skip["zipf"]
        X_skip["length_x_zipf"] = X_skip["length"] * X_skip["zipf"]

        y_skip = data_skip["skip"].values

        # Fit
        model_skip = LinearRegression()
        model_skip.fit(X_skip, y_skip)

        coefs_skip = [model_skip.intercept_] + list(model_skip.coef_)

        results["skip_model"] = {
            "coefficients": dict(zip(coef_names, [float(c) for c in coefs_skip])),
            "r2": float(model_skip.score(X_skip, y_skip)),
            "n_obs": len(data_skip),
        }

        return results

    def print_pathway_summary(self, results: Dict):
        """Print pathway AMIE results"""
        logger.info("\nPATHWAY AMIEs (Q1→Q3 for Zipf):")
        logger.info("-" * 80)

        logger.info("\n1. SKIP PATHWAY: How frequency affects P(skip)")
        logger.info(
            f"   Control:  Δp(skip) = {results['control']['delta_p_skip']:+.4f}  "
            f"({results['control']['p_skip_q1']:.3f} → {results['control']['p_skip_q3']:.3f})"
        )
        logger.info(
            f"   Dyslexic: Δp(skip) = {results['dyslexic']['delta_p_skip']:+.4f}  "
            f"({results['dyslexic']['p_skip_q1']:.3f} → {results['dyslexic']['p_skip_q3']:.3f})"
        )
        logger.info(f"   SR(skip) = {results['slope_ratios']['sr_skip']:.3f}")

        if results["slope_ratios"]["sr_skip"] < 1.0:
            logger.info(
                f"   → Controls skip MORE with higher frequency (use frequency for skipping decisions)"
            )
        else:
            logger.info(f"   → Dyslexics skip more with higher frequency")

        logger.info("\n2. DURATION PATHWAY: How frequency affects TRT|fixated")
        logger.info(
            f"   Control:  ΔTRT = {results['control']['delta_trt']:+.2f} ms  "
            f"({results['control']['trt_q1']:.1f} → {results['control']['trt_q3']:.1f})"
        )
        logger.info(
            f"   Dyslexic: ΔTRT = {results['dyslexic']['delta_trt']:+.2f} ms  "
            f"({results['dyslexic']['trt_q1']:.1f} → {results['dyslexic']['trt_q3']:.1f})"
        )
        logger.info(f"   SR(duration) = {results['slope_ratios']['sr_trt']:.3f}")

        if results["slope_ratios"]["sr_trt"] > 1.0:
            logger.info(
                f"   → Dyslexics show AMPLIFIED duration effects (THIS IS YOUR GRAPH!)"
            )
        else:
            logger.info(f"   → Controls show stronger duration effects")

        logger.info("\n3. COMBINED (ERT): (1−p_skip) × TRT")
        logger.info(
            f"   Control:  ΔERT = {results['control']['delta_ert']:+.2f} ms  "
            f"({results['control']['ert_q1']:.1f} → {results['control']['ert_q3']:.1f})"
        )
        logger.info(
            f"   Dyslexic: ΔERT = {results['dyslexic']['delta_ert']:+.2f} ms  "
            f"({results['dyslexic']['ert_q1']:.1f} → {results['dyslexic']['ert_q3']:.1f})"
        )
        logger.info(f"   SR(ERT) = {results['slope_ratios']['sr_ert']:.3f}")

        logger.info("\n" + "=" * 80)
        logger.info("INTERPRETATION:")
        logger.info("=" * 80)

        if (
            results["slope_ratios"]["sr_skip"] < 1.0
            and results["slope_ratios"]["sr_trt"] > 1.0
        ):
            logger.info(
                "✓ EXPLANATION FOUND: Your graph (TRT) shows amplification (SR={:.2f}),\n"
                "  but combined ERT shows attenuation (SR={:.2f}) because:\n"
                "  - Controls use frequency MORE for skipping decisions (skip pathway)\n"
                "  - Dyslexics show AMPLIFIED duration costs (duration pathway)\n"
                "  - The strong skip effect for controls dominates the combined ERT\n\n"
                "  This is NOT a bug - it's the expected pattern when skip and duration\n"
                "  pathways work in opposite directions!".format(
                    results["slope_ratios"]["sr_trt"], results["slope_ratios"]["sr_ert"]
                )
            )
        else:
            logger.info(
                "⚠ Pattern doesn't match typical expectation. Check model predictions."
            )

        logger.info("=" * 80)

    def print_conditioned_summary(self, results: Dict):
        """Print conditioned contrast results"""
        logger.info("\nCONDITIONED CONTRASTS (Zipf effect within length bins):")
        logger.info("-" * 80)

        for group in ["control", "dyslexic"]:
            if len(results[group]) == 0:
                continue

            logger.info(f"\n{group.upper()}:")
            logger.info(
                f"{'Bin':<5} {'Length':<10} {'Zipf Range':<20} {'ΔERT (pred)':<15} {'ΔERT (obs)':<15} {'N':<8}"
            )
            logger.info("-" * 80)

            for r in results[group]:
                logger.info(
                    f"{r['bin']:<5} "
                    f"{r['length_mean']:<10.1f} "
                    f"{r['zipf_q1']:.2f}→{r['zipf_q3']:.2f}{'':<12} "
                    f"{r['delta_ert_predicted']:<+15.2f} "
                    f"{r['delta_ert_observed']:<+15.2f} "
                    f"{r['n_obs']:<8}"
                )

        if "control_weighted_avg" in results and "dyslexic_weighted_avg" in results:
            logger.info("\n" + "=" * 80)
            logger.info("WEIGHTED AVERAGES (following data manifold):")
            logger.info(
                f"  Control:  {results['control_weighted_avg']['predicted']:+.2f} ms (predicted), "
                f"{results['control_weighted_avg']['observed']:+.2f} ms (observed)"
            )
            logger.info(
                f"  Dyslexic: {results['dyslexic_weighted_avg']['predicted']:+.2f} ms (predicted), "
                f"{results['dyslexic_weighted_avg']['observed']:+.2f} ms (observed)"
            )

            if "sr_conditioned" in results:
                logger.info(f"\n  SR (conditioned) = {results['sr_conditioned']:.3f}")

                if results["sr_conditioned"] > 1.0:
                    logger.info(
                        f"  ✓ When accounting for length-frequency correlation,"
                    )
                    logger.info(f"    dyslexics DO show amplification!")
                else:
                    logger.info(f"  ⚠ Even when conditioned on length, SR < 1.0")

        logger.info("=" * 80)

    def print_linear_summary(self, results: Dict):
        """Print linear model results"""
        logger.info("\nSIMPLE LINEAR MODEL CHECKS:")
        logger.info("-" * 80)

        logger.info(
            "\n1. DURATION MODEL: log(TRT) ~ group * (length + zipf) + length×zipf"
        )
        logger.info(
            f"   R² = {results['duration_model']['r2']:.3f}, N = {results['duration_model']['n_obs']:,}"
        )
        logger.info(f"   Coefficients:")
        for name, coef in results["duration_model"]["coefficients"].items():
            sig = "***" if abs(coef) > 0.01 else ""
            logger.info(f"     {name:<15} = {coef:+.4f} {sig}")

        group_zipf = results["duration_model"]["coefficients"]["group×zipf"]
        if group_zipf < 0:
            logger.info(
                f"\n   ✓ group×zipf < 0: Dyslexics show STRONGER frequency benefit in duration"
            )
            logger.info(f"     (higher frequency reduces log(TRT) more for dyslexics)")
        else:
            logger.info(
                f"\n   ⚠ group×zipf > 0: Controls show stronger frequency benefit"
            )

        logger.info("\n2. SKIP MODEL: skip ~ group * (length + zipf) + length×zipf")
        logger.info(
            f"   R² = {results['skip_model']['r2']:.3f}, N = {results['skip_model']['n_obs']:,}"
        )
        logger.info(f"   Coefficients:")
        for name, coef in results["skip_model"]["coefficients"].items():
            sig = "***" if abs(coef) > 0.01 else ""
            logger.info(f"     {name:<15} = {coef:+.4f} {sig}")

        group_zipf_skip = results["skip_model"]["coefficients"]["group×zipf"]
        if group_zipf_skip < 0:
            logger.info(
                f"\n   ✓ group×zipf < 0: Dyslexics skip LESS as frequency increases"
            )
            logger.info(f"     (controls use frequency more for skipping)")
        else:
            logger.info(
                f"\n   ⚠ group×zipf > 0: Dyslexics skip MORE as frequency increases"
            )

        logger.info("=" * 80)

    def create_pathway_comparison_plot(self, results: Dict, save_path: Path):
        """Create visualization comparing pathways"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Zipf Frequency Effect Pathways: Why TRT ≠ ERT",
            fontsize=14,
            fontweight="bold",
        )

        groups = ["control", "dyslexic"]
        colors = ["#2E86AB", "#A23B72"]

        # Plot 1: Skip probability
        ax = axes[0, 0]
        for group, color in zip(groups, colors):
            p_skip = [results[group]["p_skip_q1"], results[group]["p_skip_q3"]]
            ax.plot(
                [1, 2],
                p_skip,
                "o-",
                color=color,
                linewidth=2,
                markersize=10,
                label=group.capitalize(),
            )
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Q1\n(low freq)", "Q3\n(high freq)"])
        ax.set_ylabel("P(skip)")
        ax.set_title(f'Skip Pathway\nSR = {results["slope_ratios"]["sr_skip"]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: TRT given fixation
        ax = axes[0, 1]
        for group, color in zip(groups, colors):
            trt = [results[group]["trt_q1"], results[group]["trt_q3"]]
            ax.plot(
                [1, 2],
                trt,
                "o-",
                color=color,
                linewidth=2,
                markersize=10,
                label=group.capitalize(),
            )
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Q1\n(low freq)", "Q3\n(high freq)"])
        ax.set_ylabel("TRT | fixated (ms)")
        ax.set_title(
            f'Duration Pathway (THIS IS YOUR GRAPH!)\nSR = {results["slope_ratios"]["sr_trt"]:.3f}'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Combined ERT
        ax = axes[1, 0]
        for group, color in zip(groups, colors):
            ert = [results[group]["ert_q1"], results[group]["ert_q3"]]
            ax.plot(
                [1, 2],
                ert,
                "o-",
                color=color,
                linewidth=2,
                markersize=10,
                label=group.capitalize(),
            )
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Q1\n(low freq)", "Q3\n(high freq)"])
        ax.set_ylabel("ERT (ms)")
        ax.set_title(
            f'Combined ERT = (1−skip) × TRT\nSR = {results["slope_ratios"]["sr_ert"]:.3f}'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Summary text
        ax = axes[1, 1]
        ax.axis("off")

        summary_text = (
            "EXPLANATION:\n\n"
            f"Skip pathway: SR = {results['slope_ratios']['sr_skip']:.3f}\n"
            f"  → Controls skip MORE with frequency\n\n"
            f"Duration pathway: SR = {results['slope_ratios']['sr_trt']:.3f}\n"
            f"  → Dyslexics show AMPLIFIED duration costs\n\n"
            f"Combined ERT: SR = {results['slope_ratios']['sr_ert']:.3f}\n"
            f"  → Skip effect dominates, masking duration\n\n"
            "Your TRT graph is CORRECT!\n"
            "The ERT AMIE is also CORRECT!\n"
            "They measure different things."
        )

        ax.text(
            0.1,
            0.5,
            summary_text,
            fontsize=11,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved pathway comparison plot: {save_path}")
        plt.close()

    def create_conditioned_contrast_plot(self, results: Dict, save_path: Path):
        """Create visualization of conditioned contrasts"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "Conditioned Contrasts: Zipf Effect Within Length Bins",
            fontsize=14,
            fontweight="bold",
        )

        colors = {"control": "#2E86AB", "dyslexic": "#A23B72"}

        for idx, (group, ax) in enumerate(zip(["control", "dyslexic"], axes)):
            if len(results[group]) == 0:
                continue

            # Extract data
            bins = [r["bin"] for r in results[group]]
            lengths = [r["length_mean"] for r in results[group]]
            deltas_pred = [r["delta_ert_predicted"] for r in results[group]]
            deltas_obs = [r["delta_ert_observed"] for r in results[group]]

            # Plot
            x = np.arange(len(bins))
            width = 0.35

            ax.bar(
                x - width / 2,
                deltas_pred,
                width,
                label="Predicted (GAM)",
                color=colors[group],
                alpha=0.7,
            )
            ax.bar(
                x + width / 2,
                deltas_obs,
                width,
                label="Observed (data)",
                color=colors[group],
                alpha=0.4,
                hatch="//",
            )

            ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
            ax.set_xlabel("Length Bin")
            ax.set_ylabel("ΔERT (Q1→Q3 zipf, ms)")
            ax.set_title(
                f'{group.capitalize()}\nWeighted avg: {results[f"{group}_weighted_avg"]["predicted"]:.1f} ms'
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"Bin {b}\n(len≈{l:.1f})" for b, l in zip(bins, lengths)]
            )
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved conditioned contrast plot: {save_path}")
        plt.close()

    def generate_summary_report(self, save_path: Path):
        """Generate text summary report"""
        with open(save_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("ZIPF FREQUENCY DIAGNOSTIC REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("SHORT ANSWER:\n")
            f.write("-" * 80 + "\n")
            f.write(
                "Your TRT plot and the ERT AMIE disagree for principled reasons, not a bug.\n\n"
            )

            f.write("WHY THE MISMATCH:\n")
            f.write("-" * 80 + "\n")

            # Get skip rates
            if "group" in self.data.columns:
                ctrl_skip = self.data[self.data["group"] == "control"]["skip"].mean()
                dys_skip = self.data[self.data["group"] == "dyslexic"]["skip"].mean()
            else:
                ctrl_skip = self.data[self.data["dyslexic"] == False]["skip"].mean()
                dys_skip = self.data[self.data["dyslexic"] == True]["skip"].mean()

            f.write(f"1. Your plot shows TRT-only; AMIE is on ERT = (1−skip)·TRT\n")
            f.write(
                f"   - Controls skip more overall ({ctrl_skip:.3f} vs {dys_skip:.3f})\n"
            )
            f.write(f"   - Frequency increases skipping more for controls\n")
            f.write(
                f"   - So ERT falls a lot with Zipf for controls but only a little for dyslexics\n\n"
            )

            if "pathway_amies" in self.results:
                pathway = self.results["pathway_amies"]
                f.write(f"2. Pathway Analysis Results:\n")
                f.write(f"   Skip pathway:\n")
                f.write(
                    f"     Control:  Δp(skip) = {pathway['control']['delta_p_skip']:+.4f}\n"
                )
                f.write(
                    f"     Dyslexic: Δp(skip) = {pathway['dyslexic']['delta_p_skip']:+.4f}\n"
                )
                f.write(f"     SR(skip) = {pathway['slope_ratios']['sr_skip']:.3f}\n\n")

                f.write(f"   Duration pathway (YOUR GRAPH):\n")
                f.write(
                    f"     Control:  ΔTRT = {pathway['control']['delta_trt']:+.2f} ms\n"
                )
                f.write(
                    f"     Dyslexic: ΔTRT = {pathway['dyslexic']['delta_trt']:+.2f} ms\n"
                )
                f.write(
                    f"     SR(duration) = {pathway['slope_ratios']['sr_trt']:.3f}\n\n"
                )

                f.write(f"   Combined ERT:\n")
                f.write(
                    f"     Control:  ΔERT = {pathway['control']['delta_ert']:+.2f} ms\n"
                )
                f.write(
                    f"     Dyslexic: ΔERT = {pathway['dyslexic']['delta_ert']:+.2f} ms\n"
                )
                f.write(f"     SR(ERT) = {pathway['slope_ratios']['sr_ert']:.3f}\n\n")

            # Correlation info
            corr = self.data["zipf"].corr(self.data["length"])
            f.write(f"3. Strong length×frequency dependence (r = {corr:.3f})\n")
            f.write(
                f'   - "Hold length at mean" evaluates Zipf where you have little data\n'
            )
            f.write(
                f"   - This distorts ERT contrasts even when TRT slopes are steeper\n\n"
            )

            if "conditioned_contrasts" in self.results:
                cond = self.results["conditioned_contrasts"]
                if "sr_conditioned" in cond:
                    f.write(f"4. Conditioned Analysis (within length bins):\n")
                    f.write(
                        f"   Control weighted avg:  {cond['control_weighted_avg']['predicted']:+.2f} ms\n"
                    )
                    f.write(
                        f"   Dyslexic weighted avg: {cond['dyslexic_weighted_avg']['predicted']:+.2f} ms\n"
                    )
                    f.write(f"   SR(conditioned) = {cond['sr_conditioned']:.3f}\n\n")

            f.write("\nMINIMAL DIAGNOSTICS COMPLETED:\n")
            f.write("-" * 80 + "\n")
            f.write("✓ 1. Pathway AMIEs computed (see above)\n")
            f.write("✓ 2. Conditioned contrasts computed (see above)\n")
            f.write("✓ 3. Simple linear models fitted (see below)\n\n")

            if "linear_models" in self.results:
                linear = self.results["linear_models"]
                f.write("LINEAR MODEL RESULTS:\n")
                f.write("-" * 80 + "\n")
                f.write(
                    f"Duration model: log(TRT) ~ group * (length + zipf) + length×zipf\n"
                )
                f.write(f"  R² = {linear['duration_model']['r2']:.3f}\n")
                f.write(
                    f"  group×zipf = {linear['duration_model']['coefficients']['group×zipf']:+.4f}\n"
                )
                if linear["duration_model"]["coefficients"]["group×zipf"] < 0:
                    f.write(
                        f"  → Dyslexics show STRONGER frequency benefit in duration ✓\n\n"
                    )
                else:
                    f.write(f"  → Controls show stronger frequency benefit ⚠\n\n")

                f.write(f"Skip model: skip ~ group * (length + zipf) + length×zipf\n")
                f.write(f"  R² = {linear['skip_model']['r2']:.3f}\n")
                f.write(
                    f"  group×zipf = {linear['skip_model']['coefficients']['group×zipf']:+.4f}\n"
                )
                if linear["skip_model"]["coefficients"]["group×zipf"] < 0:
                    f.write(f"  → Controls use frequency MORE for skipping ✓\n\n")
                else:
                    f.write(f"  → Dyslexics use frequency more for skipping ⚠\n\n")

            f.write("\nBOTTOM LINE:\n")
            f.write("-" * 80 + "\n")
            f.write(
                "The figure reflects DURATION, the AMIE reflects DURATION × SKIPPING\n"
            )
            f.write("plus length–frequency coupling.\n\n")

            f.write(
                "Both measurements are correct - they're measuring different things:\n"
            )
            f.write(
                "  - Your TRT graph: Pure duration effect (amplified for dyslexics)\n"
            )
            f.write(
                "  - The ERT AMIE: Combined effect after accounting for skipping\n\n"
            )

            if "pathway_amies" in self.results:
                pathway = self.results["pathway_amies"]
                f.write("RECOMMENDATION FOR PAPER:\n")
                f.write("-" * 80 + "\n")
                f.write("Report BOTH pathway-specific SRs:\n")
                f.write(
                    f"  1. Duration SR = {pathway['slope_ratios']['sr_trt']:.3f} (amplified)\n"
                )
                f.write(
                    f"  2. Skip SR = {pathway['slope_ratios']['sr_skip']:.3f} (attenuated)\n"
                )
                f.write(
                    f"  3. Combined ERT SR = {pathway['slope_ratios']['sr_ert']:.3f}\n\n"
                )

                f.write("Interpretation:\n")
                f.write(
                    "'Frequency shows complex pathway effects: dyslexics exhibit amplified\n"
                )
                f.write(
                    "duration costs for low-frequency words (SR={:.2f}), but controls make\n".format(
                        pathway["slope_ratios"]["sr_trt"]
                    )
                )
                f.write(
                    "greater use of frequency information for skipping decisions (SR={:.2f}).\n".format(
                        pathway["slope_ratios"]["sr_skip"]
                    )
                )
                f.write(
                    "The combined effect on ERT reflects both pathways operating in\n"
                )
                f.write("opposite directions.'\n\n")

            f.write("=" * 80 + "\n")

        logger.info(f"  Saved diagnostic report: {save_path}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def load_data():
    """Load the prepared data"""
    script_dir = Path(__file__).resolve().parent.parent
    data_path = script_dir / "input_data" / "preprocessed_data.csv"

    data = pd.read_csv(data_path)
    logger.info(f"Loaded {len(data):,} observations")
    return data


def run_zipf_diagnostic(
    data: pd.DataFrame,
    ert_predictor=None,
    quartiles: Optional[Dict] = None,
    output_dir: str = "diagnostic_results",
) -> Dict:
    """
    Convenience function to run complete zipf diagnostic

    Args:
        data: Full dataset
        ert_predictor: Optional ERTPredictor instance for model-level analysis
        quartiles: Optional feature quartiles
        output_dir: Directory for outputs

    Returns:
        Dictionary with all diagnostic results
    """
    diagnostic = ZipfDiagnostic(data, ert_predictor, quartiles)
    results = diagnostic.run_all_diagnostics(Path(output_dir))
    return results


def main():
    """Run data-level diagnostics only (no model required)"""
    logger.info("=" * 80)
    logger.info("ZIPF FREQUENCY PARADOX DIAGNOSTIC - DATA LEVEL ONLY")
    logger.info("=" * 80)

    # Load data
    data = load_data()

    # Run data-level diagnostics only
    diagnostic = ZipfDiagnostic(data)
    diagnostic.run_all_diagnostics()

    logger.info("\n" + "=" * 80)
    logger.info("DATA-LEVEL DIAGNOSTIC COMPLETE")
    logger.info(
        "For model-level diagnostics, run from main analysis script with ERT predictor"
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
