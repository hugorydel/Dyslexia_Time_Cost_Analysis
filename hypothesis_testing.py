#!/usr/bin/env python3
"""
Hypothesis Testing for Dyslexia Time Cost Analysis
Uses Linear Mixed-Effects Models to test three main hypotheses
Outputs APA-7 formatted results with effect sizes
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM

logger = logging.getLogger(__name__)


class DyslexiaHypothesisTesting:
    """
    Hypothesis testing for dyslexia reading time effects
    Tests main effects, interactions, and variance decomposition
    """

    def __init__(self, data: pd.DataFrame, results_dir: Path):
        """
        Initialize hypothesis testing

        Args:
            data: DataFrame with word-level data including:
                - total_reading_time, first_fixation_duration, gaze_duration
                - word_length, word_frequency_zipf, surprisal
                - dyslexic (boolean), subject_id, sentence_id
                - was_fixated (to filter fixated words)
            results_dir: Directory to save results

        Note on Model Warnings:
            You may see warnings about "singular covariance" or "boundary" from statsmodels.
            These are common when random effects variance is very small and don't invalidate
            the results. For publication-quality analyses, consider using R's lme4 package
            via pymer4, which handles these cases more robustly.
        """
        self.data = data.copy()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Create subdirectories
        self.tables_dir = self.results_dir / "hypothesis_testing_tables"
        self.figures_dir = self.results_dir / "hypothesis_testing_figures"
        self.tables_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)

        # Results storage
        self.model_results = {}
        self.hypothesis_tests = {}

        logger.info("Hypothesis testing initialized")
        logger.info(
            f"Dataset: {len(self.data):,} words from {self.data['subject_id'].nunique()} subjects"
        )

    def prepare_data(self, outcomes: List[str] = None) -> pd.DataFrame:
        """
        Prepare data for hypothesis testing

        Args:
            outcomes: List of outcome variables to test

        Returns:
            Prepared DataFrame with centered predictors and complete cases
        """
        logger.info("Preparing data for hypothesis testing...")

        if outcomes is None:
            outcomes = [
                "total_reading_time",
                "first_fixation_duration",
                "gaze_duration",
            ]

        # Start with fixated words only for reading time measures
        data = self.data[self.data["was_fixated"] == True].copy()
        logger.info(f"Starting with {len(data):,} fixated words")

        # Remove rows with missing linguistic features
        required_cols = [
            "word_length",
            "word_frequency_zipf",
            "surprisal",
            "subject_id",
        ]
        required_cols.extend(outcomes)

        initial_count = len(data)
        data = data.dropna(subset=required_cols)
        final_count = len(data)

        logger.info(
            f"Removed {initial_count - final_count:,} words with missing values"
        )
        logger.info(
            f"Final sample: {final_count:,} words ({(final_count/initial_count)*100:.1f}% retention)"
        )

        # Create numeric group variable (0=control, 1=dyslexic)
        data["group"] = data["dyslexic"].astype(int)

        # Center continuous predictors for interpretability
        # This makes the intercept interpretable as the mean at average predictor values
        logger.info("Centering predictors...")

        data["length_c"] = data["word_length"] - data["word_length"].mean()
        data["frequency_c"] = (
            data["word_frequency_zipf"] - data["word_frequency_zipf"].mean()
        )
        data["surprisal_c"] = data["surprisal"] - data["surprisal"].mean()

        # Log-transform reading times to reduce skew (common in RT analysis)
        for outcome in outcomes:
            if outcome in data.columns:
                # Add small constant to avoid log(0), then log transform
                data[f"log_{outcome}"] = np.log(data[outcome] + 1)

        # Report centering statistics
        logger.info(f"Predictor means after centering:")
        logger.info(f"  Length: {data['length_c'].mean():.6f} (should be ~0)")
        logger.info(f"  Frequency: {data['frequency_c'].mean():.6f} (should be ~0)")
        logger.info(f"  Surprisal: {data['surprisal_c'].mean():.6f} (should be ~0)")

        # Group distribution
        group_counts = data.groupby("group").size()
        logger.info(f"Group distribution:")
        logger.info(f"  Control (0): {group_counts.get(0, 0):,} words")
        logger.info(f"  Dyslexic (1): {group_counts.get(1, 0):,} words")

        self.prepared_data = data
        return data

    def test_hypothesis_1(self, outcome: str = "log_total_reading_time") -> Dict:
        """
        Hypothesis 1: Feature Effects
        Test main effects of Length, Frequency, and Surprisal on reading measures

        Expected directions:
        - Length ↑ → Reading Time ↑
        - Frequency ↑ → Reading Time ↓
        - Surprisal ↑ → Reading Time ↑

        Args:
            outcome: Outcome variable (e.g., 'log_total_reading_time')

        Returns:
            Dictionary with model results and formatted output
        """
        logger.info("=" * 70)
        logger.info("HYPOTHESIS 1: Testing Feature Effects")
        logger.info("=" * 70)

        data = self.prepared_data

        # Model formula: outcome ~ length + frequency + surprisal + (1|subject)
        formula = f"{outcome} ~ length_c + frequency_c + surprisal_c"

        logger.info(f"Fitting model: {formula}")
        logger.info(f"Random effects: (1 | subject_id)")

        try:
            # Fit mixed model with random intercepts for subjects
            model = smf.mixedlm(formula, data, groups=data["subject_id"])
            result = model.fit(method="lbfgs", reml=True)

            logger.info("Model converged successfully")

            # Extract results
            h1_results = self._format_model_results(
                result, "Hypothesis 1: Feature Effects"
            )

            # Test specific hypotheses about directions
            params = result.params
            pvalues = result.pvalues

            # Check if effects are in expected directions
            length_positive = params["length_c"] > 0 and pvalues["length_c"] < 0.05
            frequency_negative = (
                params["frequency_c"] < 0 and pvalues["frequency_c"] < 0.05
            )
            surprisal_positive = (
                params["surprisal_c"] > 0 and pvalues["surprisal_c"] < 0.05
            )

            h1_results["hypothesis_support"] = {
                "length_positive": length_positive,
                "frequency_negative": frequency_negative,
                "surprisal_positive": surprisal_positive,
                "all_supported": length_positive
                and frequency_negative
                and surprisal_positive,
            }

            # Store results
            self.model_results["hypothesis_1"] = result
            self.hypothesis_tests["hypothesis_1"] = h1_results

            # Print summary (using ASCII for Windows console compatibility)
            logger.info("\nHypothesis 1 Results Summary:")
            logger.info(
                f"  Length effect: b={params['length_c']:.4f}, p={pvalues['length_c']:.4f} {'PASS' if length_positive else 'FAIL'}"
            )
            logger.info(
                f"  Frequency effect: b={params['frequency_c']:.4f}, p={pvalues['frequency_c']:.4f} {'PASS' if frequency_negative else 'FAIL'}"
            )
            logger.info(
                f"  Surprisal effect: b={params['surprisal_c']:.4f}, p={pvalues['surprisal_c']:.4f} {'PASS' if surprisal_positive else 'FAIL'}"
            )

            return h1_results

        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            raise

    def test_hypothesis_2(self, outcome: str = "log_total_reading_time") -> Dict:
        """
        Hypothesis 2: Dyslexic Amplification
        Test whether dyslexic readers show steeper effects via Group × Predictor interactions

        Expected interactions:
        - Group × Length: Dyslexics show steeper length costs
        - Group × Frequency: Dyslexics show steeper frequency effects
        - Group × Surprisal: Dyslexics show larger surprisal costs

        Args:
            outcome: Outcome variable

        Returns:
            Dictionary with interaction test results
        """
        logger.info("=" * 70)
        logger.info("HYPOTHESIS 2: Testing Dyslexic Amplification")
        logger.info("=" * 70)

        data = self.prepared_data

        # Full interaction model
        formula = f"{outcome} ~ group * (length_c + frequency_c + surprisal_c)"

        logger.info(f"Fitting model: {formula}")
        logger.info(f"Random effects: (1 | subject_id)")

        try:
            model = smf.mixedlm(formula, data, groups=data["subject_id"])
            result = model.fit(method="lbfgs", reml=True)

            logger.info("Model converged successfully")

            # Extract results
            h2_results = self._format_model_results(
                result, "Hypothesis 2: Dyslexic Amplification"
            )

            # Test interaction terms
            params = result.params
            pvalues = result.pvalues

            # Check for significant interactions in expected directions
            interactions = {}

            if "group:length_c" in params:
                interactions["length"] = {
                    "beta": float(params["group:length_c"]),
                    "p": float(pvalues["group:length_c"]),
                    "significant": pvalues["group:length_c"] < 0.05,
                    "expected_direction": params["group:length_c"]
                    > 0,  # Positive = steeper for dyslexics
                }

            if "group:frequency_c" in params:
                interactions["frequency"] = {
                    "beta": float(params["group:frequency_c"]),
                    "p": float(pvalues["group:frequency_c"]),
                    "significant": pvalues["group:frequency_c"] < 0.05,
                    "expected_direction": params["group:frequency_c"]
                    < 0,  # More negative = steeper for dyslexics
                }

            if "group:surprisal_c" in params:
                interactions["surprisal"] = {
                    "beta": float(params["group:surprisal_c"]),
                    "p": float(pvalues["group:surprisal_c"]),
                    "significant": pvalues["group:surprisal_c"] < 0.05,
                    "expected_direction": params["group:surprisal_c"]
                    > 0,  # Positive = steeper for dyslexics
                }

            h2_results["interactions"] = interactions

            # Overall support for hypothesis
            any_significant = any(
                inter["significant"] for inter in interactions.values()
            )
            h2_results["hypothesis_support"] = {
                "any_interaction_significant": any_significant,
                "interactions_tested": len(interactions),
            }

            # Store results
            self.model_results["hypothesis_2"] = result
            self.hypothesis_tests["hypothesis_2"] = h2_results

            # Print summary (using ASCII for Windows console compatibility)
            logger.info("\nHypothesis 2 Results Summary:")
            for name, inter in interactions.items():
                logger.info(
                    f"  Group x {name.capitalize()}: b={inter['beta']:.4f}, p={inter['p']:.4f} {'PASS' if inter['significant'] else 'FAIL'}"
                )

            return h2_results

        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            raise

    def test_hypothesis_3(self, outcome: str = "log_total_reading_time") -> Dict:
        """
        Hypothesis 3: Variance Decomposition
        Compare baseline vs full model to quantify how much linguistic features
        explain the dyslexia–control gap

        Models:
        - Baseline: outcome ~ group + (1|subject)
        - Full: outcome ~ group * (length + frequency + surprisal) + (1|subject)

        Args:
            outcome: Outcome variable

        Returns:
            Dictionary with model comparison results
        """
        logger.info("=" * 70)
        logger.info("HYPOTHESIS 3: Variance Decomposition")
        logger.info("=" * 70)

        data = self.prepared_data

        # Baseline model: just group difference
        formula_baseline = f"{outcome} ~ group"
        logger.info(f"Baseline model: {formula_baseline}")

        model_baseline = smf.mixedlm(formula_baseline, data, groups=data["subject_id"])
        result_baseline = model_baseline.fit(
            method="lbfgs", reml=False
        )  # ML for comparison

        # Full model: all predictors and interactions
        formula_full = f"{outcome} ~ group * (length_c + frequency_c + surprisal_c)"
        logger.info(f"Full model: {formula_full}")

        model_full = smf.mixedlm(formula_full, data, groups=data["subject_id"])
        result_full = model_full.fit(method="lbfgs", reml=False)  # ML for comparison

        logger.info("Both models converged")

        # Calculate variance explained
        # For mixed models, we use conditional and marginal R²
        # Marginal R² = variance explained by fixed effects
        # Conditional R² = variance explained by fixed + random effects

        # Likelihood ratio test
        lr_statistic = 2 * (result_full.llf - result_baseline.llf)
        df_diff = len(result_full.params) - len(result_baseline.params)
        lr_pvalue = stats.chi2.sf(lr_statistic, df_diff)

        # AIC and BIC comparison
        aic_baseline = result_baseline.aic
        aic_full = result_full.aic
        bic_baseline = result_baseline.bic
        bic_full = result_full.bic

        # Calculate pseudo-R² (proportion of variance explained)
        # Using the formula: R² = 1 - (residual_variance_full / residual_variance_baseline)
        var_baseline = result_baseline.scale
        var_full = result_full.scale
        pseudo_r2 = 1 - (var_full / var_baseline)

        # Calculate effect of group in each model
        group_effect_baseline = result_baseline.params.get("group", 0)
        group_effect_full = result_full.params.get("group", 0)
        group_effect_reduction = abs(group_effect_baseline) - abs(group_effect_full)
        pct_reduction = (
            (group_effect_reduction / abs(group_effect_baseline)) * 100
            if group_effect_baseline != 0
            else 0
        )

        h3_results = {
            "baseline_model": {
                "formula": formula_baseline,
                "aic": float(aic_baseline),
                "bic": float(bic_baseline),
                "loglikelihood": float(result_baseline.llf),
                "group_effect": float(group_effect_baseline),
                "residual_variance": float(var_baseline),
            },
            "full_model": {
                "formula": formula_full,
                "aic": float(aic_full),
                "bic": float(bic_full),
                "loglikelihood": float(result_full.llf),
                "group_effect": float(group_effect_full),
                "residual_variance": float(var_full),
            },
            "model_comparison": {
                "lr_statistic": float(lr_statistic),
                "df": int(df_diff),
                "p_value": float(lr_pvalue),
                "pseudo_r2": float(pseudo_r2),
                "aic_improvement": float(aic_baseline - aic_full),
                "bic_improvement": float(bic_baseline - bic_full),
            },
            "group_effect_decomposition": {
                "baseline_group_effect": float(group_effect_baseline),
                "full_model_group_effect": float(group_effect_full),
                "reduction": float(group_effect_reduction),
                "percent_reduction": float(pct_reduction),
            },
            "hypothesis_support": {
                "full_model_preferred": lr_pvalue < 0.05,
                "variance_explained": pseudo_r2 > 0.10,  # Arbitrary threshold
            },
        }

        # Store results
        self.model_results["hypothesis_3_baseline"] = result_baseline
        self.model_results["hypothesis_3_full"] = result_full
        self.hypothesis_tests["hypothesis_3"] = h3_results

        # Print summary (using ASCII for Windows console compatibility)
        logger.info("\nHypothesis 3 Results Summary:")
        logger.info(
            f"  Likelihood Ratio Test: X2({df_diff}) = {lr_statistic:.2f}, p = {lr_pvalue:.4f}"
        )
        logger.info(f"  Pseudo-R2: {pseudo_r2:.4f}")
        logger.info(f"  AIC improvement: {aic_baseline - aic_full:.2f}")
        logger.info(f"  Group effect reduction: {pct_reduction:.1f}%")
        logger.info(f"  Full model preferred: {'Yes' if lr_pvalue < 0.05 else 'No'}")

        return h3_results

    def _format_model_results(self, result, title: str) -> Dict:
        """
        Format model results in APA-7 style

        Args:
            result: Fitted statsmodels result
            title: Title for the results

        Returns:
            Dictionary with formatted results
        """
        # Extract parameter estimates
        params = result.params
        stderr = result.bse
        tvalues = result.tvalues
        pvalues = result.pvalues

        # Calculate 95% confidence intervals
        conf_int = result.conf_int()

        # Format results for each parameter
        formatted_params = {}
        for param_name in params.index:
            formatted_params[param_name] = {
                "beta": float(params[param_name]),
                "se": float(stderr[param_name]),
                "t": float(tvalues[param_name]),
                "p": float(pvalues[param_name]),
                "ci_lower": float(conf_int.loc[param_name, 0]),
                "ci_upper": float(conf_int.loc[param_name, 1]),
                "apa_format": self._format_apa_parameter(
                    params[param_name],
                    stderr[param_name],
                    tvalues[param_name],
                    pvalues[param_name],
                    conf_int.loc[param_name, 0],
                    conf_int.loc[param_name, 1],
                ),
            }

        # Model fit statistics
        model_fit = {
            "aic": float(result.aic),
            "bic": float(result.bic),
            "loglikelihood": float(result.llf),
            "n_obs": int(result.nobs),
            "n_groups": int(result.n_groups) if hasattr(result, "n_groups") else None,
            "residual_variance": float(result.scale),
        }

        return {
            "title": title,
            "parameters": formatted_params,
            "model_fit": model_fit,
        }

    def _format_apa_parameter(
        self,
        beta: float,
        se: float,
        t: float,
        p: float,
        ci_lower: float,
        ci_upper: float,
    ) -> str:
        """
        Format parameter in APA-7 style

        Example: β = 0.15, SE = 0.03, t = 5.00, p < .001, 95% CI [0.09, 0.21]
        """
        # Format p-value
        if p < 0.001:
            p_str = "p < .001"
        else:
            p_str = f"p = {p:.3f}".replace("0.", ".")

        # Format the result string
        apa_str = (
            f"β = {beta:.3f}, SE = {se:.3f}, t = {t:.2f}, {p_str}, "
            f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}]"
        )

        return apa_str

    def create_apa_tables(self):
        """
        Create APA-7 formatted tables for all hypotheses
        Saves to CSV files in tables directory
        """
        logger.info("Creating APA-7 formatted tables...")

        for hypothesis, results in self.hypothesis_tests.items():
            if "parameters" in results:
                # Create parameter table
                param_table = self._create_parameter_table(results["parameters"])

                # Save to CSV
                output_file = self.tables_dir / f"{hypothesis}_parameters.csv"
                param_table.to_csv(output_file, index=False)
                logger.info(f"Saved table: {output_file}")

        # Create model comparison table for Hypothesis 3
        if "hypothesis_3" in self.hypothesis_tests:
            comp_table = self._create_comparison_table()
            output_file = self.tables_dir / "hypothesis_3_model_comparison.csv"
            comp_table.to_csv(output_file, index=False)
            logger.info(f"Saved table: {output_file}")

    def create_visualizations(self):
        """
        Create visualizations for hypothesis testing results
        """
        logger.info("Creating visualizations...")

        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import seaborn as sns

            sns.set_style("whitegrid")

            # Create interaction plots for Hypothesis 2
            if "hypothesis_2" in self.model_results:
                self._create_interaction_plots()

            # Create effect size visualization
            self._create_effect_size_plot()

            logger.info("Visualizations complete")

        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")

    def _create_interaction_plots(self):
        """Create interaction plots showing Group x Predictor effects"""
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        logger.info("Creating interaction plots...")

        data = self.prepared_data
        h2_params = self.hypothesis_tests["hypothesis_2"]["parameters"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            "Group × Predictor Interactions (Hypothesis 2)",
            fontsize=16,
            fontweight="bold",
        )

        predictors = [
            ("length_c", "Word Length (centered)", "Length"),
            ("frequency_c", "Word Frequency (centered)", "Frequency"),
            ("surprisal_c", "Surprisal (centered)", "Surprisal"),
        ]

        for idx, (pred_name, pred_label, short_name) in enumerate(predictors):
            ax = axes[idx]

            # Create bins for the predictor
            pred_values = np.percentile(data[pred_name], [10, 30, 50, 70, 90])
            bin_labels = ["Very Low", "Low", "Medium", "High", "Very High"]

            # Calculate predicted values for each group
            intercept = h2_params["Intercept"]["beta"]
            group_effect = h2_params["group"]["beta"]
            main_effect = h2_params[pred_name]["beta"]
            interaction_key = f"group:{pred_name}"
            interaction_effect = h2_params.get(interaction_key, {}).get("beta", 0)

            # Control group (group=0)
            control_predictions = intercept + main_effect * pred_values

            # Dyslexic group (group=1)
            dyslexic_predictions = (
                intercept
                + group_effect
                + (main_effect + interaction_effect) * pred_values
            )

            # Plot
            ax.plot(
                pred_values,
                control_predictions,
                marker="o",
                linewidth=2.5,
                label="Control",
                color="#2E86AB",
            )
            ax.plot(
                pred_values,
                dyslexic_predictions,
                marker="s",
                linewidth=2.5,
                label="Dyslexic",
                color="#A23B72",
            )

            # Add interaction significance indicator
            if interaction_key in h2_params:
                p_val = h2_params[interaction_key]["p"]
                sig_marker = (
                    "***"
                    if p_val < 0.001
                    else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                )
                ax.text(
                    0.5,
                    0.95,
                    f"Interaction: {sig_marker}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

            ax.set_xlabel(pred_label, fontsize=11)
            ax.set_ylabel("log(Total Reading Time)", fontsize=11)
            ax.set_title(f"Group × {short_name}", fontsize=12, fontweight="bold")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.figures_dir / "interaction_plots.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved interaction plots: {output_file}")

    def _create_effect_size_plot(self):
        """Create visualization of effect sizes across hypotheses"""
        import matplotlib.pyplot as plt
        import numpy as np

        logger.info("Creating effect size visualization...")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Gather coefficients from H1 and H2
        h1_params = self.hypothesis_tests["hypothesis_1"]["parameters"]
        h2_params = self.hypothesis_tests["hypothesis_2"]["parameters"]

        effects = []

        # Main effects (H1)
        for param in ["length_c", "frequency_c", "surprisal_c"]:
            if param in h1_params:
                effects.append(
                    {
                        "name": f"{param.split('_')[0].capitalize()} (main)",
                        "beta": h1_params[param]["beta"],
                        "ci_lower": h1_params[param]["ci_lower"],
                        "ci_upper": h1_params[param]["ci_upper"],
                        "significant": h1_params[param]["p"] < 0.05,
                    }
                )

        # Interactions (H2)
        for param in ["group:length_c", "group:frequency_c", "group:surprisal_c"]:
            if param in h2_params:
                name = param.split(":")[1].split("_")[0].capitalize()
                effects.append(
                    {
                        "name": f"Group × {name}",
                        "beta": h2_params[param]["beta"],
                        "ci_lower": h2_params[param]["ci_lower"],
                        "ci_upper": h2_params[param]["ci_upper"],
                        "significant": h2_params[param]["p"] < 0.05,
                    }
                )

        # Sort by absolute effect size
        effects.sort(key=lambda x: abs(x["beta"]), reverse=True)

        # Plot
        y_pos = np.arange(len(effects))
        colors = ["#2E86AB" if e["significant"] else "#CCCCCC" for e in effects]

        ax.barh(y_pos, [e["beta"] for e in effects], color=colors, alpha=0.7)

        # Add error bars
        xerr = [
            [e["beta"] - e["ci_lower"] for e in effects],
            [e["ci_upper"] - e["beta"] for e in effects],
        ]
        ax.errorbar(
            [e["beta"] for e in effects],
            y_pos,
            xerr=xerr,
            fmt="none",
            ecolor="black",
            capsize=5,
            alpha=0.8,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([e["name"] for e in effects])
        ax.set_xlabel("Coefficient (beta)", fontsize=12)
        ax.set_title(
            "Effect Sizes with 95% Confidence Intervals", fontsize=14, fontweight="bold"
        )
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax.grid(axis="x", alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#2E86AB", alpha=0.7, label="Significant (p < .05)"),
            Patch(facecolor="#CCCCCC", alpha=0.7, label="Not Significant"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        output_file = self.figures_dir / "effect_sizes.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved effect size plot: {output_file}")

    def _create_parameter_table(self, parameters: Dict) -> pd.DataFrame:
        """Create APA-formatted parameter table"""
        rows = []
        for param_name, values in parameters.items():
            rows.append(
                {
                    "Parameter": param_name,
                    "β": f"{values['beta']:.3f}",
                    "SE": f"{values['se']:.3f}",
                    "t": f"{values['t']:.2f}",
                    "p": f"{values['p']:.3f}" if values["p"] >= 0.001 else "< .001",
                    "95% CI": f"[{values['ci_lower']:.3f}, {values['ci_upper']:.3f}]",
                }
            )
        return pd.DataFrame(rows)

    def _create_comparison_table(self) -> pd.DataFrame:
        """Create model comparison table for Hypothesis 3"""
        h3 = self.hypothesis_tests["hypothesis_3"]

        rows = [
            {
                "Model": "Baseline",
                "Formula": h3["baseline_model"]["formula"],
                "AIC": f"{h3['baseline_model']['aic']:.2f}",
                "BIC": f"{h3['baseline_model']['bic']:.2f}",
                "LogLik": f"{h3['baseline_model']['loglikelihood']:.2f}",
            },
            {
                "Model": "Full",
                "Formula": h3["full_model"]["formula"],
                "AIC": f"{h3['full_model']['aic']:.2f}",
                "BIC": f"{h3['full_model']['bic']:.2f}",
                "LogLik": f"{h3['full_model']['loglikelihood']:.2f}",
            },
            {
                "Model": "Comparison",
                "Formula": "LRT",
                "AIC": f"Δ = {h3['model_comparison']['aic_improvement']:.2f}",
                "BIC": f"Δ = {h3['model_comparison']['bic_improvement']:.2f}",
                "LogLik": f"χ²({h3['model_comparison']['df']}) = {h3['model_comparison']['lr_statistic']:.2f}, "
                + f"p = {h3['model_comparison']['p_value']:.3f}",
            },
        ]
        return pd.DataFrame(rows)

    def run_all_tests(self, outcome: str = "log_total_reading_time") -> Dict:
        """
        Run all three hypothesis tests

        Args:
            outcome: Outcome variable to test (default: log-transformed total reading time)

        Returns:
            Dictionary with all results
        """
        logger.info("=" * 70)
        logger.info("RUNNING ALL HYPOTHESIS TESTS")
        logger.info("=" * 70)

        # Prepare data
        self.prepare_data()

        # Test all hypotheses
        h1_results = self.test_hypothesis_1(outcome)
        h2_results = self.test_hypothesis_2(outcome)
        h3_results = self.test_hypothesis_3(outcome)

        # Create APA tables
        self.create_apa_tables()

        # Create visualizations
        self.create_visualizations()

        # Create summary report
        summary = {
            "outcome_variable": outcome,
            "n_observations": len(self.prepared_data),
            "n_subjects": self.prepared_data["subject_id"].nunique(),
            "hypothesis_1": h1_results,
            "hypothesis_2": h2_results,
            "hypothesis_3": h3_results,
        }

        # Save summary to JSON
        import json

        output_file = self.results_dir / "hypothesis_testing_summary.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"\nAll results saved to: {self.results_dir}")

        return summary


def run_hypothesis_testing(data: pd.DataFrame, results_dir: Path) -> Dict:
    """
    Main entry point for hypothesis testing

    Args:
        data: Prepared word-level data
        results_dir: Directory to save results

    Returns:
        Dictionary with all test results
    """
    tester = DyslexiaHypothesisTesting(data, results_dir)
    results = tester.run_all_tests(outcome="log_total_reading_time")

    # Print final summary
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTING COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {results_dir}")
    print(f"\nTables:")
    print(f"  - {results_dir}/hypothesis_testing_tables/")
    print(f"\nFigures:")
    print(f"  - {results_dir}/hypothesis_testing_figures/")
    print(f"\nSummary:")
    print(f"  - {results_dir}/hypothesis_testing_summary.json")

    # Print key findings
    h1 = results["hypothesis_1"]
    h2 = results["hypothesis_2"]
    h3 = results["hypothesis_3"]

    print(f"\n{'KEY FINDINGS':^70}")
    print("=" * 70)

    print("\n[PASS] Hypothesis 1 (Feature Effects):")
    if h1["hypothesis_support"]["all_supported"]:
        print("  All expected effects confirmed!")
    else:
        print("  Partial support:")
        print(
            f"    Length: {'PASS' if h1['hypothesis_support']['length_positive'] else 'FAIL'}"
        )
        print(
            f"    Frequency: {'PASS' if h1['hypothesis_support']['frequency_negative'] else 'FAIL'}"
        )
        print(
            f"    Surprisal: {'PASS' if h1['hypothesis_support']['surprisal_positive'] else 'FAIL'}"
        )

    print("\n[PASS] Hypothesis 2 (Dyslexic Amplification):")
    if h2["hypothesis_support"]["any_interaction_significant"]:
        print("  Significant interactions found:")
        for name, inter in h2["interactions"].items():
            if inter["significant"]:
                print(
                    f"    Group x {name.capitalize()}: b={inter['beta']:.3f}, p={inter['p']:.4f}"
                )
    else:
        print("  No significant interactions detected")

    print("\n[PASS] Hypothesis 3 (Variance Decomposition):")
    print(f"  Variance explained: {h3['model_comparison']['pseudo_r2']:.1%}")
    print(
        f"  Group effect reduced by: {h3['group_effect_decomposition']['percent_reduction']:.1f}%"
    )
    print(
        f"  Model comparison: X2({h3['model_comparison']['df']}) = {h3['model_comparison']['lr_statistic']:.2f}, "
        f"p = {h3['model_comparison']['p_value']:.4f}"
    )

    print("\n" + "=" * 70)

    return results
