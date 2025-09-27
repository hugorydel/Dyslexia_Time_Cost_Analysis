#!/usr/bin/env python3
"""
Comprehensive Statistical Modeling for Dyslexia Time Cost Analysis
Mixed-effects models and hypothesis testing for the four key predictors
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Try to import pymer4 for advanced mixed effects
try:
    from pymer4.models import Lmer

    HAS_PYMER4 = True
except ImportError:
    HAS_PYMER4 = False
    warnings.warn("pymer4 not available. Using statsmodels for mixed effects.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DyslexiaStatisticalAnalyzer:
    """
    Comprehensive statistical analysis for dyslexia reading time costs
    """

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.results = {}

    def run_comprehensive_analysis(self, data: pd.DataFrame) -> Dict:
        """
        Run the complete statistical analysis pipeline

        Parameters:
        -----------
        data : DataFrame with features and eye-tracking measures

        Returns:
        --------
        Dict containing all analysis results
        """
        logger.info("Starting comprehensive statistical analysis...")

        # Prepare data
        analysis_data = self._prepare_analysis_data(data)

        # 1. Hypothesis 1: Feature effects on eye measures
        logger.info("Testing Hypothesis 1: Feature effects")
        self.results["hypothesis_1"] = self._test_feature_effects(analysis_data)

        # 2. Hypothesis 2: Dyslexic amplification effects
        logger.info("Testing Hypothesis 2: Dyslexic amplification")
        self.results["hypothesis_2"] = self._test_dyslexic_amplification(analysis_data)

        # 3. Hypothesis 3: Group decomposition
        logger.info("Testing Hypothesis 3: Group decomposition")
        self.results["hypothesis_3"] = self._test_group_decomposition(analysis_data)

        # 4. Model validation and cross-validation
        logger.info("Running model validation")
        self.results["validation"] = self._validate_models(analysis_data)

        # 5. Effect size calculations and variance partitioning
        logger.info("Computing effect sizes and variance components")
        self.results["effect_sizes"] = self._compute_effect_sizes(analysis_data)

        logger.info("Statistical analysis completed!")
        return self.results

    def _prepare_analysis_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean data for statistical analysis"""
        logger.info("Preparing analysis data...")

        # Create a copy
        analysis_data = data.copy()

        # Filter outliers (>3 SD from mean)
        for measure in [
            "first_fixation_duration",
            "gaze_duration",
            "total_reading_time",
        ]:
            if measure in analysis_data.columns:
                mean_val = analysis_data[measure].mean()
                std_val = analysis_data[measure].std()
                analysis_data = analysis_data[
                    (analysis_data[measure] >= mean_val - 3 * std_val)
                    & (analysis_data[measure] <= mean_val + 3 * std_val)
                ]

        # Create group variable (dyslexic vs control)
        if "dyslexic" not in analysis_data.columns:
            # Infer from subject_id or group_id
            analysis_data["dyslexic"] = analysis_data.get("group_id", 0) == 1

        # Center continuous predictors
        continuous_vars = [
            "word_length",
            "log_frequency",
            "launch_site_distance",
            "predictability",
            "trial_position_norm",
        ]

        for var in continuous_vars:
            if var in analysis_data.columns:
                analysis_data[f"{var}_c"] = (
                    analysis_data[var] - analysis_data[var].mean()
                )

        # Ensure proper data types
        analysis_data["subject_id"] = analysis_data["subject_id"].astype(str)
        analysis_data["trial_id"] = analysis_data["trial_id"].astype(str)
        analysis_data["dyslexic"] = analysis_data["dyslexic"].astype(bool)

        logger.info(f"Analysis data prepared. Shape: {analysis_data.shape}")
        return analysis_data

    def _test_feature_effects(self, data: pd.DataFrame) -> Dict:
        """Test Hypothesis 1: Feature effects on eye measures"""
        results = {}

        # Define eye measures to analyze
        eye_measures = [
            "first_fixation_duration",
            "gaze_duration",
            "total_reading_time",
            "skipping_probability",
            "regression_probability",
        ]

        # Define predictors
        predictors = [
            "word_length_c",
            "log_frequency_c",
            "launch_site_distance_c",
            "predictability_c",
        ]

        for measure in eye_measures:
            if measure not in data.columns:
                continue

            logger.info(f"Analyzing {measure}...")

            if measure in ["skipping_probability", "regression_probability"]:
                # Logistic mixed-effects for binary outcomes
                model_results = self._fit_logistic_mixed_model(
                    data, measure, predictors
                )
            else:
                # Linear mixed-effects for continuous outcomes
                model_results = self._fit_linear_mixed_model(data, measure, predictors)

            results[measure] = model_results

        return results

    def _test_dyslexic_amplification(self, data: pd.DataFrame) -> Dict:
        """Test Hypothesis 2: Dyslexic amplification of feature effects"""
        results = {}

        eye_measures = [
            "first_fixation_duration",
            "gaze_duration",
            "total_reading_time",
        ]
        predictors = [
            "word_length_c",
            "log_frequency_c",
            "launch_site_distance_c",
            "predictability_c",
        ]

        for measure in eye_measures:
            if measure not in data.columns:
                continue

            logger.info(f"Testing amplification effects for {measure}...")

            # Model with interaction terms
            interaction_terms = [f"{pred} * dyslexic" for pred in predictors]
            all_terms = predictors + ["dyslexic"] + interaction_terms

            model_results = self._fit_linear_mixed_model(data, measure, all_terms)

            # Extract interaction effect sizes
            interactions = {}
            for pred in predictors:
                interaction_term = f"{pred}:dyslexic[T.True]"
                if interaction_term in model_results.get("coefficients", {}):
                    interactions[pred] = model_results["coefficients"][interaction_term]

            model_results["interactions"] = interactions
            results[measure] = model_results

        return results

    def _test_group_decomposition(self, data: pd.DataFrame) -> Dict:
        """Test Hypothesis 3: Group difference decomposition"""
        results = {}

        # Baseline model: Group only
        baseline_models = {}

        # Full model: Group + Features + Interactions
        full_models = {}

        # Progressive models: Add features incrementally
        progressive_models = {}

        for measure in ["total_reading_time", "gaze_duration"]:
            if measure not in data.columns:
                continue

            logger.info(f"Group decomposition for {measure}...")

            # Baseline: Group only
            baseline_models[measure] = self._fit_linear_mixed_model(
                data, measure, ["dyslexic"]
            )

            # Progressive addition of features
            feature_sets = [
                ["dyslexic", "word_length_c"],
                ["dyslexic", "word_length_c", "log_frequency_c"],
                [
                    "dyslexic",
                    "word_length_c",
                    "log_frequency_c",
                    "launch_site_distance_c",
                ],
                [
                    "dyslexic",
                    "word_length_c",
                    "log_frequency_c",
                    "launch_site_distance_c",
                    "predictability_c",
                ],
            ]

            progressive_models[measure] = {}
            for i, features in enumerate(feature_sets):
                model_name = f"model_{i+1}"
                progressive_models[measure][model_name] = self._fit_linear_mixed_model(
                    data, measure, features
                )

            # Full model with interactions
            all_features = [
                "word_length_c",
                "log_frequency_c",
                "launch_site_distance_c",
                "predictability_c",
            ]
            interaction_terms = [f"{feat} * dyslexic" for feat in all_features]
            full_features = ["dyslexic"] + all_features + interaction_terms

            full_models[measure] = self._fit_linear_mixed_model(
                data, measure, full_features
            )

            # Compute variance explained at each step
            variance_explained = self._compute_variance_decomposition(
                data,
                measure,
                baseline_models[measure],
                progressive_models[measure],
                full_models[measure],
            )

            results[measure] = {
                "baseline": baseline_models[measure],
                "progressive": progressive_models[measure],
                "full": full_models[measure],
                "variance_explained": variance_explained,
            }

        return results

    def _fit_linear_mixed_model(
        self, data: pd.DataFrame, outcome: str, predictors: List[str]
    ) -> Dict:
        """Fit linear mixed-effects model"""

        # Filter out missing data
        model_data = data[predictors + [outcome, "subject_id", "trial_id"]].dropna()

        if len(model_data) == 0:
            logger.warning(f"No valid data for {outcome} model")
            return {"error": "No valid data"}

        try:
            if HAS_PYMER4:
                # Use pymer4 for more advanced mixed effects
                formula = f"{outcome} ~ {' + '.join(predictors)} + (1|subject_id)"
                model = Lmer(formula, data=model_data)
                model.fit()

                results = {
                    "coefficients": model.coefs.to_dict(),
                    "random_effects": model.ranef.to_dict(),
                    "aic": model.AIC,
                    "bic": model.BIC,
                    "marginal_r2": model.marginal_r_squared,
                    "conditional_r2": model.conditional_r_squared,
                    "fitted_values": model.data[outcome + "_fitted"].values,
                    "residuals": model.data[outcome + "_resid"].values,
                    "model_object": model,
                }
            else:
                # Fallback to statsmodels
                formula = f"{outcome} ~ {' + '.join(predictors)}"
                model = smf.mixedlm(
                    formula, model_data, groups=model_data["subject_id"]
                ).fit()

                results = {
                    "coefficients": model.params.to_dict(),
                    "pvalues": model.pvalues.to_dict(),
                    "conf_int": model.conf_int().to_dict(),
                    "aic": model.aic,
                    "bic": model.bic,
                    "fitted_values": model.fittedvalues.values,
                    "residuals": model.resid.values,
                    "model_object": model,
                }

                # Compute R-squared approximation
                ss_res = np.sum(model.resid**2)
                ss_tot = np.sum((model_data[outcome] - model_data[outcome].mean()) ** 2)
                results["r_squared"] = 1 - (ss_res / ss_tot)

            return results

        except Exception as e:
            logger.error(f"Model fitting failed for {outcome}: {e}")
            return {"error": str(e)}

    def _fit_logistic_mixed_model(
        self, data: pd.DataFrame, outcome: str, predictors: List[str]
    ) -> Dict:
        """Fit logistic mixed-effects model for binary outcomes"""

        # Convert to binary if needed
        model_data = data[predictors + [outcome, "subject_id"]].dropna()

        if outcome.endswith("_probability"):
            # Create binary outcome from probability
            binary_outcome = f"{outcome.replace('_probability', '')}_binary"
            model_data[binary_outcome] = (model_data[outcome] > 0.5).astype(int)
            outcome = binary_outcome

        try:
            formula = f"{outcome} ~ {' + '.join(predictors)}"
            model = smf.mixedlm(
                formula,
                model_data,
                groups=model_data["subject_id"],
                family=sm.families.Binomial(),
            ).fit()

            results = {
                "coefficients": model.params.to_dict(),
                "pvalues": model.pvalues.to_dict(),
                "conf_int": model.conf_int().to_dict(),
                "aic": model.aic,
                "fitted_values": model.fittedvalues.values,
                "model_object": model,
            }

            return results

        except Exception as e:
            logger.error(f"Logistic model fitting failed for {outcome}: {e}")
            return {"error": str(e)}

    def _validate_models(self, data: pd.DataFrame) -> Dict:
        """Cross-validation and model diagnostics"""
        results = {}

        # K-fold cross-validation
        kf = KFold(
            n_splits=self.config.get("CROSS_VALIDATION_FOLDS", 5),
            shuffle=True,
            random_state=self.config.get("RANDOM_STATE", 42),
        )

        for measure in ["total_reading_time", "gaze_duration"]:
            if measure not in data.columns:
                continue

            logger.info(f"Cross-validating {measure} models...")

            # Prepare features
            features = [
                "word_length_c",
                "log_frequency_c",
                "launch_site_distance_c",
                "predictability_c",
                "dyslexic",
            ]

            cv_scores = []
            feature_importance = {feat: [] for feat in features}

            # Cross-validation by subjects (not by observations)
            subjects = data["subject_id"].unique()
            subject_folds = [
                (train_subjects, test_subjects)
                for train_subjects, test_subjects in kf.split(subjects)
            ]

            for fold, (train_idx, test_idx) in enumerate(subject_folds):
                train_subjects = subjects[train_idx]
                test_subjects = subjects[test_idx]

                train_data = data[data["subject_id"].isin(train_subjects)]
                test_data = data[data["subject_id"].isin(test_subjects)]

                # Fit model on training data
                model_result = self._fit_linear_mixed_model(
                    train_data, measure, features
                )

                if "error" in model_result:
                    continue

                # Predict on test data (simplified prediction)
                test_pred = self._predict_from_model(model_result, test_data, features)

                if test_pred is not None:
                    test_true = test_data[measure].values
                    cv_score = r2_score(test_true, test_pred)
                    cv_scores.append(cv_score)

                    # Feature importance (coefficient magnitudes)
                    for feat in features:
                        if feat in model_result["coefficients"]:
                            feature_importance[feat].append(
                                abs(model_result["coefficients"][feat])
                            )

            results[measure] = {
                "cv_scores": cv_scores,
                "mean_cv_score": np.mean(cv_scores) if cv_scores else 0,
                "std_cv_score": np.std(cv_scores) if cv_scores else 0,
                "feature_importance": {
                    feat: np.mean(importance) if importance else 0
                    for feat, importance in feature_importance.items()
                },
            }

        return results

    def _predict_from_model(
        self, model_result: Dict, test_data: pd.DataFrame, features: List[str]
    ) -> Optional[np.ndarray]:
        """Make predictions from fitted model (simplified)"""
        try:
            if "model_object" in model_result:
                # Use the actual model object if available
                if HAS_PYMER4:
                    # pymer4 prediction
                    return model_result["model_object"].predict(test_data[features])
                else:
                    # statsmodels prediction is more complex for mixed models
                    # Use simple linear combination for approximation
                    coeffs = model_result["coefficients"]
                    pred = np.zeros(len(test_data))

                    for feat in features:
                        if feat in coeffs and feat in test_data.columns:
                            pred += coeffs[feat] * test_data[feat].values

                    if "Intercept" in coeffs:
                        pred += coeffs["Intercept"]

                    return pred

            return None

        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return None

    def _compute_variance_decomposition(
        self,
        data: pd.DataFrame,
        outcome: str,
        baseline_model: Dict,
        progressive_models: Dict,
        full_model: Dict,
    ) -> Dict:
        """Compute variance explained by each feature set"""

        decomposition = {}

        # Get R-squared values
        baseline_r2 = baseline_model.get("r_squared", 0) or baseline_model.get(
            "marginal_r2", 0
        )

        decomposition["baseline_r2"] = baseline_r2
        decomposition["progressive_r2"] = {}
        decomposition["r2_change"] = {}

        prev_r2 = baseline_r2
        for model_name, model_result in progressive_models.items():
            current_r2 = model_result.get("r_squared", 0) or model_result.get(
                "marginal_r2", 0
            )
            decomposition["progressive_r2"][model_name] = current_r2
            decomposition["r2_change"][model_name] = current_r2 - prev_r2
            prev_r2 = current_r2

        # Full model R-squared
        full_r2 = full_model.get("r_squared", 0) or full_model.get("marginal_r2", 0)
        decomposition["full_r2"] = full_r2
        decomposition["total_explained"] = full_r2 - baseline_r2

        return decomposition

    def _compute_effect_sizes(self, data: pd.DataFrame) -> Dict:
        """Compute effect sizes and confidence intervals"""
        results = {}

        for measure in [
            "total_reading_time",
            "gaze_duration",
            "first_fixation_duration",
        ]:
            if measure not in data.columns:
                continue

            logger.info(f"Computing effect sizes for {measure}...")

            # Group differences (Cohen's d)
            dyslexic_data = data[data["dyslexic"] == True][measure]
            control_data = data[data["dyslexic"] == False][measure]

            cohens_d = self._compute_cohens_d(dyslexic_data, control_data)

            # Feature effect sizes (correlation-based)
            feature_effects = {}
            for feature in ["word_length", "log_frequency", "predictability"]:
                if feature in data.columns:
                    correlation = data[feature].corr(data[measure])
                    feature_effects[feature] = correlation

            results[measure] = {
                "cohens_d": cohens_d,
                "feature_correlations": feature_effects,
                "group_means": {
                    "dyslexic": dyslexic_data.mean(),
                    "control": control_data.mean(),
                },
                "group_stds": {
                    "dyslexic": dyslexic_data.std(),
                    "control": control_data.std(),
                },
            }

        return results

    def _compute_cohens_d(self, group1: pd.Series, group2: pd.Series) -> float:
        """Compute Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        s1, s2 = group1.std(), group2.std()

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

        # Cohen's d
        cohens_d = (group1.mean() - group2.mean()) / pooled_std

        return cohens_d

    def generate_model_summary(self) -> pd.DataFrame:
        """Generate comprehensive model summary table"""
        summary_data = []

        for hypothesis, results in self.results.items():
            if isinstance(results, dict):
                for measure, model_data in results.items():
                    if isinstance(model_data, dict) and "coefficients" in model_data:
                        for coef_name, coef_value in model_data["coefficients"].items():
                            summary_data.append(
                                {
                                    "hypothesis": hypothesis,
                                    "measure": measure,
                                    "predictor": coef_name,
                                    "coefficient": coef_value,
                                    "p_value": model_data.get("pvalues", {}).get(
                                        coef_name, np.nan
                                    ),
                                    "r_squared": model_data.get("r_squared", np.nan),
                                }
                            )

        return pd.DataFrame(summary_data)

    def save_results(self, output_dir: str):
        """Save all analysis results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main results
        pd.DataFrame(self.results).to_pickle(output_path / "statistical_results.pkl")

        # Save model summary
        summary_df = self.generate_model_summary()
        summary_df.to_csv(output_path / "model_summary.csv", index=False)

        logger.info(f"Results saved to {output_path}")


def main():
    """Test statistical analysis with sample data"""
    pass


if __name__ == "__main__":
    main()
