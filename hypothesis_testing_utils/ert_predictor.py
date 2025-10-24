"""
Expected Reading Time (ERT) Predictor
Combines skip and duration models: ERT = [1 - P(skip)] × E[TRT|fix]
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ERTPredictor:
    """
    Combines skip and duration models to predict Expected Reading Time
    """

    def __init__(self, gam_models):
        """
        Initialize with fitted GAM models

        Args:
            gam_models: DyslexiaGAMModels instance with fitted models
        """
        self.gam = gam_models

        if gam_models.skip_model is None or gam_models.duration_model is None:
            raise ValueError("Models must be fitted before creating predictor")

        logger.info("ERT Predictor initialized")

    def predict_ert(
        self, features: pd.DataFrame, group: str, return_components: bool = False
    ) -> np.ndarray:
        """
        Predict Expected Reading Time

        ERT = [1 - P(skip)] × E[TRT|fix]

        Args:
            features: DataFrame with columns [length, zipf, surprisal]
            group: "control" or "dyslexic"
            return_components: If True, return (ERT, P_skip, TRT) tuple

        Returns:
            Array of ERT predictions (ms), or tuple if return_components=True
        """
        # Convert to numpy array
        X = features[["length", "zipf", "surprisal"]].values

        # Get predictions from both models
        p_skip = self.gam.predict_skip(X, group)
        trt_given_fix = self.gam.predict_trt(X, group)

        # Combine: ERT = [1 - P(skip)] × E[TRT|fix]
        ert = (1 - p_skip) * trt_given_fix

        if return_components:
            return ert, p_skip, trt_given_fix
        else:
            return ert

    def predict_ert_both_groups(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict ERT for both groups

        Args:
            features: DataFrame with feature columns

        Returns:
            Dictionary with keys "control" and "dyslexic"
        """
        return {
            "control": self.predict_ert(features, "control"),
            "dyslexic": self.predict_ert(features, "dyslexic"),
        }

    def predict_skip_probability(
        self, features: pd.DataFrame, group: str
    ) -> np.ndarray:
        """Convenience method for skip predictions"""
        X = features[["length", "zipf", "surprisal"]].values
        return self.gam.predict_skip(X, group)

    def predict_trt_given_fixation(
        self, features: pd.DataFrame, group: str
    ) -> np.ndarray:
        """Convenience method for TRT predictions"""
        X = features[["length", "zipf", "surprisal"]].values
        return self.gam.predict_trt(X, group)

    def create_prediction_grid(
        self,
        data: pd.DataFrame,
        feature: str,
        n_points: int = 100,
        feature_range: Optional[Tuple[float, float]] = None,
    ) -> pd.DataFrame:
        """
        Create prediction grid for plotting effects

        Args:
            data: Original data (for computing ranges and means)
            feature: Feature to vary ('length', 'zipf', or 'surprisal')
            n_points: Number of points in grid
            feature_range: (min, max) tuple, or None to use data range

        Returns:
            DataFrame ready for prediction
        """
        if feature_range is None:
            # Use 5th to 95th percentile to avoid extremes
            feature_range = (
                data[feature].quantile(0.05),
                data[feature].quantile(0.95),
            )

        # Create grid
        grid = pd.DataFrame()
        grid[feature] = np.linspace(feature_range[0], feature_range[1], n_points)

        # Fix other features at mean
        for feat in ["length", "zipf", "surprisal"]:
            if feat != feature:
                grid[feat] = data[feat].mean()

        return grid

    def compute_marginal_effect(
        self,
        features: pd.DataFrame,
        feature: str,
        group: str,
        delta: float = 0.01,
    ) -> np.ndarray:
        """
        Compute marginal effect ∂ERT/∂feature using finite differences

        Args:
            features: Prediction grid
            feature: Feature to compute derivative for
            group: "control" or "dyslexic"
            delta: Step size for finite difference

        Returns:
            Array of marginal effects (ms per unit change in feature)
        """
        # Baseline prediction
        ert_base = self.predict_ert(features, group)

        # Perturb feature slightly
        features_perturbed = features.copy()
        features_perturbed[feature] = features_perturbed[feature] + delta

        # Perturbed prediction
        ert_perturbed = self.predict_ert(features_perturbed, group)

        # Finite difference
        marginal_effect = (ert_perturbed - ert_base) / delta

        return marginal_effect


def create_ert_predictor(gam_models) -> ERTPredictor:
    """
    Convenience function to create ERT predictor

    Args:
        gam_models: DyslexiaGAMModels instance with fitted models

    Returns:
        ERTPredictor instance
    """
    return ERTPredictor(gam_models)
