"""
GAM Model Fitting for Dyslexia Analysis - FIXED
Uses pygam for GAM fitting with group-specific smooths
"""

import logging
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pygam import GammaGAM, LogisticGAM, s
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DyslexiaGAMModels:
    """
    Fits skip and duration GAMs with group-specific smooths using pygam
    """

    def __init__(self):
        """Initialize"""
        self.skip_model = None
        self.duration_model = None
        self.feature_means = None
        logger.info("GAM Models initialized (pygam)")

    def fit_skip_model(self, data: pd.DataFrame) -> Dict:
        """
        Fit skip model (logistic GAM):
        skip ~ group + s(length, by=group) + s(zipf, by=group) + s(surprisal, by=group)
        """
        logger.info("=" * 60)
        logger.info("FITTING SKIP MODEL (Logistic GAM)")
        logger.info("=" * 60)

        # Prepare data
        X, y, group = self._prepare_skip_data(data)

        # Store feature means
        self.feature_means = {
            "length": X[:, 0].mean(),
            "zipf": X[:, 1].mean(),
            "surprisal": X[:, 2].mean(),
        }

        logger.info("Fitting skip model...")
        try:
            # Split by group
            ctrl_mask = group == 0
            dys_mask = group == 1

            X_ctrl, y_ctrl = X[ctrl_mask], y[ctrl_mask]
            X_dys, y_dys = X[dys_mask], y[dys_mask]

            logger.info(
                f"  Control: {len(X_ctrl):,} obs, skip rate: {y_ctrl.mean():.3f}"
            )
            logger.info(
                f"  Dyslexic: {len(X_dys):,} obs, skip rate: {y_dys.mean():.3f}"
            )

            # Fit control model
            logger.info("  Fitting control model...")
            gam_ctrl = LogisticGAM(
                s(0, n_splines=5) + s(1, n_splines=5) + s(2, n_splines=5)
            )
            gam_ctrl.gridsearch(X_ctrl, y_ctrl, progress=False)

            # Fit dyslexic model
            logger.info("  Fitting dyslexic model...")
            gam_dys = LogisticGAM(
                s(0, n_splines=5) + s(1, n_splines=5) + s(2, n_splines=5)
            )
            gam_dys.gridsearch(X_dys, y_dys, progress=False)

            # Store both models
            self.skip_model = {
                "control": gam_ctrl,
                "dyslexic": gam_dys,
            }

            # Compute proper metrics (AUC instead of accuracy)
            ctrl_pred_proba = gam_ctrl.predict_proba(X_ctrl)
            dys_pred_proba = gam_dys.predict_proba(X_dys)

            ctrl_auc = roc_auc_score(y_ctrl, ctrl_pred_proba)
            dys_auc = roc_auc_score(y_dys, dys_pred_proba)

            logger.info("Skip models fitted successfully")
            logger.info(f"  Control AUC: {ctrl_auc:.3f}")
            logger.info(f"  Dyslexic AUC: {dys_auc:.3f}")

            return {
                # Don't include model objects - just metadata
                "family": "binomial",
                "n_obs_control": int(len(X_ctrl)),
                "n_obs_dyslexic": int(len(X_dys)),
                "skip_rate_control": float(y_ctrl.mean()),
                "skip_rate_dyslexic": float(y_dys.mean()),
                "auc_control": float(ctrl_auc),
                "auc_dyslexic": float(dys_auc),
                "n_splines": 5,
            }

        except Exception as e:
            logger.error(f"Skip model fitting failed: {e}")
            raise

    def fit_duration_model(self, data: pd.DataFrame) -> Dict:
        """
        Fit duration model (Gamma GAM)
        """
        logger.info("=" * 60)
        logger.info("FITTING DURATION MODEL (Gamma GAM)")
        logger.info("=" * 60)

        # Filter to fixated words only
        fixated = data[data["skip"] == 0].copy()
        logger.info(f"Fitting on {len(fixated):,} fixated words")

        # Prepare data
        X, y, group = self._prepare_duration_data(fixated)

        logger.info("Fitting duration model...")
        try:
            # Split by group
            ctrl_mask = group == 0
            dys_mask = group == 1

            X_ctrl, y_ctrl = X[ctrl_mask], y[ctrl_mask]
            X_dys, y_dys = X[dys_mask], y[dys_mask]

            logger.info(
                f"  Control: {len(X_ctrl):,} obs, mean TRT: {y_ctrl.mean():.1f}ms"
            )
            logger.info(
                f"  Dyslexic: {len(X_dys):,} obs, mean TRT: {y_dys.mean():.1f}ms"
            )

            # Fit control model
            logger.info("  Fitting control model...")
            gam_ctrl = GammaGAM(
                s(0, n_splines=5) + s(1, n_splines=5) + s(2, n_splines=5)
            )
            gam_ctrl.gridsearch(X_ctrl, y_ctrl, progress=False)

            # Fit dyslexic model
            logger.info("  Fitting dyslexic model...")
            gam_dys = GammaGAM(
                s(0, n_splines=5) + s(1, n_splines=5) + s(2, n_splines=5)
            )
            gam_dys.gridsearch(X_dys, y_dys, progress=False)

            # Store both models
            self.duration_model = {
                "control": gam_ctrl,
                "dyslexic": gam_dys,
            }

            # Compute pseudo-R²
            ctrl_r2 = gam_ctrl.statistics_["pseudo_r2"]["explained_deviance"]
            dys_r2 = gam_dys.statistics_["pseudo_r2"]["explained_deviance"]

            logger.info("Duration models fitted successfully")
            logger.info(f"  Control pseudo-R²: {ctrl_r2:.3f}")
            logger.info(f"  Dyslexic pseudo-R²: {dys_r2:.3f}")

            return {
                # Don't include model objects
                "family": "Gamma",
                "n_obs_control": int(len(X_ctrl)),
                "n_obs_dyslexic": int(len(X_dys)),
                "mean_trt_control": float(y_ctrl.mean()),
                "mean_trt_dyslexic": float(y_dys.mean()),
                "r2_control": float(ctrl_r2),
                "r2_dyslexic": float(dys_r2),
                "n_splines": 5,
            }

        except Exception as e:
            logger.error(f"Duration model fitting failed: {e}")
            raise

    def _prepare_skip_data(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for skip model"""
        required = ["length", "zipf", "surprisal", "skip", "group_numeric"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Features: length, zipf, surprisal
        X = data[["length", "zipf", "surprisal"]].values
        y = data["skip"].values
        group = data["group_numeric"].values

        # Check for any issues before filtering
        n_before = len(X)

        # Remove any NaN rows
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isnan(group)
        X = X[valid_mask]
        y = y[valid_mask]
        group = group[valid_mask]

        n_after = len(X)
        if n_after < n_before:
            logger.warning(f"  Dropped {n_before - n_after} rows with NaN values")

        logger.info(f"  Skip data prepared: {len(X):,} observations")
        logger.info(f"    Control: {(group==0).sum():,}")
        logger.info(f"    Dyslexic: {(group==1).sum():,}")

        return X, y, group

    def _prepare_duration_data(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for duration model"""
        required = ["length", "zipf", "surprisal", "TRT", "group_numeric"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        X = data[["length", "zipf", "surprisal"]].values
        y = data["TRT"].values
        group = data["group_numeric"].values

        # Remove any NaN rows and ensure TRT > 0
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & (y > 0)
        X = X[valid_mask]
        y = y[valid_mask]
        group = group[valid_mask]

        logger.info(f"  Duration data prepared: {len(X):,} observations")
        logger.info(f"    Control: {(group==0).sum():,}")
        logger.info(f"    Dyslexic: {(group==1).sum():,}")

        return X, y, group

    def predict_skip(self, X: np.ndarray, group: str) -> np.ndarray:
        """Predict skip probability"""
        if self.skip_model is None:
            raise ValueError("Skip model not fitted yet")

        model = self.skip_model[group]
        predictions = model.predict_proba(X)
        return predictions

    def predict_trt(self, X: np.ndarray, group: str) -> np.ndarray:
        """Predict TRT (given fixation)"""
        if self.duration_model is None:
            raise ValueError("Duration model not fitted yet")

        model = self.duration_model[group]
        predictions = model.predict(X)
        return predictions


def fit_gam_models(data: pd.DataFrame) -> Tuple[Dict, Dict, DyslexiaGAMModels]:
    """
    Convenience function to fit both models
    Returns metadata only (not model objects)
    """
    gam = DyslexiaGAMModels()

    skip_model_metadata = gam.fit_skip_model(data)
    duration_model_metadata = gam.fit_duration_model(data)

    return skip_model_metadata, duration_model_metadata, gam
