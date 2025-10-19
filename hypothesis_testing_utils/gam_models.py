"""
GAM Models with Tensor Product Interactions - REVISED
Key changes:
- Added te(length, zipf) tensor product interaction
- Grid search over n_splines and lambda
- Proper GroupKFold CV
- Smearing factor for log-Gaussian duration model
"""

import logging
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pygam import LinearGAM, LogisticGAM, s, te
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DyslexiaGAMModels:
    """
    GAM models with tensor product interactions
    """

    def __init__(self):
        self.skip_model = None
        self.duration_model = None
        self.feature_means = None
        self.smearing_factors = None  # For log-Gaussian duration
        logger.info("GAM Models initialized with tensor product support")

    def fit_skip_model(
        self,
        data: pd.DataFrame,
        n_splines_search: list = [8, 10, 12],
        lam_search: np.ndarray = None,
    ) -> Dict:
        """
        Fit skip model with tensor product interaction
        skip ~ s(length) + s(zipf) + s(surprisal) + te(length, zipf)
        """
        logger.info("=" * 60)
        logger.info("FITTING SKIP MODEL (Logistic GAM + Tensor Product)")
        logger.info("=" * 60)

        if lam_search is None:
            lam_search = np.logspace(-3, 3, 11)

        # Prepare data
        X, y, group, subjects = self._prepare_skip_data(data)

        # Store feature means
        self.feature_means = {
            "length": X[:, 0].mean(),
            "zipf": X[:, 1].mean(),
            "surprisal": X[:, 2].mean(),
        }

        # Split by group
        ctrl_mask = group == 0
        dys_mask = group == 1

        X_ctrl, y_ctrl, subj_ctrl = X[ctrl_mask], y[ctrl_mask], subjects[ctrl_mask]
        X_dys, y_dys, subj_dys = X[dys_mask], y[dys_mask], subjects[dys_mask]

        logger.info(f"  Control: {len(X_ctrl):,} obs, skip rate: {y_ctrl.mean():.3f}")
        logger.info(f"  Dyslexic: {len(X_dys):,} obs, skip rate: {y_dys.mean():.3f}")

        # Fit control model with grid search
        logger.info("  Fitting control model (with CV grid search)...")
        gam_ctrl = self._fit_skip_model_cv(
            X_ctrl, y_ctrl, subj_ctrl, n_splines_search, lam_search
        )

        # Fit dyslexic model
        logger.info("  Fitting dyslexic model (with CV grid search)...")
        gam_dys = self._fit_skip_model_cv(
            X_dys, y_dys, subj_dys, n_splines_search, lam_search
        )

        # Store models
        self.skip_model = {"control": gam_ctrl, "dyslexic": gam_dys}

        # Compute metrics
        ctrl_pred = gam_ctrl.predict_proba(X_ctrl)
        dys_pred = gam_dys.predict_proba(X_dys)

        ctrl_auc = roc_auc_score(y_ctrl, ctrl_pred)
        dys_auc = roc_auc_score(y_dys, dys_pred)

        logger.info("Skip models fitted successfully")
        logger.info(f"  Control AUC: {ctrl_auc:.3f}")
        logger.info(f"  Dyslexic AUC: {dys_auc:.3f}")

        return {
            "family": "binomial",
            "n_obs_control": int(len(X_ctrl)),
            "n_obs_dyslexic": int(len(X_dys)),
            "auc_control": float(ctrl_auc),
            "auc_dyslexic": float(dys_auc),
            "has_tensor_product": True,
        }

    def _fit_skip_model_cv(self, X, y, subjects, n_splines_search, lam_search):
        """Fit skip model with GroupKFold CV"""

        # Model specification with tensor product
        # s(0) = length, s(1) = zipf, s(2) = surprisal, te(0,1) = length × zipf
        best_auc = -np.inf
        best_model = None

        # Grid search
        for n_sp in n_splines_search:
            gam = LogisticGAM(
                s(0, n_splines=n_sp)  # length
                + s(1, n_splines=n_sp)  # zipf
                + s(2, n_splines=n_sp)  # surprisal
                + te(0, 1, n_splines=[n_sp, n_sp])  # length × zipf interaction
            )

            # Grid search over lambda
            gam.gridsearch(X, y, lam=lam_search, progress=False)

            # Cross-validate with GroupKFold
            gkf = GroupKFold(n_splits=min(10, len(np.unique(subjects))))
            cv_aucs = []

            for train_idx, val_idx in gkf.split(X, y, groups=subjects):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Refit on fold
                gam_fold = LogisticGAM(
                    s(0, n_splines=n_sp)
                    + s(1, n_splines=n_sp)
                    + s(2, n_splines=n_sp)
                    + te(0, 1, n_splines=[n_sp, n_sp])
                )
                gam_fold.gridsearch(X_train, y_train, lam=lam_search, progress=False)

                # Evaluate
                pred_val = gam_fold.predict_proba(X_val)
                auc_val = roc_auc_score(y_val, pred_val)
                cv_aucs.append(auc_val)

            mean_auc = np.mean(cv_aucs)

            if mean_auc > best_auc:
                best_auc = mean_auc
                best_model = gam

        logger.info(f"    Best CV AUC: {best_auc:.3f}")

        return best_model

    def fit_duration_model(
        self,
        data: pd.DataFrame,
        n_splines_search: list = [8, 10, 12],
        lam_search: np.ndarray = None,
        use_log_transform: bool = True,
    ) -> Dict:
        """
        Fit duration model (log-Gaussian with smearing factor)
        log(TRT) ~ s(length) + s(zipf) + s(surprisal) + te(length, zipf)
        """
        logger.info("=" * 60)
        logger.info(
            f"FITTING DURATION MODEL ({'Log-Gaussian' if use_log_transform else 'Gamma'} GAM)"
        )
        logger.info("=" * 60)

        if lam_search is None:
            lam_search = np.logspace(-3, 3, 11)

        # Filter to fixated words
        fixated = data[data["skip"] == 0].copy()
        logger.info(f"Fitting on {len(fixated):,} fixated words")

        # Prepare data
        X, y, group, subjects = self._prepare_duration_data(fixated, use_log_transform)

        # Split by group
        ctrl_mask = group == 0
        dys_mask = group == 1

        X_ctrl, y_ctrl, subj_ctrl = X[ctrl_mask], y[ctrl_mask], subjects[ctrl_mask]
        X_dys, y_dys, subj_dys = X[dys_mask], y[dys_mask], subjects[dys_mask]

        if use_log_transform:
            logger.info(
                f"  Control: {len(X_ctrl):,} obs, mean log(TRT): {y_ctrl.mean():.3f}"
            )
            logger.info(
                f"  Dyslexic: {len(X_dys):,} obs, mean log(TRT): {y_dys.mean():.3f}"
            )
        else:
            logger.info(
                f"  Control: {len(X_ctrl):,} obs, mean TRT: {y_ctrl.mean():.1f}ms"
            )
            logger.info(
                f"  Dyslexic: {len(X_dys):,} obs, mean TRT: {y_dys.mean():.1f}ms"
            )

        # Fit models
        logger.info("  Fitting control model...")
        gam_ctrl, residuals_ctrl = self._fit_duration_model_cv(
            X_ctrl, y_ctrl, subj_ctrl, n_splines_search, lam_search
        )

        logger.info("  Fitting dyslexic model...")
        gam_dys, residuals_dys = self._fit_duration_model_cv(
            X_dys, y_dys, subj_dys, n_splines_search, lam_search
        )

        # Store models
        self.duration_model = {"control": gam_ctrl, "dyslexic": gam_dys}

        # Compute smearing factors (for log-transform)
        if use_log_transform:
            self.smearing_factors = {
                "control": float(np.mean(np.exp(residuals_ctrl))),
                "dyslexic": float(np.mean(np.exp(residuals_dys))),
            }
            logger.info(
                f"  Smearing factors: Control={self.smearing_factors['control']:.4f}, "
                f"Dyslexic={self.smearing_factors['dyslexic']:.4f}"
            )
        else:
            self.smearing_factors = {"control": 1.0, "dyslexic": 1.0}

        # Compute R²
        ctrl_r2 = gam_ctrl.statistics_["pseudo_r2"]["explained_deviance"]
        dys_r2 = gam_dys.statistics_["pseudo_r2"]["explained_deviance"]

        logger.info("Duration models fitted successfully")
        logger.info(f"  Control pseudo-R²: {ctrl_r2:.3f}")
        logger.info(f"  Dyslexic pseudo-R²: {dys_r2:.3f}")

        return {
            "family": "Gaussian (log)" if use_log_transform else "Gamma",
            "n_obs_control": int(len(X_ctrl)),
            "n_obs_dyslexic": int(len(X_dys)),
            "r2_control": float(ctrl_r2),
            "r2_dyslexic": float(dys_r2),
            "has_tensor_product": True,
            "uses_smearing_factor": use_log_transform,
        }

    def _fit_duration_model_cv(self, X, y, subjects, n_splines_search, lam_search):
        """Fit duration model with GroupKFold CV"""

        best_rmse = np.inf
        best_model = None

        for n_sp in n_splines_search:
            gam = LinearGAM(
                s(0, n_splines=n_sp)
                + s(1, n_splines=n_sp)
                + s(2, n_splines=n_sp)
                + te(0, 1, n_splines=[n_sp, n_sp])
            )

            gam.gridsearch(X, y, lam=lam_search, progress=False)

            # Cross-validate
            gkf = GroupKFold(n_splits=min(10, len(np.unique(subjects))))
            cv_rmses = []

            for train_idx, val_idx in gkf.split(X, y, groups=subjects):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                gam_fold = LinearGAM(
                    s(0, n_splines=n_sp)
                    + s(1, n_splines=n_sp)
                    + s(2, n_splines=n_sp)
                    + te(0, 1, n_splines=[n_sp, n_sp])
                )
                gam_fold.gridsearch(X_train, y_train, lam=lam_search, progress=False)

                pred_val = gam_fold.predict(X_val)
                rmse = np.sqrt(np.mean((y_val - pred_val) ** 2))
                cv_rmses.append(rmse)

            mean_rmse = np.mean(cv_rmses)

            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_model = gam

        # Compute residuals for smearing factor
        predictions = best_model.predict(X)
        residuals = y - predictions

        logger.info(f"    Best CV RMSE: {best_rmse:.3f}")

        return best_model, residuals

    def _prepare_skip_data(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare skip data"""
        X = data[["length", "zipf", "surprisal"]].values
        y = data["skip"].values
        group = data["group_numeric"].values
        subjects = data["subject_id"].values

        # Remove NaN
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)

        return X[valid_mask], y[valid_mask], group[valid_mask], subjects[valid_mask]

    def _prepare_duration_data(
        self, data: pd.DataFrame, use_log: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare duration data"""
        X = data[["length", "zipf", "surprisal"]].values
        y = data["TRT"].values

        if use_log:
            y = np.log(np.clip(y, 1, None))  # log transform

        group = data["group_numeric"].values
        subjects = data["subject_id"].values

        valid_mask = (
            ~np.isnan(X).any(axis=1) & ~np.isnan(y) & (y > 0 if not use_log else True)
        )

        return X[valid_mask], y[valid_mask], group[valid_mask], subjects[valid_mask]

    def predict_skip(self, X: np.ndarray, group: str) -> np.ndarray:
        """Predict skip probability"""
        if self.skip_model is None:
            raise ValueError("Skip model not fitted")

        return self.skip_model[group].predict_proba(X)

    def predict_trt(self, X: np.ndarray, group: str) -> np.ndarray:
        """Predict TRT (with smearing factor if log-transform used)"""
        if self.duration_model is None:
            raise ValueError("Duration model not fitted")

        predictions = self.duration_model[group].predict(X)

        # Apply smearing factor if log-transform was used
        if self.smearing_factors is not None:
            smearing = self.smearing_factors[group]
            predictions = np.exp(predictions) * smearing

        return predictions


def fit_gam_models(
    data: pd.DataFrame, use_log_duration: bool = True
) -> Tuple[Dict, Dict, DyslexiaGAMModels]:
    """
    Fit both GAM models with tensor products

    Args:
        data: Prepared data
        use_log_duration: Use log-Gaussian (True) or Gamma (False) for duration

    Returns:
        (skip_metadata, duration_metadata, gam_instance) tuple
    """
    gam = DyslexiaGAMModels()

    skip_metadata = gam.fit_skip_model(data)
    duration_metadata = gam.fit_duration_model(data, use_log_transform=use_log_duration)

    return skip_metadata, duration_metadata, gam
