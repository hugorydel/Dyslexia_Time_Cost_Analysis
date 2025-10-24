"""
GAM Models
"""

import logging
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pygam import LinearGAM, LogisticGAM, s
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DyslexiaGAMModels:
    """
    GAM models with additive smooths only (no tensor product)
    Uses proper nested CV with 1-SE rule
    """

    def __init__(self):
        self.skip_model = None
        self.duration_model = None
        self.feature_means = None
        self.smearing_factors = None
        logger.info("GAM Models initialized (additive smooths only, no tensor product)")

    def to_cache_blob(self):
        return {
            "skip_control": self.skip_model["control"],
            "skip_dyslexic": self.skip_model["dyslexic"],
            "dur_control": self.duration_model["control"],
            "dur_dyslexic": self.duration_model["dyslexic"],
            "feature_means": self.feature_means,
            "smearing_factors": self.smearing_factors,
        }

    @classmethod
    def from_cache_blob(cls, blob):
        obj = cls()
        obj.skip_model = {
            "control": blob["skip_control"],
            "dyslexic": blob["skip_dyslexic"],
        }
        obj.duration_model = {
            "control": blob["dur_control"],
            "dyslexic": blob["dur_dyslexic"],
        }
        obj.feature_means = blob.get("feature_means")
        obj.smearing_factors = blob.get("smearing_factors")
        return obj

    def _fit_single_fold(self, X, y, tr, va, n_splines, lam_search, model_type):
        """Fit single fold for CV"""
        if model_type == "skip":
            gam = LogisticGAM(
                s(0, n_splines=n_splines)
                + s(1, n_splines=n_splines, constraints="monotonic_inc")  # Zipf ↑
                + s(2, n_splines=n_splines)
            )
        else:
            gam = LinearGAM(
                s(0, n_splines=n_splines)
                + s(1, n_splines=n_splines, constraints="monotonic_dec")  # Zipf ↓
                + s(2, n_splines=n_splines)
            )

        # Fixed λ vs gridsearch
        def _is_single_lam(lam):
            if np.isscalar(lam):
                return True
            try:
                seq = list(lam)
            except TypeError:
                return True
            return len(seq) == 1 and not hasattr(seq[0], "__iter__")

        if _is_single_lam(lam_search):
            lam_val = float(np.atleast_1d(lam_search)[0])
            gam.lam = lam_val
            gam.fit(X[tr], y[tr])
        else:
            gam.gridsearch(X[tr], y[tr], lam=lam_search, progress=False)

        if model_type == "skip":
            p = gam.predict_proba(X[va])
            return roc_auc_score(y[va], p)
        else:
            yhat = gam.predict(X[va])
            return np.sqrt(np.mean((y[va] - yhat) ** 2))

    def fit_skip_model(
        self,
        data: pd.DataFrame,
        n_splines_search: list = None,
        lam_search: np.ndarray = None,
        quick_mode: bool = False,
    ) -> Dict:
        """
        Fit skip model (NO tensor product)
        skip ~ s(length) + s(zipf) + s(surprisal)
        """
        logger.info("=" * 60)
        logger.info("FITTING SKIP MODEL (Logistic GAM, Additive)")
        logger.info("=" * 60)

        if n_splines_search is None:
            n_splines_search = [8, 10] if not quick_mode else [10]

        if lam_search is None:
            lam_search = (
                np.logspace(-3, 3, 7) if not quick_mode else np.logspace(-2, 2, 5)
            )

        if quick_mode:
            logger.info("  ⚡ QUICK MODE enabled")

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

        # === CONTROL MODEL ===
        logger.info("\n  Fitting control model...")

        n_sp_ctrl, lam_ctrl = self._preselect_hyperparameters(
            X_ctrl, y_ctrl, subj_ctrl, n_splines_search, lam_search, "skip"
        )

        lam_grid_dupe = np.array([lam_ctrl, lam_ctrl], dtype=float)
        cv_auc_ctrl = self._validate_with_frozen_hyperparameters(
            X_ctrl, y_ctrl, subj_ctrl, n_sp_ctrl, lam_grid_dupe, "skip"
        )

        logger.info("    Fitting final model on all control data...")
        gam_ctrl = LogisticGAM(
            s(0, n_splines=n_sp_ctrl)
            + s(1, n_splines=n_sp_ctrl, constraints="monotonic_inc")  # Zipf ↑
            + s(2, n_splines=n_sp_ctrl)
        )
        gam_ctrl.lam = float(lam_ctrl)
        gam_ctrl.fit(X_ctrl, y_ctrl)

        ctrl_pred = gam_ctrl.predict_proba(X_ctrl)
        ctrl_auc = roc_auc_score(y_ctrl, ctrl_pred)
        logger.info(f"    Final control AUC: {ctrl_auc:.3f}")
        logger.info(
            f"    Final control EDF: {np.sum(gam_ctrl.statistics_['edof']):.1f}"
        )

        # === DYSLEXIC MODEL ===
        logger.info("\n  Fitting dyslexic model...")

        n_sp_dys, lam_dys = self._preselect_hyperparameters(
            X_dys, y_dys, subj_dys, n_splines_search, lam_search, "skip"
        )

        lam_grid_dupe = np.array([lam_dys, lam_dys], dtype=float)
        cv_auc_dys = self._validate_with_frozen_hyperparameters(
            X_dys, y_dys, subj_dys, n_sp_dys, lam_grid_dupe, "skip"
        )

        logger.info("    Fitting final model on all dyslexic data...")
        gam_dys = LogisticGAM(
            s(0, n_splines=n_sp_dys)
            + s(1, n_splines=n_sp_dys, constraints="monotonic_inc")  # Zipf ↑ (ADD THIS)
            + s(2, n_splines=n_sp_dys)
        )
        gam_dys.lam = float(lam_dys)
        gam_dys.fit(X_dys, y_dys)

        dys_pred = gam_dys.predict_proba(X_dys)
        dys_auc = roc_auc_score(y_dys, dys_pred)
        logger.info(f"    Final dyslexic AUC: {dys_auc:.3f}")
        logger.info(
            f"    Final dyslexic EDF: {np.sum(gam_dys.statistics_['edof']):.1f}"
        )

        self.skip_model = {"control": gam_ctrl, "dyslexic": gam_dys}

        logger.info("\n✓ Skip models fitted successfully")
        logger.info(f"  Control: AUC={ctrl_auc:.3f} (CV: {cv_auc_ctrl:.3f})")
        logger.info(f"  Dyslexic: AUC={dys_auc:.3f} (CV: {cv_auc_dys:.3f})")

        return {
            "family": "binomial",
            "n_obs_control": int(len(X_ctrl)),
            "n_obs_dyslexic": int(len(X_dys)),
            "auc_control": float(ctrl_auc),
            "auc_dyslexic": float(dys_auc),
            "cv_auc_control": float(cv_auc_ctrl),
            "cv_auc_dyslexic": float(cv_auc_dys),
            "n_splines_control": int(n_sp_ctrl),
            "n_splines_dyslexic": int(n_sp_dys),
            "lambda_control": float(lam_ctrl),
            "lambda_dyslexic": float(lam_dys),
            "edf_control": float(np.sum(gam_ctrl.statistics_["edof"])),
            "edf_dyslexic": float(np.sum(gam_dys.statistics_["edof"])),
            "has_tensor_product": False,
            "method": "additive_smooths_only",
        }

    def _prepare_skip_data(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare skip data"""
        X = data[["length", "zipf", "surprisal"]].values
        y = data["skip"].values
        group = data["group_numeric"].values
        subjects = data["subject_id"].values

        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)

        X_filtered = X[valid_mask].astype(np.float32)
        return X_filtered, y[valid_mask], group[valid_mask], subjects[valid_mask]

    def _prepare_duration_data(
        self, data: pd.DataFrame, use_log: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare duration data"""
        X = data[["length", "zipf", "surprisal"]].values
        y = data["TRT"].values

        if use_log:
            y = np.log(np.clip(y, 1, None))

        group = data["group_numeric"].values
        subjects = data["subject_id"].values

        valid_mask = (
            ~np.isnan(X).any(axis=1) & ~np.isnan(y) & (y > 0 if not use_log else True)
        )

        X_filtered = X[valid_mask].astype(np.float32)
        return X_filtered, y[valid_mask], group[valid_mask], subjects[valid_mask]

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

        if self.smearing_factors is not None:
            smearing = self.smearing_factors[group]
            predictions = np.exp(predictions) * smearing

        return predictions

    def fit_duration_model(
        self,
        data: pd.DataFrame,
        n_splines_search: list = None,
        lam_search: np.ndarray = None,
        use_log_transform: bool = True,
        quick_mode: bool = False,
    ) -> Dict:
        """
        Fit duration model (NO tensor product)
        log(TRT) ~ s(length) + s(zipf) + s(surprisal)
        """
        logger.info("=" * 60)
        logger.info(
            f"FITTING DURATION MODEL ({'Log-Gaussian' if use_log_transform else 'Gamma'} GAM, Additive)"
        )
        logger.info("=" * 60)

        if n_splines_search is None:
            n_splines_search = [8, 10] if not quick_mode else [10]

        if lam_search is None:
            lam_search = (
                np.logspace(-3, 3, 7) if not quick_mode else np.logspace(-2, 2, 5)
            )

        if quick_mode:
            logger.info("  ⚡ QUICK MODE enabled")

        fixated = data[data["skip"] == 0].copy()
        logger.info(f"  Fitting on {len(fixated):,} fixated words")

        X, y, group, subjects = self._prepare_duration_data(fixated, use_log_transform)

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

        # === CONTROL MODEL ===
        logger.info("\n  Fitting control model...")

        n_sp_ctrl, lam_ctrl = self._preselect_hyperparameters(
            X_ctrl, y_ctrl, subj_ctrl, n_splines_search, lam_search, "duration"
        )

        lam_grid_dupe = np.array([lam_ctrl, lam_ctrl], dtype=float)
        cv_rmse_ctrl = self._validate_with_frozen_hyperparameters(
            X_ctrl, y_ctrl, subj_ctrl, n_sp_ctrl, lam_grid_dupe, "duration"
        )

        logger.info("    Fitting final model on all control data...")
        gam_ctrl = LinearGAM(
            s(0, n_splines=n_sp_ctrl)
            + s(1, n_splines=n_sp_ctrl, constraints="monotonic_dec")  # Zipf ↓
            + s(2, n_splines=n_sp_ctrl)
        )
        gam_ctrl.lam = float(lam_ctrl)
        gam_ctrl.fit(X_ctrl, y_ctrl)

        residuals_ctrl = y_ctrl - gam_ctrl.predict(X_ctrl)
        ctrl_r2 = gam_ctrl.statistics_["pseudo_r2"]["explained_deviance"]
        logger.info(f"    Final control R²: {ctrl_r2:.3f}")
        logger.info(
            f"    Final control EDF: {np.sum(gam_ctrl.statistics_['edof']):.1f}"
        )

        # === DYSLEXIC MODEL ===
        logger.info("\n  Fitting dyslexic model...")

        n_sp_dys, lam_dys = self._preselect_hyperparameters(
            X_dys, y_dys, subj_dys, n_splines_search, lam_search, "duration"
        )

        lam_grid_dupe = np.array([lam_dys, lam_dys], dtype=float)
        cv_rmse_dys = self._validate_with_frozen_hyperparameters(
            X_dys, y_dys, subj_dys, n_sp_dys, lam_grid_dupe, "duration"
        )

        logger.info("    Fitting final model on all dyslexic data...")
        gam_dys = LinearGAM(
            s(0, n_splines=n_sp_dys)
            + s(1, n_splines=n_sp_dys, constraints="monotonic_dec")  # Zipf ↓
            + s(2, n_splines=n_sp_dys)
        )
        gam_dys.lam = float(lam_dys)
        gam_dys.fit(X_dys, y_dys)

        residuals_dys = y_dys - gam_dys.predict(X_dys)
        dys_r2 = gam_dys.statistics_["pseudo_r2"]["explained_deviance"]
        logger.info(f"    Final dyslexic R²: {dys_r2:.3f}")
        logger.info(
            f"    Final dyslexic EDF: {np.sum(gam_dys.statistics_['edof']):.1f}"
        )

        self.duration_model = {"control": gam_ctrl, "dyslexic": gam_dys}

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

        logger.info("\n✓ Duration models fitted successfully")
        logger.info(f"  Control: R²={ctrl_r2:.3f} (CV RMSE: {cv_rmse_ctrl:.3f})")
        logger.info(f"  Dyslexic: R²={dys_r2:.3f} (CV RMSE: {cv_rmse_dys:.3f})")

        return {
            "family": "Gaussian (log)" if use_log_transform else "Gamma",
            "n_obs_control": int(len(X_ctrl)),
            "n_obs_dyslexic": int(len(X_dys)),
            "r2_control": float(ctrl_r2),
            "r2_dyslexic": float(dys_r2),
            "cv_rmse_control": float(cv_rmse_ctrl),
            "cv_rmse_dyslexic": float(cv_rmse_dys),
            "n_splines_control": int(n_sp_ctrl),
            "n_splines_dyslexic": int(n_sp_dys),
            "lambda_control": float(lam_ctrl),
            "lambda_dyslexic": float(lam_dys),
            "edf_control": float(np.sum(gam_ctrl.statistics_["edof"])),
            "edf_dyslexic": float(np.sum(gam_dys.statistics_["edof"])),
            "has_tensor_product": False,
            "uses_smearing_factor": use_log_transform,
            "method": "additive_smooths_only",
        }

    def _flatten_lams(self, lam):
        """Recursively flatten a possibly nested lam structure"""
        out = []
        stack = [lam]
        while stack:
            x = stack.pop()
            if isinstance(x, (list, tuple, np.ndarray)):
                stack.extend(list(x))
            else:
                out.append(float(x))
        return out

    def _geomean(self, vals):
        """Geometric mean with epsilon"""
        vals = np.asarray(vals, dtype=float)
        eps = 1e-12
        return float(np.exp(np.mean(np.log(vals + eps))))

    def _preselect_hyperparameters(
        self, X, y, subjects, n_splines_search, lam_search, model_type="skip"
    ):
        """Pre-select hyperparameters on subject subsample"""
        logger.info("    Stage 1: Pre-selecting hyperparameters...")

        rng = np.random.default_rng(42)

        unique_subjects = np.unique(subjects)
        n_total = unique_subjects.size

        target = max(30, n_total // 3)
        n_subsample = min(n_total, max(3, min(120, target)))

        subsample_subjects = rng.choice(unique_subjects, n_subsample, replace=False)
        mask = np.isin(subjects, subsample_subjects)

        X_sub, y_sub, subj_sub = X[mask], y[mask], subjects[mask]
        logger.info(
            f"      Using {len(subsample_subjects)} of {n_total} subjects ({len(X_sub):,} obs)"
        )

        n_splits = max(2, min(3, np.unique(subj_sub).size))
        gkf = GroupKFold(n_splits=n_splits)

        results = []
        lam_history = {}

        for n_sp in tqdm(
            n_splines_search, desc="      Pre-select n_splines", leave=False
        ):
            fold_scores, fold_lams = [], []

            for tr, va in gkf.split(X_sub, y_sub, groups=subj_sub):
                if model_type == "skip":
                    gam = LogisticGAM(
                        s(0, n_splines=n_sp)
                        + s(1, n_splines=n_sp, constraints="monotonic_inc")  # Zipf ↑
                        + s(2, n_splines=n_sp)
                    )
                    gam.gridsearch(X_sub[tr], y_sub[tr], lam=lam_search, progress=False)
                    p = gam.predict_proba(X_sub[va])
                    fold_scores.append(roc_auc_score(y_sub[va], p))
                else:
                    gam = LinearGAM(
                        s(0, n_splines=n_sp)
                        + s(1, n_splines=n_sp, constraints="monotonic_dec")  # Zipf ↓
                        + s(2, n_splines=n_sp)
                    )
                    gam.gridsearch(X_sub[tr], y_sub[tr], lam=lam_search, progress=False)
                    yhat = gam.predict(X_sub[va])
                    fold_scores.append(np.sqrt(np.mean((y_sub[va] - yhat) ** 2)))

                lam_values = self._flatten_lams(gam.lam)
                lam_chosen = self._geomean(lam_values)
                fold_lams.append(lam_chosen)

            lam_history[n_sp] = fold_lams

            mean_score = np.mean(fold_scores)
            se_score = np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores))
            results.append({"n_sp": n_sp, "mean": mean_score, "se": se_score})

        # 1-SE rule
        if model_type == "skip":
            best = max(results, key=lambda r: r["mean"])
            threshold = best["mean"] - best["se"]
            within = [r for r in results if r["mean"] >= threshold]
        else:
            best = min(results, key=lambda r: r["mean"])
            threshold = best["mean"] + best["se"]
            within = [r for r in results if r["mean"] <= threshold]

        selected = min(within, key=lambda r: r["n_sp"])
        n_sp_star = selected["n_sp"]

        lams_raw = lam_history.get(n_sp_star, [])
        if len(lams_raw) == 0:
            lam_star = float(np.asarray(lam_search).flatten()[0])
        else:
            flat = []
            for lam_val in lams_raw:
                flat.extend(self._flatten_lams(lam_val))
            target = self._geomean(flat)
            lam_star = float(
                lam_search[np.argmin(np.abs(np.log(lam_search) - np.log(target)))]
            )

        logger.info(f"      ✓ Selected: n_splines={n_sp_star}, λ≈{lam_star:.3g}")

        return n_sp_star, lam_star

    def _validate_with_frozen_hyperparameters(
        self, X, y, subjects, n_splines, lam_search, model_type="skip"
    ):
        """Stage 2: Validate with 10-fold CV using pre-selected hyperparameters"""
        logger.info(
            f"    Stage 2: 10-fold CV validation (n_splines={n_splines} frozen)..."
        )

        uniq_subj = np.unique(subjects).size
        n_splits = max(2, min(10, uniq_subj))
        gkf = GroupKFold(n_splits=n_splits)

        fold_indices = list(gkf.split(X, y, groups=subjects))

        fold_scores = Parallel(n_jobs=-1, verbose=0, prefer="processes")(
            delayed(self._fit_single_fold)(
                X, y, tr, va, n_splines, lam_search, model_type
            )
            for tr, va in tqdm(
                fold_indices, total=n_splits, desc="      CV folds", leave=False
            )
        )

        mean_score = float(np.mean(fold_scores))
        se_score = float(np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores)))

        if model_type == "skip":
            logger.info(f"      CV AUC: {mean_score:.4f} (±{se_score:.4f})")
        else:
            logger.info(f"      CV RMSE: {mean_score:.4f} (±{se_score:.4f})")

        return mean_score


def fit_gam_models(
    data: pd.DataFrame, use_log_duration: bool = True, quick_mode: bool = False
) -> Tuple[Dict, Dict, DyslexiaGAMModels]:
    """
    Fit both GAM models (skip and duration) with additive smooths only
    """
    gam = DyslexiaGAMModels()

    skip_metadata = gam.fit_skip_model(data, quick_mode=quick_mode)
    duration_metadata = gam.fit_duration_model(
        data, use_log_transform=use_log_duration, quick_mode=quick_mode
    )

    return skip_metadata, duration_metadata, gam
