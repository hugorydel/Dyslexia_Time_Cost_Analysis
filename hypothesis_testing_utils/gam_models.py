"""
GAM Models with Tensor Product Interactions - V1 TWEAKED
Key changes from original:
1. Removed upfront full-data gridsearch (eliminates label leakage)
2. Added 1-SE rule for model selection
3. Added EDF saturation checks
"""

import logging
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pygam import LinearGAM, LogisticGAM, s, te
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DyslexiaGAMModels:
    """
    GAM models with tensor product interactions
    Uses proper nested CV with 1-SE rule
    """

    def __init__(self):
        self.skip_model = None
        self.duration_model = None
        self.feature_means = None
        self.smearing_factors = None
        logger.info("GAM Models initialized with tensor product support")

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
        if model_type == "skip":
            gam = LogisticGAM(
                s(0, n_splines=n_splines)
                + s(1, n_splines=n_splines)
                + s(2, n_splines=n_splines)
                + te(0, 1, n_splines=[n_splines, n_splines])
            )
        else:
            gam = LinearGAM(
                s(0, n_splines=n_splines)
                + s(1, n_splines=n_splines)
                + s(2, n_splines=n_splines)
                + te(0, 1, n_splines=[n_splines, n_splines])
            )

        # decide: fixed λ vs gridsearch
        def _is_single_lam(lam):
            if np.isscalar(lam):
                return True
            try:
                seq = list(lam)
            except TypeError:
                return True
            return len(seq) == 1 and not hasattr(seq[0], "__iter__")

        if _is_single_lam(lam_search):
            lam_val = float(np.atleast_1d(lam_search)[0])  # broadcast inside pyGAM
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
        Fit skip model with TWO-STAGE approach:
        Stage 1: Pre-select n_splines on subsample (3-fold CV)
        Stage 2: Validate on full data (10-fold CV with frozen n_splines & λ)

        skip ~ s(length) + s(zipf) + s(surprisal) + te(length, zipf)
        """
        logger.info("=" * 60)
        logger.info("FITTING SKIP MODEL (Logistic GAM + Tensor Product)")
        logger.info("=" * 60)

        # Set defaults
        if n_splines_search is None:
            n_splines_search = [8, 10] if not quick_mode else [10]

        if lam_search is None:
            lam_search = (
                np.logspace(-3, 3, 7) if not quick_mode else np.logspace(-2, 2, 5)
            )

        if quick_mode:
            logger.info("  ⚡ QUICK MODE enabled")
            logger.info(f"     n_splines: {n_splines_search}")
            logger.info(f"     lambda: {len(lam_search)} values")

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

        # Stage 1: Pre-select hyperparameters (n_splines & λ)
        n_sp_ctrl, lam_ctrl = self._preselect_hyperparameters(
            X_ctrl, y_ctrl, subj_ctrl, n_splines_search, lam_search, "skip"
        )

        # Stage 2: Validate with frozen hyperparameters
        # NOTE: pass a 2-length duplicate λ grid so pyGAM.gridsearch accepts it,
        # even if _fit_single_fold still calls gridsearch internally.
        lam_grid_dupe = np.array([lam_ctrl, lam_ctrl], dtype=float)
        cv_auc_ctrl = self._validate_with_frozen_hyperparameters(
            X_ctrl, y_ctrl, subj_ctrl, n_sp_ctrl, lam_grid_dupe, "skip"
        )

        # Final fit on all data (freeze λ) -> set lam and .fit()
        logger.info("    Fitting final model on all control data...")
        gam_ctrl = LogisticGAM(
            s(0, n_splines=n_sp_ctrl)
            + s(1, n_splines=n_sp_ctrl)
            + s(2, n_splines=n_sp_ctrl)
            + te(0, 1, n_splines=[n_sp_ctrl, n_sp_ctrl])
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

        # Stage 1: Pre-select hyperparameters (n_splines & λ)
        n_sp_dys, lam_dys = self._preselect_hyperparameters(
            X_dys, y_dys, subj_dys, n_splines_search, lam_search, "skip"
        )

        # Stage 2: Validate with frozen hyperparameters (duplicate λ trick again)
        lam_grid_dupe = np.array([lam_dys, lam_dys], dtype=float)
        cv_auc_dys = self._validate_with_frozen_hyperparameters(
            X_dys, y_dys, subj_dys, n_sp_dys, lam_grid_dupe, "skip"
        )

        # Final fit on all data (freeze λ)
        logger.info("    Fitting final model on all dyslexic data...")
        gam_dys = LogisticGAM(
            s(0, n_splines=n_sp_dys)
            + s(1, n_splines=n_sp_dys)
            + s(2, n_splines=n_sp_dys)
            + te(0, 1, n_splines=[n_sp_dys, n_sp_dys])
        )
        gam_dys.lam = float(lam_dys)
        gam_dys.fit(X_dys, y_dys)

        dys_pred = gam_dys.predict_proba(X_dys)
        dys_auc = roc_auc_score(y_dys, dys_pred)
        logger.info(f"    Final dyslexic AUC: {dys_auc:.3f}")
        logger.info(
            f"    Final dyslexic EDF: {np.sum(gam_dys.statistics_['edof']):.1f}"
        )

        # Store models
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
            "has_tensor_product": True,
            "method": "two_stage_preselection",
        }

    def _fit_skip_model_cv(self, X, y, subjects, n_splines_search, lam_search):
        """
        Logistic skip model — nested CV + 1-SE rule on AUC.
        Returns a refit LogisticGAM on all X,y with the selected n_splines and λ.
        """
        uniq_subj = np.unique(subjects).size
        n_splits = max(2, min(10, uniq_subj))

        results = []
        for n_sp in tqdm(n_splines_search, desc="    Grid search (skip)", leave=False):
            gkf = GroupKFold(n_splits=n_splits)
            fold_aucs, fold_edfs = [], []

            for tr, va in tqdm(
                gkf.split(X, y, groups=subjects),
                total=n_splits,
                desc=f"      CV folds (n_sp={n_sp})",
                leave=False,
            ):
                gam = LogisticGAM(
                    s(0, n_splines=n_sp)
                    + s(1, n_splines=n_sp)
                    + s(2, n_splines=n_sp)
                    + te(0, 1, n_splines=[n_sp, n_sp])
                )
                # λ tuned on TRAIN only
                gam.gridsearch(X[tr], y[tr], lam=lam_search, progress=False)

                p = gam.predict_proba(X[va])
                fold_aucs.append(roc_auc_score(y[va], p))
                fold_edfs.append(float(np.sum(gam.statistics_["edof"])))

            mean_auc = float(np.mean(fold_aucs))
            se_auc = float(np.std(fold_aucs, ddof=1) / np.sqrt(len(fold_aucs)))
            mean_edf = float(np.mean(fold_edfs))

            results.append(
                dict(n_sp=n_sp, mean_auc=mean_auc, se_auc=se_auc, mean_edf=mean_edf)
            )
            logger.info(
                f"        n_splines={n_sp}: CV AUC={mean_auc:.4f} (±{se_auc:.4f}), EDF≈{mean_edf:.1f}"
            )

        # 1-SE rule on AUC (pick smallest n_sp within 1-SE of best)
        best = max(results, key=lambda r: r["mean_auc"])
        threshold = best["mean_auc"] - best["se_auc"]
        within = [r for r in results if r["mean_auc"] >= threshold]
        selected = min(within, key=lambda r: r["n_sp"])

        approx_max_edf = selected["n_sp"] * 3 + selected["n_sp"] * selected["n_sp"]
        if selected["mean_edf"] > 0.9 * approx_max_edf:
            logger.warning(
                f"      ⚠ EDF near saturation ({selected['mean_edf']:.1f} / {approx_max_edf})"
            )
        else:
            logger.info(
                f"      ✓ EDF OK ({selected['mean_edf']:.1f} / {approx_max_edf})"
            )

        logger.info(
            f"    Refitting full-data skip model with n_splines={selected['n_sp']}..."
        )
        final = LogisticGAM(
            s(0, n_splines=selected["n_sp"])
            + s(1, n_splines=selected["n_sp"])
            + s(2, n_splines=selected["n_sp"])
            + te(0, 1, n_splines=[selected["n_sp"], selected["n_sp"]])
        )
        final.gridsearch(X, y, lam=lam_search, progress=False)
        logger.info(
            f"    Final skip EDF={np.sum(final.statistics_['edof']):.1f}, λ={final.lam}"
        )
        return final

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

        # Cast X to float32 for memory efficiency (no precision loss for GAMs)
        X_filtered = X[valid_mask].astype(np.float32)
        return X_filtered, y[valid_mask], group[valid_mask], subjects[valid_mask]

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

        # Cast X to float32 for memory efficiency (no precision loss for GAMs)
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

        # Apply smearing factor if log-transform was used
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
        Fit duration model with TWO-STAGE approach:
        Stage 1: Pre-select n_splines on subsample (3-fold CV)
        Stage 2: Validate on full data (10-fold CV with frozen n_splines & λ)

        log(TRT) ~ s(length) + s(zipf) + s(surprisal) + te(length, zipf)
        """
        logger.info("=" * 60)
        logger.info(
            f"FITTING DURATION MODEL ({'Log-Gaussian' if use_log_transform else 'Gamma'} GAM)"
        )
        logger.info("=" * 60)

        # Set defaults
        if n_splines_search is None:
            n_splines_search = [8, 10] if not quick_mode else [10]

        if lam_search is None:
            lam_search = (
                np.logspace(-3, 3, 7) if not quick_mode else np.logspace(-2, 2, 5)
            )

        if quick_mode:
            logger.info("  ⚡ QUICK MODE enabled")
            logger.info(f"     n_splines: {n_splines_search}")
            logger.info(f"     lambda: {len(lam_search)} values")

        # Filter to fixated words
        fixated = data[data["skip"] == 0].copy()
        logger.info(f"  Fitting on {len(fixated):,} fixated words")

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

        # === CONTROL MODEL ===
        logger.info("\n  Fitting control model...")

        # Stage 1: Pre-select hyperparameters (n_splines & λ)
        n_sp_ctrl, lam_ctrl = self._preselect_hyperparameters(
            X_ctrl, y_ctrl, subj_ctrl, n_splines_search, lam_search, "duration"
        )

        # Stage 2: Validate with frozen hyperparameters
        lam_grid_dupe = np.array([lam_ctrl, lam_ctrl], dtype=float)
        cv_rmse_ctrl = self._validate_with_frozen_hyperparameters(
            X_ctrl, y_ctrl, subj_ctrl, n_sp_ctrl, lam_grid_dupe, "duration"
        )

        # Final fit on all data (freeze λ)
        logger.info("    Fitting final model on all control data...")
        gam_ctrl = LinearGAM(
            s(0, n_splines=n_sp_ctrl)
            + s(1, n_splines=n_sp_ctrl)
            + s(2, n_splines=n_sp_ctrl)
            + te(0, 1, n_splines=[n_sp_ctrl, n_sp_ctrl])
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

        # Stage 1: Pre-select hyperparameters (n_splines & λ)
        n_sp_dys, lam_dys = self._preselect_hyperparameters(
            X_dys, y_dys, subj_dys, n_splines_search, lam_search, "duration"
        )

        # Stage 2: Validate with frozen hyperparameters
        lam_grid_dupe = np.array([lam_dys, lam_dys], dtype=float)
        cv_rmse_dys = self._validate_with_frozen_hyperparameters(
            X_dys, y_dys, subj_dys, n_sp_dys, lam_grid_dupe, "duration"
        )

        # Final fit on all data (freeze λ)
        logger.info("    Fitting final model on all dyslexic data...")
        gam_dys = LinearGAM(
            s(0, n_splines=n_sp_dys)
            + s(1, n_splines=n_sp_dys)
            + s(2, n_splines=n_sp_dys)
            + te(0, 1, n_splines=[n_sp_dys, n_sp_dys])
        )
        gam_dys.lam = float(lam_dys)
        gam_dys.fit(X_dys, y_dys)

        residuals_dys = y_dys - gam_dys.predict(X_dys)
        dys_r2 = gam_dys.statistics_["pseudo_r2"]["explained_deviance"]
        logger.info(f"    Final dyslexic R²: {dys_r2:.3f}")
        logger.info(
            f"    Final dyslexic EDF: {np.sum(gam_dys.statistics_['edof']):.1f}"
        )

        # Store models
        self.duration_model = {"control": gam_ctrl, "dyslexic": gam_dys}

        # Compute smearing factors if log-transform used
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
            "has_tensor_product": True,
            "uses_smearing_factor": use_log_transform,
            "method": "two_stage_preselection",
        }

    def _flatten_lams(self, lam):
        """Recursively flatten a possibly nested lam structure into a list of floats."""
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
        """Geometric mean with a tiny epsilon for numerical safety."""
        vals = np.asarray(vals, dtype=float)
        eps = 1e-12
        return float(np.exp(np.mean(np.log(vals + eps))))

    def _preselect_hyperparameters(
        self, X, y, subjects, n_splines_search, lam_search, model_type="skip"
    ):
        """
        Pre-select hyperparameters on subject subsample (Stage 1)
        Returns: (best_n_splines, best_lam)
        """
        logger.info("    Stage 1: Pre-selecting hyperparameters...")

        rng = np.random.default_rng(42)

        unique_subjects = np.unique(subjects)
        n_total = unique_subjects.size

        # target ~ max(30, n_total//3), capped in [3, 120], but never > n_total
        target = max(30, n_total // 3)
        n_subsample = min(n_total, max(3, min(120, target)))

        subsample_subjects = rng.choice(unique_subjects, n_subsample, replace=False)
        mask = np.isin(subjects, subsample_subjects)

        X_sub, y_sub, subj_sub = X[mask], y[mask], subjects[mask]
        logger.info(
            f"      Using {len(subsample_subjects)} of {n_total} subjects ({len(X_sub):,} obs)"
        )

        # CV folds must not exceed number of groups; need at least 2
        n_splits = max(2, min(3, np.unique(subj_sub).size))
        if n_splits < 2:
            raise ValueError(
                f"Not enough subjects for CV in this group (got {np.unique(subj_sub).size})."
            )
        gkf = GroupKFold(n_splits=n_splits)

        results = []
        lam_history = {}  # store per-fold best λ for each n_splines

        for n_sp in tqdm(
            n_splines_search, desc="      Pre-select n_splines", leave=False
        ):
            fold_scores, fold_lams = [], []

            for tr, va in gkf.split(X_sub, y_sub, groups=subj_sub):
                if model_type == "skip":
                    gam = LogisticGAM(
                        s(0, n_splines=n_sp)
                        + s(1, n_splines=n_sp)
                        + s(2, n_splines=n_sp)
                        + te(0, 1, n_splines=[n_sp, n_sp])
                    )
                    gam.gridsearch(X_sub[tr], y_sub[tr], lam=lam_search, progress=False)
                    p = gam.predict_proba(X_sub[va])
                    fold_scores.append(roc_auc_score(y_sub[va], p))
                else:  # duration
                    gam = LinearGAM(
                        s(0, n_splines=n_sp)
                        + s(1, n_splines=n_sp)
                        + s(2, n_splines=n_sp)
                        + te(0, 1, n_splines=[n_sp, n_sp])
                    )
                    gam.gridsearch(X_sub[tr], y_sub[tr], lam=lam_search, progress=False)
                    yhat = gam.predict(X_sub[va])
                    fold_scores.append(np.sqrt(np.mean((y_sub[va] - yhat) ** 2)))

                # record best λ chosen by gridsearch on this fold
                lam_values = self._flatten_lams(gam.lam)
                lam_chosen = self._geomean(lam_values)
                fold_lams.append(lam_chosen)

            lam_history[n_sp] = fold_lams

            mean_score = np.mean(fold_scores)
            se_score = np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores))
            results.append({"n_sp": n_sp, "mean": mean_score, "se": se_score})

            if model_type == "skip":
                logger.info(
                    f"        n_splines={n_sp}: AUC={mean_score:.4f} (±{se_score:.4f})"
                )
            else:
                logger.info(
                    f"        n_splines={n_sp}: RMSE={mean_score:.4f} (±{se_score:.4f})"
                )

        # 1-SE rule
        if model_type == "skip":
            best = max(results, key=lambda r: r["mean"])
            threshold = best["mean"] - best["se"]
            within = [r for r in results if r["mean"] >= threshold]
        else:  # duration (minimize RMSE)
            best = min(results, key=lambda r: r["mean"])
            threshold = best["mean"] + best["se"]
            within = [r for r in results if r["mean"] <= threshold]

        selected = min(within, key=lambda r: r["n_sp"])
        n_sp_star = selected["n_sp"]

        # Aggregate λ across folds for the chosen n_splines:
        lams_raw = lam_history.get(n_sp_star, [])
        if len(lams_raw) == 0:
            lam_star = float(np.asarray(lam_search).flatten()[0])
        else:
            flat = []
            for lam_val in lams_raw:
                flat.extend(self._flatten_lams(lam_val))
            target = self._geomean(flat)  # geometric “center” in log-space
            lam_star = float(
                lam_search[np.argmin(np.abs(np.log(lam_search) - np.log(target)))]
            )

        logger.info(f"      ✓ Selected: n_splines={n_sp_star}, λ≈{lam_star:.3g}")

        return n_sp_star, lam_star

    def _validate_with_frozen_hyperparameters(
        self, X, y, subjects, n_splines, lam_search, model_type="skip"
    ):
        """
        Stage 2: Validate with 10-fold CV using pre-selected hyperparameters.

        Args:
            X, y, subjects: Full data arrays
            n_splines: Pre-selected n_splines (frozen)
            lam_search: Array of lambda values (typically single pre-selected value)
            model_type: 'skip' or 'duration'

        Returns:
            mean_cv_score: Mean cross-validation score
        """
        logger.info(
            f"    Stage 2: 10-fold CV validation (n_splines={n_splines} frozen)..."
        )

        uniq_subj = np.unique(subjects).size
        n_splits = max(2, min(10, uniq_subj))
        gkf = GroupKFold(n_splits=n_splits)

        # Get all fold indices upfront for parallel processing
        fold_indices = list(gkf.split(X, y, groups=subjects))

        # Parallel processing of folds (n_jobs=-1 uses all cores)
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

    def _fit_duration_model_cv(self, X, y, subjects, n_splines_search, lam_search):
        """
        Linear (log-Gaussian) duration model — nested CV + 1-SE rule on RMSE.
        Returns (final LinearGAM refit on all data, residuals on y) for smearing.
        """
        uniq_subj = np.unique(subjects).size
        n_splits = max(2, min(10, uniq_subj))

        results = []
        for n_sp in tqdm(
            n_splines_search, desc="    Grid search (duration)", leave=False
        ):
            gkf = GroupKFold(n_splits=n_splits)
            fold_rmses, fold_edfs = [], []

            for tr, va in tqdm(
                gkf.split(X, y, groups=subjects),
                total=n_splits,
                desc=f"      CV folds (n_sp={n_sp})",
                leave=False,
            ):
                gam = LinearGAM(
                    s(0, n_splines=n_sp)
                    + s(1, n_splines=n_sp)
                    + s(2, n_splines=n_sp)
                    + te(0, 1, n_splines=[n_sp, n_sp])
                )
                # λ tuned on TRAIN only
                gam.gridsearch(X[tr], y[tr], lam=lam_search, progress=False)

                yhat = gam.predict(X[va])
                rmse = float(np.sqrt(np.mean((y[va] - yhat) ** 2)))
                fold_rmses.append(rmse)
                fold_edfs.append(float(np.sum(gam.statistics_["edof"])))

            mean_rmse = float(np.mean(fold_rmses))
            se_rmse = float(np.std(fold_rmses, ddof=1) / np.sqrt(len(fold_rmses)))
            mean_edf = float(np.mean(fold_edfs))

            results.append(
                dict(n_sp=n_sp, mean_rmse=mean_rmse, se_rmse=se_rmse, mean_edf=mean_edf)
            )
            logger.info(
                f"        n_splines={n_sp}: CV RMSE={mean_rmse:.4f} (±{se_rmse:.4f}), EDF≈{mean_edf:.1f}"
            )

        # 1-SE rule on RMSE (pick smallest n_sp within 1-SE of the best/lowest RMSE)
        best = min(results, key=lambda r: r["mean_rmse"])
        threshold = best["mean_rmse"] + best["se_rmse"]
        within = [r for r in results if r["mean_rmse"] <= threshold]
        selected = min(within, key=lambda r: r["n_sp"])

        approx_max_edf = selected["n_sp"] * 3 + selected["n_sp"] * selected["n_sp"]
        if selected["mean_edf"] > 0.9 * approx_max_edf:
            logger.warning(
                f"      ⚠ EDF near saturation ({selected['mean_edf']:.1f} / {approx_max_edf})"
            )
        else:
            logger.info(
                f"      ✓ EDF OK ({selected['mean_edf']:.1f} / {approx_max_edf})"
            )

        logger.info(
            f"    Refitting full-data duration model with n_splines={selected['n_sp']}..."
        )
        final = LinearGAM(
            s(0, n_splines=selected["n_sp"])
            + s(1, n_splines=selected["n_sp"])
            + s(2, n_splines=selected["n_sp"])
            + te(0, 1, n_splines=[selected["n_sp"], selected["n_sp"]])
        )
        final.gridsearch(X, y, lam=lam_search, progress=False)

        residuals = y - final.predict(X)  # used for smearing if log-Gaussian
        logger.info(
            f"    Final duration EDF={np.sum(final.statistics_['edof']):.1f}, λ={final.lam}"
        )
        return final, residuals


def fit_gam_models(
    data: pd.DataFrame, use_log_duration: bool = True, quick_mode: bool = False
) -> Tuple[Dict, Dict, DyslexiaGAMModels]:
    """
    Fit both GAM models with two-stage hyperparameter selection.

    Args:
        data: Prepared data
        use_log_duration: Use log-Gaussian (True) or Gamma (False) for duration
        quick_mode: Use faster settings for testing/development

    Returns:
        (skip_metadata, duration_metadata, gam_instance) tuple
    """
    gam = DyslexiaGAMModels()

    skip_metadata = gam.fit_skip_model(data, quick_mode=quick_mode)
    duration_metadata = gam.fit_duration_model(
        data, use_log_transform=use_log_duration, quick_mode=quick_mode
    )

    return skip_metadata, duration_metadata, gam
