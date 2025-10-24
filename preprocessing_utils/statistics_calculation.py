"""
Statistical analysis utilities for dyslexia research
"""

import logging
from typing import Any, Dict, List, Optional  # <-- MODIFIED

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """
    Calculate Cohen's d effect size between two groups

    Args:
        group1: First group data
        group2: Second group data

    Returns:
        Cohen's d effect size
    """
    if len(group1) == 0 or len(group2) == 0:
        return np.nan

    # Remove NaN values
    group1 = group1.dropna()
    group2 = group2.dropna()

    if len(group1) == 0 or len(group2) == 0:
        return np.nan

    # Calculate pooled standard deviation
    pooled_std = np.sqrt(
        ((len(group1) - 1) * group1.std() ** 2 + (len(group2) - 1) * group2.std() ** 2)
        / (len(group1) + len(group2) - 2)
    )

    if pooled_std == 0:
        return np.nan

    # Calculate Cohen's d
    cohens_d = (group2.mean() - group1.mean()) / pooled_std
    return cohens_d


def calculate_group_summary_stats(
    data: pd.DataFrame, skipping_analysis: Dict = None
) -> Dict[str, Any]:
    """
    Calculate summary statistics by group for key measures

    Args:
        data: DataFrame with dyslexic column and eye-tracking measures
        skipping_analysis: Optional enhanced skipping analysis results

    Returns:
        Dictionary with group statistics for each measure
    """
    if "dyslexic" not in data.columns:
        logger.warning("No dyslexic column found - cannot calculate group statistics")
        return {}

    group_stats = {}

    # Word-level measures by group
    word_level_measures = [
        "total_reading_time",
        "first_fixation_duration",
        "gaze_duration",
        "word_go_past_time",
        "n_fixations",
        "regression_probability",
    ]

    # Add skipping_probability to measures if we have enhanced analysis
    if skipping_analysis and "skipping_by_group" in skipping_analysis:
        word_level_measures.append("skipping_probability")

    for measure in word_level_measures:
        if measure == "skipping_probability" and skipping_analysis:
            # Use enhanced skipping analysis results instead of data
            skipping_by_group = skipping_analysis.get("skipping_by_group", {})

            control_stats = skipping_by_group.get("control", {})
            dyslexic_stats = skipping_by_group.get("dyslexic", {})

            group_stats[f"{measure}_by_group"] = {
                **{f"control_{k}": v for k, v in control_stats.items()},
                **{f"dyslexic_{k}": v for k, v in dyslexic_stats.items()},
            }

            # Add Cohen's d if we have both groups
            if control_stats and dyslexic_stats:
                # Calculate Cohen's d from group means and stds
                control_mean = control_stats.get("mean", 0)
                dyslexic_mean = dyslexic_stats.get("mean", 0)
                control_std = control_stats.get("std", 0)
                dyslexic_std = dyslexic_stats.get("std", 0)
                control_n = control_stats.get("count", 0)
                dyslexic_n = dyslexic_stats.get("count", 0)

                if (
                    control_n > 0
                    and dyslexic_n > 0
                    and (control_std > 0 or dyslexic_std > 0)
                ):
                    # Pooled standard deviation
                    pooled_std = np.sqrt(
                        (
                            (control_n - 1) * control_std**2
                            + (dyslexic_n - 1) * dyslexic_std**2
                        )
                        / (control_n + dyslexic_n - 2)
                    )
                    if pooled_std > 0:
                        cohens_d = (dyslexic_mean - control_mean) / pooled_std
                        group_stats[f"{measure}_by_group"]["cohens_d"] = float(cohens_d)

        elif measure in data.columns:
            # Use regular data for other measures
            group_summary = (
                data.groupby("dyslexic")[measure].agg(["mean", "std", "count"]).round(2)
            )

            # Extract values safely
            control_stats = {}
            dyslexic_stats = {}

            if False in group_summary.index:
                control_stats = {
                    "control_mean": float(group_summary.loc[False, "mean"]),
                    "control_std": float(group_summary.loc[False, "std"]),
                    "control_count": int(group_summary.loc[False, "count"]),
                }

            if True in group_summary.index:
                dyslexic_stats = {
                    "dyslexic_mean": float(group_summary.loc[True, "mean"]),
                    "dyslexic_std": float(group_summary.loc[True, "std"]),
                    "dyslexic_count": int(group_summary.loc[True, "count"]),
                }

            group_stats[f"{measure}_by_group"] = {**control_stats, **dyslexic_stats}

            # Calculate effect size (Cohen's d) if both groups exist
            if False in group_summary.index and True in group_summary.index:
                control_data = data[data["dyslexic"] == False][measure]
                dyslexic_data = data[data["dyslexic"] == True][measure]

                cohens_d = calculate_cohens_d(control_data, dyslexic_data)
                if not np.isnan(cohens_d):
                    group_stats[f"{measure}_by_group"]["cohens_d"] = float(cohens_d)

    # Subject-level measures by group (averaged within subjects first)
    subject_level = (
        data.groupby(["subject_id", "dyslexic"])
        .agg(
            {
                measure: "mean"
                for measure in word_level_measures
                if measure in data.columns and measure != "skipping_probability"
            }
        )
        .reset_index()
    )

    for measure in word_level_measures:
        if measure == "skipping_probability" and skipping_analysis:
            # Use enhanced skipping analysis for subject-level skipping
            subject_skipping = skipping_analysis.get(
                "subject_level_skipping_by_group", {}
            )

            control_stats = subject_skipping.get("control", {})
            dyslexic_stats = subject_skipping.get("dyslexic", {})

            group_stats[f"{measure}_subject_level_by_group"] = {
                **{f"control_{k}": v for k, v in control_stats.items()},
                **{f"dyslexic_{k}": v for k, v in dyslexic_stats.items()},
            }

        elif measure in subject_level.columns:
            subj_group_summary = (
                subject_level.groupby("dyslexic")[measure]
                .agg(["mean", "std", "count"])
                .round(2)
            )

            # Extract values safely
            control_stats = {}
            dyslexic_stats = {}

            if False in subj_group_summary.index:
                control_stats = {
                    "control_mean": float(subj_group_summary.loc[False, "mean"]),
                    "control_std": float(subj_group_summary.loc[False, "std"]),
                    "control_n_subjects": int(subj_group_summary.loc[False, "count"]),
                }

            if True in subj_group_summary.index:
                dyslexic_stats = {
                    "dyslexic_mean": float(subj_group_summary.loc[True, "mean"]),
                    "dyslexic_std": float(subj_group_summary.loc[True, "std"]),
                    "dyslexic_n_subjects": int(subj_group_summary.loc[True, "count"]),
                }

            group_stats[f"{measure}_subject_level_by_group"] = {
                **control_stats,
                **dyslexic_stats,
            }

    return group_stats


def calculate_basic_statistics(
    data: pd.DataFrame, skipping_analysis: Dict = None
) -> Dict[str, Any]:
    """
    Calculate basic descriptive statistics for the dataset

    Args:
        data: DataFrame with eye-tracking data
        skipping_analysis: Optional enhanced skipping analysis results

    Returns:
        Dictionary with basic statistics
    """
    stats = {
        "data_shape": [int(x) for x in data.shape],
        "columns": list(data.columns),
        "subjects": int(data["subject_id"].nunique()),
    }

    # Data quality metrics (more useful than misleading missing data counts)
    if "n_fixations" in data.columns:
        stats["data_quality"] = {
            "total_words": len(data),
            "skipped_words": int((data["n_fixations"] == 0).sum()),
            "fixated_words": int((data["n_fixations"] > 0).sum()),
            "skipping_rate": float((data["n_fixations"] == 0).mean()),
        }

        # Only report genuinely concerning missing data
        concerning_missing = {}
        critical_id_cols = ["subject_id", "word_text", "trial_id", "word_position"]
        for col in critical_id_cols:
            if col in data.columns:
                missing_count = data[col].isna().sum()
                if missing_count > 0:
                    concerning_missing[col] = int(missing_count)

        if concerning_missing:
            stats["data_quality"]["concerning_missing_data"] = concerning_missing
        else:
            stats["data_quality"][
                "concerning_missing_data"
            ] = "None - all critical identifiers present"

    # Trial and word statistics
    if "trial_id" in data.columns:
        stats["trials"] = int(data["trial_id"].nunique())

    # Group statistics
    if "dyslexic" in data.columns:
        subject_groups = data.groupby("subject_id")["dyslexic"].first()
        stats["dyslexic_subjects"] = int(subject_groups.sum())
        stats["control_subjects"] = int(len(subject_groups) - subject_groups.sum())
        stats["dyslexic_proportion"] = float(subject_groups.mean())

    # Word and reading time statistics
    if "word_length" in data.columns:
        stats["word_stats"] = {
            "mean_length": float(data["word_length"].mean()),
            "std_length": float(data["word_length"].std()),
            "min_length": int(data["word_length"].min()),
            "max_length": int(data["word_length"].max()),
        }

    # Reading time stats - only for fixated words
    if "total_reading_time" in data.columns and "n_fixations" in data.columns:
        fixated_data = data[data["n_fixations"] > 0]["total_reading_time"]
        if len(fixated_data) > 0:
            stats["reading_time_stats"] = {
                "mean": float(fixated_data.mean()),
                "std": float(fixated_data.std()),
                "min": float(fixated_data.min()),
                "max": float(fixated_data.max()),
                "note": "Statistics for fixated words only (skipped words excluded)",
            }

    # Add skipping statistics if enhanced analysis was performed
    if skipping_analysis:
        stats["skipping_analysis"] = {
            "enhanced_analysis_performed": True,
            "overall_skipping_rate": float(
                skipping_analysis.get("overall_skipping_rate", 0)
            ),
            "total_words_analyzed": int(skipping_analysis.get("total_words", 0)),
            "fixated_words": int(skipping_analysis.get("fixated_words", 0)),
            "skipped_words": int(skipping_analysis.get("skipped_words", 0)),
        }

        # Add group-level skipping statistics if available
        if "skipping_by_group" in skipping_analysis:
            stats["skipping_analysis"]["by_group"] = skipping_analysis[
                "skipping_by_group"
            ]

        if "subject_level_skipping_by_group" in skipping_analysis:
            stats["skipping_analysis"]["subject_level_by_group"] = skipping_analysis[
                "subject_level_skipping_by_group"
            ]

        logger.info(f"Added enhanced skipping statistics to basic stats")
        logger.info(
            f"Overall skipping rate: {stats['skipping_analysis']['overall_skipping_rate']*100:.1f}%"
        )
    else:
        stats["skipping_analysis"] = {
            "enhanced_analysis_performed": False,
            "note": "Enhanced skipping analysis not available - using basic measures from ExtractedFeatures only",
        }

        # Add basic skipping info from the data if available
        if "skipped" in data.columns:
            stats["skipping_analysis"]["basic_skipping_rate"] = float(
                data["skipped"].mean()
            )
        if "skipping_probability" in data.columns:
            stats["skipping_analysis"]["mean_skipping_probability"] = float(
                data["skipping_probability"].mean()
            )

    return stats


# ================================
# Comprehensive participant statistics
# ================================
def _safe_mode(series: pd.Series) -> Optional[Any]:
    s = series.dropna()
    if len(s) == 0:
        return None
    m = s.mode()
    return m.iloc[0] if len(m) else None


# ================================
# Comprehensive participant statistics
# ================================
def calculate_participant_statistics(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Build comprehensive participant-level statistics and aggregate summaries
    for Methods reporting. Works on the merged, standardized dataframe.

    Returns a JSON-serializable dict including:
      - n_participants, n_control, n_dyslexic, participant_ids_by_group
      - demographics (age, sex/native_language/vision distributions)
      - reading_and_screening (WPM, comprehension, tests, etc.)
      - exposure (words/sentences/speeches/paragraphs, skip rate)
      - eye_movement_baselines (TRT/GD/FFD/n_fixations/regression prob)
    """
    results: Dict[str, Any] = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "columns_available": list(data.columns),
    }

    if "subject_id" not in data.columns:
        logger.warning("No subject_id column; cannot compute participant statistics.")
        return results

    # Base subject frame (explicit, prevents index misalignment)
    per_subject = pd.DataFrame(
        {"subject_id": pd.Index(data["subject_id"].unique(), name="subject_id")}
    )

    gb = data.groupby("subject_id", observed=True)

    # --- Group flag (align via merge, never via index assignment) ---
    if "dyslexic" in data.columns:
        dys_df = gb["dyslexic"].first().rename("dyslexic").reset_index()
        per_subject = per_subject.merge(dys_df, on="subject_id", how="left")
        # Ensure plain bool (no pd.NA) for counts; unknowns set to False
        per_subject["dyslexic_bool"] = per_subject["dyslexic"].astype("boolean")
        per_subject["dyslexic_bool"] = (
            per_subject["dyslexic_bool"].fillna(False).astype(bool)
        )
    else:
        per_subject["dyslexic_bool"] = False

    # --- Demographic / screening fields ---
    demo_fields = [
        "age",
        "sex",
        "native_language",
        "vision",
        "comprehension_accuracy",
        "score_reading_comprehension_test",
        "pseudohomophone_score",
        "words_per_minute",
        "number_of_speeches",
        "number_of_questions",
        "absolute_reading_time",
        "relative_reading_time",
        "dyslexia",  # raw flag/string if present
    ]

    for col in demo_fields:
        if col not in data.columns:
            continue
        if col == "sex":
            tmp = gb[col].agg(_safe_mode).rename(col).reset_index()
        else:
            tmp = gb[col].first().rename(col).reset_index()
        per_subject = per_subject.merge(tmp, on="subject_id", how="left")

    # --- Exposure counts ---
    # n_words
    n_words = gb.size().rename("n_words").reset_index()
    per_subject = per_subject.merge(n_words, on="subject_id", how="left")

    # Unique counts
    for src, outcol in [
        ("speech_id", "n_speeches"),
        ("paragraph_id", "n_paragraphs"),
        ("sentence_id", "n_sentences"),
    ]:
        if src in data.columns:
            tmp = gb[src].nunique().rename(outcol).reset_index()
            per_subject = per_subject.merge(tmp, on="subject_id", how="left")

    # Fixated vs skipped counts
    if "n_fixations" in data.columns:
        fix_mask = data["n_fixations"] > 0
        tmp_fix = (
            fix_mask.groupby(data["subject_id"]).sum().rename("n_fixated").reset_index()
        )
        tmp_skip = (
            (~fix_mask)
            .groupby(data["subject_id"])
            .sum()
            .rename("n_skipped")
            .reset_index()
        )
        per_subject = per_subject.merge(tmp_fix, on="subject_id", how="left")
        per_subject = per_subject.merge(tmp_skip, on="subject_id", how="left")
        with np.errstate(invalid="ignore", divide="ignore"):
            per_subject["skipping_rate"] = (
                per_subject["n_skipped"] / per_subject["n_words"]
            )

    # --- Eye-movement baselines (means over fixated words) ---
    def _mean_by_subject_fixated(col: str) -> Optional[pd.DataFrame]:
        if col not in data.columns:
            return None
        d = data
        if "n_fixations" in d.columns:
            d = d[d["n_fixations"] > 0]
        if d.empty:
            return None
        out = d.groupby("subject_id")[col].mean().rename(col).reset_index()
        return out

    for src, outcol in [
        ("total_reading_time", "mean_total_reading_time_ms"),
        ("gaze_duration", "mean_gaze_duration_ms"),
        ("first_fixation_duration", "mean_first_fixation_duration_ms"),
    ]:
        tmp = _mean_by_subject_fixated(src)
        if tmp is not None:
            tmp = tmp.rename(columns={src: outcol})
            per_subject = per_subject.merge(tmp, on="subject_id", how="left")

    if "n_fixations" in data.columns:
        tmp = gb["n_fixations"].mean().rename("mean_n_fixations").reset_index()
        per_subject = per_subject.merge(tmp, on="subject_id", how="left")

    if "regression_probability" in data.columns:
        tmp = (
            gb["regression_probability"]
            .mean()
            .rename("mean_regression_probability")
            .reset_index()
        )
        per_subject = per_subject.merge(tmp, on="subject_id", how="left")

    # Round selected numeric columns (avoid NaN pollution)
    round_cols = [
        "skipping_rate",
        "mean_total_reading_time_ms",
        "mean_gaze_duration_ms",
        "mean_first_fixation_duration_ms",
        "mean_n_fixations",
        "mean_regression_probability",
        "words_per_minute",
        "comprehension_accuracy",
        "score_reading_comprehension_test",
        "pseudohomophone_score",
    ]
    for c in round_cols:
        if c in per_subject.columns:
            per_subject[c] = pd.to_numeric(per_subject[c], errors="coerce").round(3)

    # =========================
    # Aggregate summaries
    # =========================
    results["n_participants"] = int(per_subject.shape[0])
    results["n_control"] = int((~per_subject["dyslexic_bool"]).sum())
    results["n_dyslexic"] = int(per_subject["dyslexic_bool"].sum())
    results["participant_ids_by_group"] = {
        "control": sorted(
            per_subject.loc[~per_subject["dyslexic_bool"], "subject_id"]
            .astype(str)
            .tolist()
        ),
        "dyslexic": sorted(
            per_subject.loc[per_subject["dyslexic_bool"], "subject_id"]
            .astype(str)
            .tolist()
        ),
    }

    # Demographics
    demo_summary: Dict[str, Any] = {}

    if "age" in per_subject.columns:
        age_s = pd.to_numeric(per_subject["age"], errors="coerce")
        if age_s.notna().any():
            demo_summary["age"] = {
                "mean": float(age_s.mean()),
                "sd": float(age_s.std()),
                "min": float(age_s.min()),
                "max": float(age_s.max()),
                "n": int(age_s.notna().sum()),
            }
            demo_summary["age_by_group"] = {
                "control": {
                    "mean": float(
                        pd.to_numeric(
                            per_subject.loc[~per_subject["dyslexic_bool"], "age"],
                            errors="coerce",
                        ).mean()
                    ),
                    "sd": float(
                        pd.to_numeric(
                            per_subject.loc[~per_subject["dyslexic_bool"], "age"],
                            errors="coerce",
                        ).std()
                    ),
                    "n": int((~per_subject["dyslexic_bool"]).sum()),
                },
                "dyslexic": {
                    "mean": float(
                        pd.to_numeric(
                            per_subject.loc[per_subject["dyslexic_bool"], "age"],
                            errors="coerce",
                        ).mean()
                    ),
                    "sd": float(
                        pd.to_numeric(
                            per_subject.loc[per_subject["dyslexic_bool"], "age"],
                            errors="coerce",
                        ).std()
                    ),
                    "n": int((per_subject["dyslexic_bool"]).sum()),
                },
            }

    for cat in ["sex", "native_language", "vision"]:
        if cat in per_subject.columns:
            overall = per_subject[cat].fillna("Unknown")
            overall_counts = overall.value_counts().to_dict()
            demo_summary[cat] = {
                "overall": overall_counts,
                "by_group": {
                    "control": per_subject.loc[~per_subject["dyslexic_bool"], cat]
                    .fillna("Unknown")
                    .value_counts()
                    .to_dict(),
                    "dyslexic": per_subject.loc[per_subject["dyslexic_bool"], cat]
                    .fillna("Unknown")
                    .value_counts()
                    .to_dict(),
                },
            }

    results["demographics"] = demo_summary

    # Helper for numeric summaries (skip if no data to avoid NaN in JSON)
    def _num_summary(col: str) -> Optional[Dict[str, Any]]:
        if col not in per_subject.columns:
            return None
        s = pd.to_numeric(per_subject[col], errors="coerce")
        if s.notna().sum() == 0:
            return None
        out = {
            "mean": float(s.mean()),
            "sd": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "n": int(s.notna().sum()),
            "by_group": {
                "control": {
                    "mean": float(
                        pd.to_numeric(
                            per_subject.loc[~per_subject["dyslexic_bool"], col],
                            errors="coerce",
                        ).mean()
                    ),
                    "sd": float(
                        pd.to_numeric(
                            per_subject.loc[~per_subject["dyslexic_bool"], col],
                            errors="coerce",
                        ).std()
                    ),
                    "n": int((~per_subject["dyslexic_bool"]).sum()),
                },
                "dyslexic": {
                    "mean": float(
                        pd.to_numeric(
                            per_subject.loc[per_subject["dyslexic_bool"], col],
                            errors="coerce",
                        ).mean()
                    ),
                    "sd": float(
                        pd.to_numeric(
                            per_subject.loc[per_subject["dyslexic_bool"], col],
                            errors="coerce",
                        ).std()
                    ),
                    "n": int((per_subject["dyslexic_bool"]).sum()),
                },
            },
        }
        return out

    # Reading/screening block
    screening = {}
    for k in [
        "comprehension_accuracy",
        "score_reading_comprehension_test",
        "pseudohomophone_score",
        "words_per_minute",
        "number_of_speeches",
        "number_of_questions",
        "absolute_reading_time",
        "relative_reading_time",
    ]:
        s = _num_summary(k)
        if s is not None:
            screening[k] = s
    results["reading_and_screening"] = screening

    # Exposure block
    exposure_summary = {}
    for k in [
        "n_words",
        "n_speeches",
        "n_paragraphs",
        "n_sentences",
        "n_fixated",
        "n_skipped",
        "skipping_rate",
    ]:
        s = _num_summary(k)
        if s is not None:
            exposure_summary[k] = s
    results["exposure"] = exposure_summary

    # Eye movement baselines
    em_summary = {}
    for k in [
        "mean_total_reading_time_ms",
        "mean_gaze_duration_ms",
        "mean_first_fixation_duration_ms",
        "mean_n_fixations",
        "mean_regression_probability",
    ]:
        s = _num_summary(k)
        if s is not None:
            em_summary[k] = s
    results["eye_movement_baselines"] = em_summary

    logger.info(
        f"Participant stats: N={results.get('n_participants', 0)} "
        f"(control={results.get('n_control', 0)}, dyslexic={results.get('n_dyslexic', 0)})"
    )
    return results
