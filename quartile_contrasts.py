#!/usr/bin/env python3
"""
quartile_contrasts.py
---------------------
Tiny independent module that:
- Loads results/processed_data_full.csv
- Builds an Expected Reading Time (ERT) = 0 if skipped else total_reading_time
- For each feature (default: word_length, word_frequency_zipf, surprisal):
    * Computes pooled Q1 (bottom 25%) and Q4 (top 25%) cutpoints
    * Keeps only Q1 and Q4 observations
    * Fits a simple 2x2 OLS: ERT ~ Bin(Q1=0,Q4=1) + Group(Non=0,Dys=1) + Interaction
      with cluster-robust (by subject) SEs
    * Prints and saves p-values for Bin (main), Group (main), and Interaction
Outputs:
- Console table
- CSV at results/quartile_contrasts_stats.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def _to_bool(series: pd.Series) -> pd.Series:
    """Coerce various True/False encodings to boolean."""
    s = series.copy()
    if s.dtype == bool:
        return s
    s = s.astype(str).str.strip().str.lower()
    return s.map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}).fillna(False)


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Coerce columns we rely on
    if "skipped" not in df.columns:
        raise ValueError("Column 'skipped' not found in CSV.")
    if "total_reading_time" not in df.columns:
        raise ValueError("Column 'total_reading_time' not found in CSV.")
    if "subject_id" not in df.columns:
        raise ValueError("Column 'subject_id' not found in CSV.")
    # Prefer 'dyslexic'; fallback to 'dyslexia'
    group_col = "dyslexic" if "dyslexic" in df.columns else ("dyslexia" if "dyslexia" in df.columns else None)
    if group_col is None:
        raise ValueError("Neither 'dyslexic' nor 'dyslexia' columns found in CSV.")
    df = df.copy()
    df["Group"] = _to_bool(df[group_col]).astype(int)  # 1=dyslexic, 0=control
    df["skipped_bool"] = _to_bool(df["skipped"])
    # ERT: 0 if skipped, else TRT (fill NA with 0 to be safe)
    df["ERT"] = np.where(df["skipped_bool"], 0.0, df["total_reading_time"].astype(float).fillna(0.0))
    # Ensure subject_id is a string for clustering
    df["subject_id"] = df["subject_id"].astype(str)
    return df


def pooled_q1_q4(series: pd.Series) -> tuple[float, float]:
    """Return pooled Q1 and Q3 cutpoints (i.e., 25th and 75th percentiles)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.quantile(0.25)), float(s.quantile(0.75))


def fit_ols_bin_group(sub: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    OLS with cluster-robust SEs (by subject):
        ERT ~ Bin + Group + Bin:Group
    where Bin is 0 (Q1) / 1 (Q4), Group is 0 (control) / 1 (dyslexic).
    """
    # Guard: need at least two bins and two groups represented
    if sub["Bin"].nunique() < 2 or sub["Group"].nunique() < 2:
        raise ValueError("Subset lacks sufficient variation in Bin or Group for OLS.")
    model = smf.ols("ERT ~ Bin + Group + Bin:Group", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["subject_id"]}, use_t=True
    )
    return model


def analyze(csv_path: Path, features: list[str]) -> pd.DataFrame:
    df = load_data(csv_path)

    rows = []
    for feat in features:
        if feat not in df.columns:
            print(f"[WARN] Feature '{feat}' not found; skipping.", file=sys.stderr)
            continue

        q1, q3 = pooled_q1_q4(df[feat])
        # Keep only bottom 25% (<= q1) and top 25% (>= q3)
        sub = df.loc[(df[feat] <= q1) | (df[feat] >= q3)].copy()
        sub["Bin"] = (sub[feat] >= q3).astype(int)  # 0=Q1, 1=Q4

        try:
            model = fit_ols_bin_group(sub)
            # Extract p-values for terms of interest
            p_bin = model.pvalues.get("Bin", np.nan)
            p_group = model.pvalues.get("Group", np.nan)
            # Interaction term label may vary; use explicit key
            p_inter = model.pvalues.get("Bin:Group", np.nan)

            # Also collect simple means (useful context)
            means = sub.groupby(["Group", "Bin"])["ERT"].mean().rename("mean_ERT").reset_index()
            # Wide layout for quick glance
            mean_q1_ctrl = float(means.query("Group==0 and Bin==0")["mean_ERT"].iloc[0]) if not means.query("Group==0 and Bin==0").empty else np.nan
            mean_q4_ctrl = float(means.query("Group==0 and Bin==1")["mean_ERT"].iloc[0]) if not means.query("Group==0 and Bin==1").empty else np.nan
            mean_q1_dys  = float(means.query("Group==1 and Bin==0")["mean_ERT"].iloc[0]) if not means.query("Group==1 and Bin==0").empty else np.nan
            mean_q4_dys  = float(means.query("Group==1 and Bin==1")["mean_ERT"].iloc[0]) if not means.query("Group==1 and Bin==1").empty else np.nan

            rows.append({
                "feature": feat,
                "q1_cut": q1,
                "q4_cut": q3,
                "n_q1": int((sub["Bin"]==0).sum()),
                "n_q4": int((sub["Bin"]==1).sum()),
                "p_main_bin(Q4_vs_Q1)": p_bin,
                "p_main_group(Dys_vs_Ctrl)": p_group,
                "p_interaction(Bin×Group)": p_inter,
                "mean_q1_ctrl_ms": mean_q1_ctrl,
                "mean_q4_ctrl_ms": mean_q4_ctrl,
                "mean_q1_dys_ms":  mean_q1_dys,
                "mean_q4_dys_ms":  mean_q4_dys,
            })
        except Exception as e:
            rows.append({
                "feature": feat,
                "q1_cut": q1,
                "q4_cut": q3,
                "error": str(e)
            })

    out = pd.DataFrame(rows)
    out_path = csv_path.parent / "quartile_contrasts_stats.csv"
    out.to_csv(out_path, index=False)
    print("\nQuartile contrasts (cluster-robust by subject)\n")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved: {out_path}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Quartile contrasts (Q1 vs Q4) with group interaction")
    parser.add_argument("--csv", type=str, default=str(Path("results") / "processed_data_full.csv"),
                        help="Path to processed_data_full.csv")
    parser.add_argument("--features", type=str, nargs="*", default=["word_length", "word_frequency_zipf", "surprisal"],
                        help="Feature columns to analyze")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    features = args.features
    analyze(csv_path, features)


if __name__ == "__main__":
    main()
