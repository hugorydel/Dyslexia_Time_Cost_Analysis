"""
Shared statistical utilities for hypothesis testing
Includes corrected p-value calculation with +1 correction and effect size measures
"""

import numpy as np


def compute_two_tailed_pvalue_corrected(
    observed_value: float, bootstrap_samples: np.ndarray
) -> float:
    """
    Compute two-tailed p-value with +1 correction: (r+1)/(B+1)

    Parameters
    ----------
    observed_value : float
        The observed statistic (e.g., mean, slope ratio)
    bootstrap_samples : np.ndarray
        Bootstrap distribution of the statistic

    Returns
    -------
    float
        Two-tailed p-value with +1 correction

    Notes
    -----
    Uses the conservative (r+1)/(B+1) correction for bootstrap p-values.
    For two-tailed test, counts extreme values on the opposite side of 0
    (or 1 for ratio statistics) and doubles the one-tailed p-value.
    """
    samples = np.asarray(bootstrap_samples, dtype=float)
    samples = samples[np.isfinite(samples)]

    if samples.size == 0:
        return np.nan

    B = samples.size

    # Count extreme values (opposite direction from observed)
    if observed_value > 0:
        extreme = np.sum(samples <= 0.0)
    else:
        extreme = np.sum(samples >= 0.0)

    # Apply +1 correction and compute two-tailed p-value
    p_one_tailed = (extreme + 1.0) / (B + 1.0)
    p_two_tailed = min(1.0, 2.0 * p_one_tailed)

    return float(p_two_tailed)


def compute_two_tailed_pvalue_corrected_ratio(
    observed_ratio: float, bootstrap_samples: np.ndarray, null_value: float = 1.0
) -> float:
    """
    Compute two-tailed p-value with +1 correction for ratio statistics (e.g., slope ratios)

    Parameters
    ----------
    observed_ratio : float
        The observed ratio statistic
    bootstrap_samples : np.ndarray
        Bootstrap distribution of the ratio
    null_value : float, optional
        The null hypothesis value (default: 1.0 for ratio tests)

    Returns
    -------
    float
        Two-tailed p-value with +1 correction

    Notes
    -----
    Similar to compute_two_tailed_pvalue_corrected but tests against a ratio null value.
    For slope ratios, we typically test H0: SR = 1 (no amplification/reduction).
    """
    samples = np.asarray(bootstrap_samples, dtype=float)
    samples = samples[np.isfinite(samples)]

    if samples.size == 0:
        return np.nan

    B = samples.size

    # Count extreme values (opposite direction from observed relative to null_value)
    if observed_ratio > null_value:
        extreme = np.sum(samples <= null_value)
    else:
        extreme = np.sum(samples >= null_value)

    # Apply +1 correction and compute two-tailed p-value
    p_one_tailed = (extreme + 1.0) / (B + 1.0)
    p_two_tailed = min(1.0, 2.0 * p_one_tailed)

    return float(p_two_tailed)


def cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for two proportions (e.g., skip rates)

    Parameters
    ----------
    p1 : float
        First proportion (0 to 1)
    p2 : float
        Second proportion (0 to 1)

    Returns
    -------
    float
        Cohen's h (signed difference on arcsine-sqrt scale)

    Notes
    -----
    Cohen's h is the appropriate effect size for comparing proportions.
    Typical interpretation: |h| < 0.2 (small), 0.2-0.5 (medium), > 0.5 (large)
    """
    p1 = np.clip(p1, 0.0, 1.0)
    p2 = np.clip(p2, 0.0, 1.0)
    return 2.0 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def hedges_J(n: int) -> float:
    """
    Hedges' J correction factor for small sample bias in Cohen's d

    Parameters
    ----------
    n : int
        Total sample size (n1 + n2)

    Returns
    -------
    float
        Correction factor J (multiply Cohen's d by this to get Hedges' g)

    Notes
    -----
    Hedges' g = J * Cohen's d, where J approximates the exact unbiased estimator.
    For large samples, J ≈ 1. For small samples, J < 1 (reduces the estimate).
    """
    if n is None or n <= 2:
        return 1.0
    return 1.0 - 3.0 / (4.0 * n - 9.0)


def compute_cohens_d_from_data(
    vals_group1: np.ndarray, vals_group2: np.ndarray
) -> float:
    """
    Compute Cohen's d with pooled standard deviation and Hedges' J correction

    Parameters
    ----------
    vals_group1 : np.ndarray
        Values from group 1 (e.g., Q1 observations)
    vals_group2 : np.ndarray
        Values from group 2 (e.g., Q3 observations)

    Returns
    -------
    float
        Hedges' g (bias-corrected Cohen's d using pooled SD)

    Notes
    -----
    This is the descriptive effect size between two groups of observations.
    Uses pooled standard deviation and applies Hedges' J small-sample correction.
    Returns NaN if either group has < 2 observations or if pooled SD = 0.
    """
    v1 = np.asarray(vals_group1, dtype=float)
    v2 = np.asarray(vals_group2, dtype=float)
    v1 = v1[np.isfinite(v1)]
    v2 = v2[np.isfinite(v2)]

    n1, n2 = v1.size, v2.size
    if n1 < 2 or n2 < 2:
        return float("nan")

    m1, m2 = float(np.mean(v1)), float(np.mean(v2))
    s1, s2 = float(np.std(v1, ddof=1)), float(np.std(v2, ddof=1))

    if s1 == 0.0 and s2 == 0.0:
        return float("nan")

    # Pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    if s_pooled == 0.0:
        return float("nan")

    # Cohen's d
    d = (m2 - m1) / s_pooled

    # Apply Hedges' J correction for small samples
    J = hedges_J(n1 + n2)
    g = J * d

    return float(g)
