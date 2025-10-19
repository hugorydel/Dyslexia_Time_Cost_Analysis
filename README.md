# Dyslexia GAM Analysis

## 📋 Overview

This codebase implements the complete Analysis Plan for studying dyslexic reading patterns using Generalized Additive Models (GAMs).

### Key Features

✅ **Raw zipf frequency** (natural correlation with length)  
✅ **Pooled binning** infrastructure  
✅ **Conditional zipf evaluation** (within length bins)  
✅ **Tensor product interactions** `te(length, zipf)` in GAMs  
✅ **Subject-clustered bootstrap** (resamples subjects, not observations)  
✅ **Comprehensive hypothesis testing** (H1, H2, H3)  
✅ **Automated validation** checks  
✅ **Publication figures** and tables

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone [your-repo]
cd dyslexia_analysis

# Create environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm pygam
```

### Run data pre-processing script

```bash
python preprocess_data.py
```

### Run Analysis

```bash
# Full analysis (~30 minutes with 1000 bootstrap iterations)
python run_complete_analysis.py --data-path input_data/preprocessed_data.csv

# Quick test (~5 minutes with 100 bootstrap iterations)
python run_complete_analysis.py --data-path input_data/preprocessed_data.csv --quick
```

### View Results

```bash
# Summary
cat results_final/RESULTS_SUMMARY.md

# Figures
open results_final/figures/

# Tables
open results_final/tables/
```

---

## 📁 Repository Structure

```
dyslexia_analysis/
├── hypothesis_testing_utils/          # Core analysis modules
│   ├── data_preparation_v2.py         # ✨ NEW: Raw zipf + pooled bins
│   ├── gam_models_v2.py               # ✨ NEW: Tensor products + CV
│   ├── h1_feature_effects_v2.py       # ✨ NEW: Conditional zipf
│   ├── h2_amplification_v2.py         # ✨ NEW: Comprehensive bootstrap
│   ├── h3_gap_decomposition.py        # (mostly reused)
│   ├── ert_predictor.py               # (unchanged)
│   ├── visualization_utils.py         # ✨ NEW: Figure generation
│   ├── table_generator.py             # ✨ NEW: Table generation
│   └── validate_migration.py          # ✨ NEW: Validation checks
│
├── input_data/
│   └── preprocessed_data.csv          # Your eye-tracking data
│
├── run_complete_analysis.py           # ✨ NEW: Main pipeline
├── GETTING_STARTED.md                 # Step-by-step guide
├── REVISION_SUMMARY.md                # What changed and why
├── QUICK_REFERENCE.md                 # Old vs new code
└── README.md                          # This file
```

---

## 🔄 What Changed?

### Critical Fixes

| Issue                | OLD ❌                 | NEW ✓                             |
| -------------------- | ---------------------- | --------------------------------- |
| **Zipf handling**    | Orthogonalized (r≈0)   | Raw (r≈-0.80)                     |
| **Zipf evaluation**  | Unconditional          | Conditional within length bins    |
| **GAM interactions** | None                   | `te(length, zipf)` tensor product |
| **Bootstrap**        | Resampled observations | Resamples subjects                |
| **Bootstrap scope**  | Separate per metric    | Comprehensive (all metrics)       |
| **Binning**          | Per-group or none      | Pooled once, applied to both      |

### New Features

- ✨ Pathway decomposition (skip, duration, ERT) for ALL features
- ✨ Cohen's h effect size for skip pathway
- ✨ Automated validation with detailed diagnostics
- ✨ Publication-ready figures (3 main + supplementary)
- ✨ Formatted tables (CSV, TXT, LaTeX)
- ✨ Comprehensive markdown summary

---

## 📊 Analysis Pipeline

### Phase 1: Data Preparation

```python
prepared_data, metadata = prepare_data_pipeline(data)
```

**Outputs:**

- Raw zipf values (no orthogonalization)
- Pooled length bins (5 bins)
- Feature quartiles
- Orientation checks

### Phase 2: Model Fitting

```python
skip_meta, duration_meta, gam_models = fit_gam_models(data)
```

**Models:**

- Skip: `LogisticGAM` with `te(length, zipf)`
- Duration: `LinearGAM` on log(TRT) with `te(length, zipf)`
- Grid search over n_splines [8, 10, 12]
- GroupKFold CV by subject

### Phase 3: Hypothesis Testing

**H1: Feature Effects**

- Standard AMIE for length, surprisal
- Conditional AMIE for zipf (within length bins)
- Pathway decomposition (skip, duration, ERT)

**H2: Amplification**

- Slope ratios for all features × all pathways
- Subject-clustered bootstrap (1000 iterations)
- Conditional SR for zipf

**H3: Gap Decomposition**

- Shapley decomposition (skip vs duration)
- Equal-ease counterfactual
- Per-feature equalization

### Phase 4: Validation

Automated checks:

- ✅ Raw zipf correlation (≈ -0.80)
- ✅ Pooled bins present
- ✅ Tensor products in GAMs
- ✅ Conditional zipf evaluation
- ✅ Bootstrap structure correct

### Phase 5-6: Outputs

**Figures:**

1. Overall effects overview
2. Zipf pathway decomposition
3. Gap decomposition waterfall

**Tables:**

1. Feature effects & amplification
2. Gap decomposition & counterfactuals
3. Model performance

---

## 📈 Expected Results

### H1: Feature Effects ✓

| Feature   | Direction | Control AMIE  | Dyslexic AMIE |
| --------- | --------- | ------------- | ------------- |
| Length    | +         | +100-120 ms   | +120-150 ms   |
| Zipf\*    | −         | -15 to -20 ms | -5 to -10 ms  |
| Surprisal | +         | +15-20 ms     | +30-40 ms     |

\*Zipf evaluated conditionally within length bins

### H2: Amplification ✓

| Feature   | SR(skip) | SR(duration) | SR(ERT) |
| --------- | -------- | ------------ | ------- |
| Length    | ~0.80    | ~1.45        | ~1.20   |
| Zipf\*    | ~0.05†   | ~4.60        | ~0.40   |
| Surprisal | ~1.10    | ~2.35        | ~2.20   |

\*Zipf shows complex pathway effects  
†Unstable: control Δp ≈ 0

### H3: Gap Decomposition ✓

- **Total gap:** ~90 ms
- **Skip contribution:** ~30% (28-30 ms)
- **Duration contribution:** ~70% (65-70 ms)
- **Equal-ease reduction:** ~55-60%

---

## 🔬 Methodology Highlights

### Why Conditional Evaluation for Zipf?

Zipf frequency correlates strongly with length (r ≈ -0.80):

- Longer words tend to be rarer
- Shorter words tend to be more frequent

**Problem:** Unconditioned Q1→Q3 confounds length and frequency effects

**Solution:** Evaluate zipf **within each length bin**:

1. Hold length approximately constant
2. Vary only frequency
3. Weight by pooled distribution
4. This isolates "frequency beyond length" effect

### Why Tensor Products?

`te(length, zipf)` captures:

- Non-linear length × frequency interactions
- Different frequency effects at different lengths
- Resolves TRT vs ERT paradox

### Why Subject-Clustered Bootstrap?

Observations within subjects are correlated.

**Wrong approach:** Resample observations

- Breaks within-subject correlation
- Underestimates standard errors

**Correct approach:** Resample entire subjects

- Preserves correlation structure
- Valid confidence intervals

---

## 🧪 Testing & Validation

### Validation Script

```bash
python validate_migration.py
```

Checks:

- Raw zipf (not orthogonalized)
- Pooled binning infrastructure
- Tensor products in models
- Conditional zipf evaluation
- Bootstrap resamples subjects
- All expected metrics present

### Unit Tests (Future)

```bash
# Run tests (when available)
pytest tests/
```

---

## 📖 Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Step-by-step tutorial
- **[REVISION_SUMMARY.md](REVISION_SUMMARY.md)** - Detailed change log
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Old vs new code comparison
- **Analysis Plan V1** - Original specification document

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1:** "High zipf correlation warning"  
**Solution:** This is CORRECT for raw zipf. Ignore warning.

**Issue 2:** "Unstable SR for zipf skip pathway"  
**Solution:** EXPECTED. Control shows minimal skip effect. Focus on duration.

**Issue 3:** "Bootstrap taking too long"  
**Solution:** Use `--quick` flag or reduce `--n-bootstrap`

See [GETTING_STARTED.md](GETTING_STARTED.md) for more troubleshooting.

---

## 🤝 Contributing

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all functions
- Keep functions focused and small

### Testing

- Add unit tests for new features
- Run validation script before committing
- Check that bootstrap is properly clustered

### Documentation

- Update relevant .md files
- Add examples for new features
- Keep REVISION_SUMMARY.md current

---

## 📚 Citation

If using this code, please cite:

```bibtex
@software{dyslexia_gam_v2,
  title={Dyslexia GAM Analysis Pipeline V2},
  author={[Your Name]},
  year={2025},
  note={Revised implementation with conditional zipf evaluation,
        tensor product interactions, and subject-clustered bootstrap},
  url={[your-repo-url]}
}
```

---

## 📄 License

[Your License Here]

---

## 🙏 Acknowledgments

- Analysis plan developed from reading research literature
- PyGAM library for GAM implementation
- Scikit-learn for cross-validation utilities

---

## 📞 Contact & Support

- **Issues:** [GitHub Issues](your-repo/issues)
- **Email:** [your-email]
- **Documentation:** See docs/ folder

---

## ✅ Checklist for New Users

- [ ] Read GETTING_STARTED.md
- [ ] Install dependencies
- [ ] Prepare data in correct format
- [ ] Run analysis with `--quick` flag first
- [ ] Review validation output
- [ ] Check RESULTS_SUMMARY.md
- [ ] Review figures and tables
- [ ] Run full analysis (1000 bootstrap)
- [ ] Write up methods section

---

## 🎯 Project Status

**Current Version:** 2.0.0 (Revised)  
**Status:** ✅ Production Ready  
**Last Updated:** 2025-01-XX

### Version History

- **v2.0.0** (2025-01) - Complete rewrite aligned with Analysis Plan V1
  - Raw zipf + conditional evaluation
  - Tensor products + comprehensive bootstrap
  - Automated validation + visualization
- **v1.0.0** (2024-XX) - Initial implementation
  - Orthogonalized zipf (deprecated)
  - Simple smooths without interactions
  - Separate bootstrap per metric

---

**Ready to analyze dyslexic reading patterns?** Start with [GETTING_STARTED.md](GETTING_STARTED.md)! 🚀
