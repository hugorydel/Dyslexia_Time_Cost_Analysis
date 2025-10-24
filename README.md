Here’s a clean, step-by-step **README.md** you can drop in the repo.

---

# Dyslexia Time Cost Analysis

End-to-end pipeline to preprocess CopCo eye-tracking data, compute linguistic features, fit GAM models for skipping and duration, and test three hypotheses (H1–H3) with APA-7-styled figures.

---

## 0) Quick start

```bash
# 1) Create & activate an environment (example with venv)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Configure paths (see next section), then:
python preprocess_data.py
python run_complete_analysis.py --data-path preprocessing_output/preprocessed_data.csv --results-dir results_final
```

---

## 1) Configure data paths

Set the path to the CopCo dataset in **`utils/config.py`** (`COPCO_PATH`), which should contain `ExtractedFeatures/` and (optionally) `DatasetStatistics/`. The preprocessing entrypoint validates this setting at startup, so you’ll get a clear warning/error if it’s missing or incorrect.

---

## 2) Prepare frequency data (for linguistic features)

The pipeline expects a Danish frequency list at:

```
preprocessing_output/danish_frequencies/danish_leipzig_for_analysis.txt
```

That file is read automatically by the linguistic features component (`DanishLinguisticFeatures`), which computes Zipf frequencies (and later surprisal) for each word. If the file is missing, you’ll get a `FileNotFoundError` pointing to that path. You can generate or place the file there before running preprocessing.

> Where it’s used: the frequency dictionary is loaded during feature computation; surprisal values are cached to `preprocessing_output/word_surprisal/`.

---

## 3) Preprocess the CopCo data

Run:

```bash
python preprocess_data.py
```

What it does:

- Loads **`ExtractedFeatures/`** (and merges `DatasetStatistics/` if available), cleans/derives measures, identifies dyslexic participants, and computes skipping-related measures.
- Computes linguistic features (Zipf frequency, length, surprisal) and caches surprisal.
- Saves **`preprocessing_output/preprocessed_data.csv`** plus a data dictionary **`preprocessing_output/preprocessed_data.txt`**.
- Writes summary JSONs and PNGs to **`preprocessing_output/preprocessing_summary/`**:

  - `exploratory_summary.json`, `participant_statistics.json`
  - `exploratory_plots.png`, `group_summary_plots.png` (descriptive visuals)

- Logs to `dyslexia_analysis.log`.

> Tip: The preprocessing step saves the CSV at `preprocessing_output/preprocessed_data.csv`. Use that exact path for the hypothesis pipeline unless you override it.

---

## 4) Run the hypothesis pipeline (H1–H3)

Run:

```bash
python run_complete_analysis.py \
  --data-path preprocessing_output/preprocessed_data.csv \
  --results-dir results_final
```

Key CLI flags:

- `--data-path` : path to the CSV from step 3 (defaults to a preprocessed CSV; passing it explicitly is safest).
- `--results-dir` : output directory for everything (default `results_final`).
- `--no-cache` : force recomputation; otherwise caching is used by default.
- `--quick` : fast mode (100 bootstraps) to sanity-check the pipeline.

What happens:

1. **Data prep for modeling** (binning, quartiles, pooled weights) → `data_metadata.json`.
2. **Zipf–length diagnostic** written to the results directory (under diagnostics).
3. **Model fitting with caching**: GAMs for skipping (LogisticGAM) and log-duration (LinearGAM). Caches to `cache_full/` (or `cache_quick/` in quick mode).

   - Caching is handled via `load_or_recompute(...)`.

4. **Hypotheses** (saves JSONs in the results root):

   - H1 feature effects → `h1_results.json`
   - H2 amplification (slope ratios & AMIE) → `h2_results.json`
   - H3 gap decomposition (G₀, counterfactual gap, shrink, Shapley) → `h3_results.json`

5. **Figures** in `results_dir/figures/` (APA-7 style):

   - `figure_1_overall_effects.png`, `figure_2_gam_smooths.png`, `figure_3_gap_decomposition.png` (+ machine-readable JSON alongside each).
   - Model stats → `model_statistics.json`.

6. **Final summary**: `final_report.json` and a detailed `hypothesis_testing_output.log`. The pipeline prints a completion recap with where to find everything.

---

## 5) Outputs at a glance

Inside `--results-dir` you’ll find:

- `figures/`: publication-ready PNGs + the exact data used to render them as `.json`.
- `h1_results.json`, `h2_results.json`, `h3_results.json`, `data_metadata.json`, `model_metadata.json`, `final_report.json`.
- `hypothesis_testing_output.log` and a printed checklist of “what’s next”.

---

## 6) Reproducibility & performance notes

- **Bootstraps**: default `n_bootstrap=2000` (use `--quick` to test with 100).
- **Caching**: models & results are cached under `results_dir/cache_full/` (or `cache_quick/`). You can force recompute with `--no-cache` or manually clear the cache directory. The caching utilities handle atomic writes and safe loads.
- **Seeds**: model selection and bootstraps set internal RNGs in various places (e.g., hyperparameter pre-selection uses a fixed RNG for subsampling). Exact seeds are logged where relevant.

---

## 7) Troubleshooting

- **`FileNotFoundError: CopCo data not found …`**
  Ensure `COPCO_PATH` in `utils/config.py` points to the CopCo root containing `ExtractedFeatures/`.

- **`Frequency file not found … danish_leipzig_for_analysis.txt`**
  Place the file at `preprocessing_output/danish_frequencies/` before preprocessing (or generate it with your helper script), then re-run `preprocess_data.py`.

- **“Cache load failed” messages**
  The pipeline will recompute automatically and attempt to overwrite the cache atomically. You can also delete `results_dir/cache_full/*.pkl`.

- **Figure styling**
  All figures are produced via `hypothesis_testing_utils/visualization_utils.py`. You can regenerate/modify styles by editing that file and re-running the pipeline (JSON data are saved next to each PNG).

---

## 8) How to cite / License

See `LICENSE` for licensing; if you use the code, please cite the repository and (if applicable) your accompanying manuscript.

---

### Appendix: What each stage computes

- **Preprocessing**: cleaning, dyslexia grouping, derived measures (skipping, etc.), linguistic features (Zipf, length, surprisal), exploratory summaries & plots.
- **Modeling**: group-specific GAMs for skipping (LogisticGAM) and log-duration (LinearGAM) with cross-validated hyperparameters.
- **H1**: feature effects with uncertainty; **H2**: amplification (slope ratios/AMIE); **H3**: gap decomposition (baseline gap G₀, equal-ease counterfactual, shrink % + Shapley contributions).

---
