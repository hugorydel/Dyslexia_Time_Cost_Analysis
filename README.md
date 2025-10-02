# Linguistic Features Setup & Usage Guide

## Overview

This integration adds **three key linguistic variables** from current eye-tracking research to your CopCo dyslexia analysis:

1. **Word Length** - Character count (you already have this)
2. **Word Frequency** - Danish lemma frequencies from korpus.dsl.dk, Zipf-transformed (1-7 scale)
3. **Surprisal** - Contextual predictability from transformer models (optional, computationally expensive)

Based on research showing these are the primary predictors of reading time differences between dyslexic and control readers.

---

## Setup Instructions

### 1. File Placement

Place the `linguistic_features.py` file in your `utils/` directory:

```
your_project/
├── main.py
├── config.py
├── utils/
│   ├── linguistic_features.py  ← NEW FILE HERE
│   ├── data_utils.py
│   ├── stats_utils.py
│   └── lemma/
│       └── lemma-10k-2017-in.txt  ← FREQUENCY DATA HERE
```

### 2. Frequency Data

You've already placed the lemma file at `utils/lemma/lemma-10k-2017-in.txt` ✓

This file contains the top 10,000 Danish word frequencies from the CopCo processing pipeline, sourced from korpus.dsl.dk (the validated Danish corpus).

### 3. Dependencies

**Required** (you likely already have these):

```bash
pip install pandas numpy
```

**Optional** (for surprisal computation):

```bash
pip install transformers torch
```

⚠️ **Note**: Surprisal computation is **slow** (hours for full dataset) and requires significant memory. Start without it to validate the pipeline.

---

## Usage

### Quick Start (Without Surprisal)

Run exploratory analysis with linguistic features:

```bash
python main.py --explore
```

This will:

- Load your CopCo data
- Compute word frequency (Zipf scale) for all words
- Analyze frequency and length effects on reading times
- Generate visualizations showing:
  - Frequency distributions
  - Length effects by group
  - Frequency effects on skipping
  - Group comparisons (dyslexic vs control)

### With Surprisal (Slow!)

To include surprisal computation:

```bash
python main.py --explore --surprisal
```

⚠️ **Warning**: This will take **several hours** for the full CopCo dataset. Consider testing on a subset first.

---

## Output Files

After running, check `results/` directory:

### 1. `exploratory_summary.json`

Now includes a new `linguistic_features` section:

```json
{
	"linguistic_features": {
		"linguistic_features_summary": {
			"word_length": {
				"mean": 4.69,
				"std": 3.17,
				"min": 1,
				"max": 29
			},
			"word_frequency_zipf": {
				"mean": 4.23,
				"std": 0.89,
				"min": 0.01,
				"max": 6.45
			}
		},
		"frequency_effects": {
			"total_reading_time": {
				"Low (≤3)": { "mean": 456.2, "std": 412.1 },
				"High (>4)": { "mean": 298.5, "std": 285.3 }
			}
		},
		"group_differences": {
			"control": { "word_frequency_zipf_correlation": -0.34 },
			"dyslexic": { "word_frequency_zipf_correlation": -0.42 }
		}
	}
}
```

### 2. `linguistic_features_plots.png`

9-panel visualization showing:

- Frequency & length distributions
- Effects on reading time
- Skipping patterns
- Group comparisons

---

## Understanding the Features

### Word Frequency (Zipf Scale)

**Scale**: 1 (very rare) to 7 (very frequent)

- **≤3**: Low frequency (~1 per million words or less)
- **3-4**: Medium-low frequency
- **4-5**: Medium-high frequency
- **>5**: High frequency (top ~1000 words)

**Expected effects**:

- Lower frequency → longer fixations
- Lower frequency → less skipping
- **Dyslexics show amplified effects** (steeper slope)

### Surprisal

**Scale**: 0-25 bits (typically 2-15)

- Low surprisal: Predictable words ("the cat sat on the...")
- High surprisal: Unexpected words ("the cat sat on the submarine")

**Expected effects**:

- Higher surprisal → longer fixations
- Higher surprisal → more regressions
- **Dyslexics may show larger costs** for unpredictable words (working memory hypothesis)

---

## Validation Checklist

After running, verify:

✓ **Coverage**: Check `missing_pct` in JSON output

- Word frequency should have <10% missing
- Higher missing = many rare/compound words not in top 10K list

✓ **Range checks**:

- Zipf frequency: 0-8 (typically 1-7)
- Word length: 1-29 characters
- Surprisal: 0-25 bits (typically 2-15)

✓ **Expected patterns**:

- Frequency negatively correlates with reading time
- Length positively correlates with reading time
- Dyslexic group shows steeper effects

---

## Troubleshooting

### "FileNotFoundError: Lemma frequency file not found"

Check that `utils/lemma/lemma-10k-2017-in.txt` exists with correct path.

### "ImportError: transformers library required"

You tried to compute surprisal without installing transformers:

```bash
pip install transformers torch
```

Or run without surprisal: `python main.py --explore` (no `--surprisal` flag)

### High percentage of missing frequencies

This is normal for:

- Proper nouns (not in frequency lists)
- Rare compound words
- Technical terms

Words outside top 10K get assigned minimum frequency (Zipf ~0-1).

### Surprisal computation is very slow

Expected behavior. For 335,166 words:

- **GPU**: ~2-4 hours
- **CPU**: ~6-12 hours

Consider:

- Testing on subset first: `data.sample(10000)`
- Using faster model: `"bert-base-multilingual-cased"`
- Running overnight
- Skipping surprisal initially to validate other features

---

## Next Steps

Once linguistic features are computing successfully:

1. **Validate against CopCo benchmarks**: Compare your frequency effects to published CopCo statistics
2. **Test group differences**: Confirm dyslexic readers show amplified frequency/length effects
3. **Statistical modeling**: Use these features in mixed-effects models predicting reading times
4. **Decomposition analysis**: Quantify how much of the dyslexia-control gap each feature explains

---

## Technical Details

### Danish-Specific Considerations

The code uses **Danish BERT** (`Maltehb/danish-bert-botxo`) for surprisal by default, trained on Danish Gigaword corpus. This outperforms multilingual BERT for Danish text.

**Opaque orthography effects**: Danish has 2.5x slower reading acquisition than transparent orthographies. Expect:

- Larger frequency effects than in Finnish/Spanish studies
- Speed deficits more diagnostic than accuracy deficits
- Pronounced word length effects due to phonological processing costs

### Performance Optimization

For surprisal computation:

- Processes sentence-by-sentence (preserves context)
- Uses GPU automatically if available
- Batch processing with progress bars
- Word-token alignment for multi-token words

---

## Citation

If using this approach in publications, cite:

**CopCo corpus**:
Hollenstein, N., Barrett, M., & Björnsdóttir, K. B. (2022). The Copenhagen Corpus of Eye Tracking Recordings from Natural Reading of Danish Texts. LREC 2022.

**Frequency source**:
korpus.dsl.dk (Danish Society for Language and Literature)

**Zipf transformation**:
van Heuven et al. (2014). SUBTLEX-UK methodology.

**Transformer surprisal**:
Shain et al. (2024). Language models outperform cloze predictability in cognitive models of reading. PLOS Computational Biology.
