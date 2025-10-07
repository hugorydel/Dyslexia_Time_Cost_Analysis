#!/usr/bin/env python3
"""
Download and combine Leipzig Corpora Danish word frequency lists
Enhanced with filtering and log-transformation for psycholinguistic research
"""

import gzip
import io
import math
import re
import tarfile
from pathlib import Path

import pandas as pd
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

# Leipzig Corpora download URLs
LEIPZIG_BASE = "https://downloads.wortschatz-leipzig.de/corpora/"

DANISH_CORPORA = {
    "dan_mixed_2014": "dan_mixed_2014_1M.tar.gz",
    "dan_news_2007": "dan_news_2007_1M.tar.gz",
    "dan_news_2021": "dan_news_2021_1M.tar.gz",
    "dan_news_2020": "dan_news_2020_1M.tar.gz",
    "dan_newscrawl_2017": "dan_newscrawl_2017_1M.tar.gz",
    "dan_newscrawl_2019": "dan_newscrawl_2019_1M.tar.gz",
    "dan_newscrawl_2023": "dan_newscrawl_2023_1M.tar.gz",
    "dan_web_2014": "dan-dk_web_2014_1M.tar.gz",
    "dan_web_2015": "dan-dk_web_2015_1M.tar.gz",
    "dan_web_2019": "dan-dk_web_2019_1M.tar.gz",
    "dan_web_public_2019": "dan-dk_web-public_2019_1M.tar.gz",
    "dan_wiki_2014": "dan_wikipedia_2014_1M.tar.gz",
    "dan_wiki_2016": "dan_wikipedia_2016_1M.tar.gz",
    "dan_wiki_2021": "dan_wikipedia_2021_1M.tar.gz",
}

OUTPUT_DIR = Path("danish_frequencies")
OUTPUT_DIR.mkdir(exist_ok=True)

# Processing parameters
TOP_N_WORDS = 1500000  # Keep top 1.5M words
MIN_FREQUENCY = 2  # Minimum frequency threshold
REMOVE_PUNCT_ONLY = True  # Remove entries that are only punctuation
ADD_LOG_TRANSFORM = True  # Add log-transformed frequency columns

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def is_punctuation_only(word):
    """Check if a word contains only punctuation/special characters"""
    return bool(re.match(r"^[^\w\s]+$", word, re.UNICODE))


def download_file(url, filename):
    """Download a file with progress indication"""
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        filepath = OUTPUT_DIR / filename
        with open(filepath, "wb") as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    downloaded += len(chunk)
                    f.write(chunk)
                    progress = (downloaded / total_size) * 100
                    print(f"  Progress: {progress:.1f}%", end="\r")
        print(f"  ✓ Downloaded: {filepath}")
        return filepath
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Failed to download: {e}")
        return None


def extract_tar_gz(filepath):
    """Extract tar.gz archive and find words file"""
    print(f"Extracting {filepath.name}...")
    try:
        with tarfile.open(filepath, "r:gz") as tar:
            # Find the words file (usually ends with -words.txt)
            words_file = None
            for member in tar.getmembers():
                if "-words.txt" in member.name:
                    words_file = member
                    break

            if words_file:
                extracted = tar.extractfile(words_file)
                content = extracted.read().decode("utf-8")

                # Save extracted words file
                output_path = OUTPUT_DIR / f"{filepath.stem}_words.txt"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"  ✓ Extracted: {output_path}")
                return output_path
            else:
                print(f"  ✗ No words file found in archive")
                return None
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        return None


def load_words_file(filepath):
    """Load words file into dataframe"""
    print(f"Loading {filepath.name}...")
    try:
        df = pd.read_csv(
            filepath,
            sep="\t",
            header=None,
            names=["id", "word", "freq"],
            quoting=3,
            on_bad_lines="skip",
        )
        print(f"  ✓ Loaded {len(df):,} words")
        return df[["word", "freq"]]
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        return None


def process_frequency_data(df):
    """Filter and transform frequency data"""
    print("\nProcessing frequency data...")

    original_count = len(df)

    # STANDARDIZE: Convert all words to lowercase and merge duplicates
    print("  Standardizing words to lowercase...")
    before_standardize = len(df)
    df["word"] = df["word"].str.lower()
    df = df.groupby("word", as_index=False)["freq"].sum()
    df = df.sort_values("freq", ascending=False).reset_index(drop=True)
    after_standardize = len(df)
    duplicates_merged = before_standardize - after_standardize
    print(f"  ✓ Standardized and merged {duplicates_merged:,} case-variant duplicates")
    print(f"    {before_standardize:,} → {after_standardize:,} unique lowercase words")

    # Filter by minimum frequency
    if MIN_FREQUENCY > 1:
        df = df[df["freq"] >= MIN_FREQUENCY]
        print(f"  ✓ Filtered by min frequency ({MIN_FREQUENCY}): {len(df):,} words")

    # Remove punctuation-only entries
    if REMOVE_PUNCT_ONLY:
        before = len(df)
        df = df[~df["word"].apply(is_punctuation_only)]
        removed = before - len(df)
        print(
            f"  ✓ Removed punctuation-only entries: {removed:,} removed, {len(df):,} remaining"
        )

    # Keep top N words
    if TOP_N_WORDS and len(df) > TOP_N_WORDS:
        df = df.nlargest(TOP_N_WORDS, "freq")
        print(f"  ✓ Kept top {TOP_N_WORDS:,} words")

    # Recalculate rank and proportion after filtering
    df["rank"] = range(1, len(df) + 1)
    df["proportion"] = df["freq"] / df["freq"].sum()

    # Add log transformations
    if ADD_LOG_TRANSFORM:
        # Add small constant to avoid log(0)
        df["log_freq"] = df["freq"].apply(lambda x: math.log(x) if x > 0 else 0)
        df["log_proportion"] = df["proportion"].apply(
            lambda x: math.log(x) if x > 0 else math.log(1e-10)
        )
        print(f"  ✓ Added log-transformed columns")

    print(
        f"  ✓ Final dataset: {len(df):,} words ({(len(df)/original_count)*100:.1f}% of original)"
    )

    return df


def main():
    print("=" * 70)
    print("LEIPZIG CORPORA DANISH FREQUENCY LISTS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Top N words: {TOP_N_WORDS:,}")
    print(f"  Min frequency: {MIN_FREQUENCY}")
    print(f"  Remove punctuation-only: {REMOVE_PUNCT_ONLY}")
    print(f"  Add log transforms: {ADD_LOG_TRANSFORM}")
    print("=" * 70 + "\n")

    # Check for existing files first
    words_files = []
    missing_corpora = {}

    for corpus_name, filename in DANISH_CORPORA.items():
        extracted_file = OUTPUT_DIR / f"{filename.replace('.gz', '')}_words.txt"
        archive_file = OUTPUT_DIR / filename

        if extracted_file.exists():
            print(f"✓ Found existing: {extracted_file.name}")
            words_files.append(extracted_file)
        elif archive_file.exists():
            print(f"Found archive: {filename}, extracting...")
            words_file = extract_tar_gz(archive_file)
            if words_file:
                words_files.append(words_file)
        else:
            missing_corpora[corpus_name] = filename

    # Download missing files
    if missing_corpora:
        print(f"\nNeed to download {len(missing_corpora)} corpus/corpora...")
        print("Note: If downloads fail, you may need to download manually from:")
        print("https://wortschatz.uni-leipzig.de/en/download/Danish\n")

        for corpus_name, filename in missing_corpora.items():
            url = LEIPZIG_BASE + filename
            archive_path = download_file(url, filename)

            if archive_path:
                words_file = extract_tar_gz(archive_path)
                if words_file:
                    words_files.append(words_file)
                # Clean up archive
                archive_path.unlink()

    # Check if we got any files
    if not words_files:
        print("\n" + "=" * 70)
        print("NO DATA FILES FOUND")
        print("=" * 70)
        print("\nPlease download manually:")
        print("1. Visit: https://wortschatz.uni-leipzig.de/en/download/Danish")
        print("2. Download these files:")
        for name, file in DANISH_CORPORA.items():
            print(f"   - {file}")
        print(f"3. Extract them to: {OUTPUT_DIR.absolute()}")
        print("4. Run this script again")
        return

    # Load and combine
    print("\n" + "=" * 70)
    print("COMBINING FREQUENCY LISTS")
    print("=" * 70)

    dfs = []
    for words_file in words_files:
        df = load_words_file(words_file)
        if df is not None:
            dfs.append(df)

    if not dfs:
        print("No valid word files loaded. Exiting.")
        return

    # Combine and aggregate (before standardization)
    print("\nCombining frequencies from all corpora...")
    freq = pd.concat(dfs).groupby("word", as_index=False)["freq"].sum()
    freq = freq.sort_values("freq", ascending=False).reset_index(drop=True)

    print(f"✓ Combined: {len(freq):,} unique words (mixed case)")
    print(f"  Total frequency: {freq['freq'].sum():,}")

    # Process the data (includes lowercase standardization and merging)
    freq = process_frequency_data(freq)

    # Reorder columns
    if ADD_LOG_TRANSFORM:
        freq = freq[
            ["rank", "word", "freq", "log_freq", "proportion", "log_proportion"]
        ]
    else:
        freq = freq[["rank", "word", "freq", "proportion"]]

    # Save full version
    output_full = OUTPUT_DIR / "danish_leipzig_processed.csv"
    freq.to_csv(output_full, index=False)
    print(f"\n✓ Saved processed data: {output_full}")

    # Save simple format for linguistic_features.py (word\tproportion)
    output_simple = OUTPUT_DIR / "danish_leipzig_for_analysis.txt"
    freq[["word", "proportion"]].to_csv(
        output_simple, sep="\t", index=False, header=False
    )
    print(f"✓ Saved simple format: {output_simple}")

    # Show statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Total words: {len(freq):,}")
    print(f"Total frequency count: {freq['freq'].sum():,}")
    print(f"Mean frequency: {freq['freq'].mean():.1f}")
    print(f"Median frequency: {freq['freq'].median():.1f}")
    if ADD_LOG_TRANSFORM:
        print(f"Mean log frequency: {freq['log_freq'].mean():.2f}")
        print(f"Median log frequency: {freq['log_freq'].median():.2f}")

    # Show top 20 words
    print("\nTop 20 most frequent words:")
    display_cols = ["rank", "word", "freq"]
    if ADD_LOG_TRANSFORM:
        display_cols.append("log_freq")
    display_cols.append("proportion")
    print(freq.head(20)[display_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print(f"\nTo use with linguistic_features.py:")
    print(f"1. The file {output_simple.name} is already standardized (all lowercase)")
    print(f"2. Use in your code:")
    print(f"   DanishLinguisticFeatures(")
    print(f"       lemma_file='danish_frequencies/{output_simple.name}',")
    print(f"       use_proportions=True")
    print(f"   )")
    print("=" * 70)


if __name__ == "__main__":
    main()
