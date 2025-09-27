#!/usr/bin/env python3
"""
Setup script for Dyslexia Time Cost Analysis
Installation and environment configuration
"""

import os
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Ensure Python 3.8+"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} detected")


def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✓ All packages installed successfully")
    except subprocess.CalledProcessError:
        print("Error: Failed to install requirements")
        sys.exit(1)


def setup_nltk_data():
    """Download necessary NLTK data"""
    print("Setting up NLTK data...")
    import nltk

    try:
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        nltk.download("brown", quiet=True)
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"Warning: NLTK setup failed: {e}")


def setup_spacy_model():
    """Download spaCy English model"""
    print("Setting up spaCy model...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
        )
        print("✓ spaCy model installed")
    except subprocess.CalledProcessError:
        print("Warning: spaCy model installation failed")


def create_directories():
    """Create necessary directories"""
    dirs = [
        "data/processed",
        "data/features",
        "results/models",
        "results/figures",
        "results/tables",
        "cache",
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")


def setup_config():
    """Create configuration file"""
    config_content = """# Dyslexia Time Cost Analysis Configuration

    # Data paths
    DATA_ROOT = "D:"  # External drive location
    COPCO_PATH = f"{DATA_ROOT}/CopCo"  # Adjust based on your structure

    # Analysis parameters
    RANDOM_STATE = 42
    N_BOOTSTRAP = 1000
    CROSS_VALIDATION_FOLDS = 5

    # Model parameters
    MAX_WORD_LENGTH = 20  # Filter extremely long words
    MIN_WORD_FREQ = 5     # Minimum corpus frequency
    MAX_FIXATION_DURATION = 2000  # ms, filter outliers

    # Feature extraction
    USE_TRANSFORMER_PREDICTABILITY = True  # vs simple n-gram
    LAUNCH_SITE_BINS = [0, 2, 4, 8, 15]  # Distance bins for preview analysis
    SPACING_THRESHOLD = 0.5  # Character spacing threshold

    # Visualization
    FIGURE_DPI = 300
    FIGURE_FORMAT = "pdf"  # or "png"
    COLOR_PALETTE = "viridis"
    """

    with open("config.py", "w") as f:
        f.write(config_content)
    print("✓ Configuration file created")


def main():
    """Main setup routine"""
    print("=" * 50)
    print("Dyslexia Time Cost Analysis - Setup")
    print("=" * 50)

    check_python_version()
    install_requirements()
    setup_nltk_data()
    setup_spacy_model()
    create_directories()
    setup_config()

    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("Next steps:")
    print("1. Verify CopCo data location in config.py")
    print("2. Run: python main.py --explore")
    print("3. Run full analysis: python main.py --full-analysis")
    print("=" * 50)


if __name__ == "__main__":
    main()
