#!/usr/bin/env python3
"""
Quick fix for config.py to add the get() function
"""


def fix_config():
    """Add get() function to existing config.py"""

    # Read current config
    try:
        with open("config.py", "r") as f:
            current_config = f.read()
    except FileNotFoundError:
        print("config.py not found. Creating new one...")
        current_config = ""

    # Check if get function already exists
    if "def get(" in current_config:
        print("✓ config.py already has get() function")
        return

    # Add get function if missing
    get_function = '''
# Helper function for accessing config values
def get(key, default=None):
    """Get configuration value with default fallback"""
    return globals().get(key, default)
'''

    # If config is empty, create full config
    if not current_config.strip():
        full_config = (
            """# Dyslexia Time Cost Analysis Configuration

# Data paths
DATA_ROOT = "D:"  # External drive location
COPCO_PATH = r"D:\\Dyslexia Research\\CopCo_Data"  # Your actual CopCo path

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
FIGURE_DIR = "results/figures"
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"  # or "png"
COLOR_PALETTE = "viridis"
"""
            + get_function
        )

        with open("config.py", "w") as f:
            f.write(full_config)
        print("✓ Created complete config.py with get() function")

    else:
        # Add get function to existing config
        updated_config = current_config + get_function

        with open("config.py", "w") as f:
            f.write(updated_config)
        print("✓ Added get() function to existing config.py")


if __name__ == "__main__":
    fix_config()
