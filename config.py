# Data paths
DATA_ROOT = "D:"  # External drive location
COPCO_PATH = (
    f"{DATA_ROOT}/Dyslexia Research/CopCo_Data"  # Adjust based on your structure
)

# Analysis parameters
RANDOM_STATE = 42
N_BOOTSTRAP = 1000
CROSS_VALIDATION_FOLDS = 5

# Model parameters
MAX_WORD_LENGTH = 50  # Filter extremely long words
MIN_WORD_FREQ = 5  # Minimum corpus frequency
MAX_FIXATION_DURATION = 4000  # ms, filter outliers

# Feature extraction
USE_TRANSFORMER_PREDICTABILITY = True  # vs simple n-gram
LAUNCH_SITE_BINS = [0, 2, 4, 8, 15]  # Distance bins for preview analysis
SPACING_THRESHOLD = 0.5  # Character spacing threshold

# Visualization
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"  # or "png"
COLOR_PALETTE = "viridis"


# Helper function for accessing config values
def get(key, default=None):
    """Get configuration value with default fallback"""
    return globals().get(key, default)
