# Data paths
DATA_ROOT = "D:"  # External drive location
COPCO_PATH = (
    f"{DATA_ROOT}/Dyslexia Research/CopCo_Data"  # Adjust based on your structure
)


# Model parameters
MAX_WORD_LENGTH = 30  # Filter extremely long words
MAX_FIXATION_DURATION = 8000  # ms, filter outliers
MIN_FIXATION_DURATION = 80  # ms, filter outliers


# Helper function for accessing config values
def get(key, default=None):
    """Get configuration value with default fallback"""
    return globals().get(key, default)
