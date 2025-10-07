"""
Column mapping definitions for CopCo dataset
Standardizes column names across different data sources
"""

# Mapping from CopCo ExtractedFeatures column names to standardized analysis names
EXTRACTED_FEATURES_MAPPING = {
    "trialId": "trial_id",
    "speechId": "speech_id",
    "paragraphId": "paragraph_id",
    "sentenceId": "sentence_id",
    "wordId": "word_position",
    "word": "word_text",
    "word_first_fix_dur": "first_fixation_duration",
    "word_first_pass_dur": "gaze_duration",
    "word_total_fix_dur": "total_reading_time",
    "landing_position": "landing_position",
    "number_of_fixations": "n_fixations",
}

# Mapping for participant statistics file
PARTICIPANT_STATS_MAPPING = {
    "subj": "subject_id",
    "participant": "subject_id",
    "subject": "subject_id",
    "part": "subject_id",
}

# Expected column names in different data files
EXPECTED_COLUMNS = {
    "extracted_features": [
        "part",
        "trialId",
        "speechId",
        "paragraphId",
        "sentenceId",
        "wordId",
        "word",
        "char_IA_ids",
        "landing_position",
        "word_first_fix_dur",
        "word_first_pass_dur",
        "word_go_past_time",
        "word_mean_fix_dur",
        "word_total_fix_dur",
        "number_of_fixations",
        "word_mean_sacc_dur",
        "word_peak_sacc_velocity",
    ],
    "participant_stats": [
        "subj",
        "comprehension_accuracy",
        "number_of_speeches",
        "number_of_questions",
        "absolute_reading_time",
        "relative_reading_time",
        "words_per_minute",
        "age",
        "sex",
        "native_language",
        "vision",
        "score_reading_comprehension_test",
        "dyslexia",
        "pseudohomophone_score",
    ],
    "text_stats": [
        "id",
        "frequency_prop",
        "number_of_sents",
        "tokens_per_speech",
        "types_per_speech",
        "tokens_per_sent",
        "avg_token_length",
        "text_type",
    ],
}

# Column data types for validation
COLUMN_DTYPES = {
    "subject_id": "object",
    "trial_id": "int64",
    "speech_id": "int64",
    "word_position": "int64",
    "word_text": "object",
    "first_fixation_duration": "float64",
    "gaze_duration": "float64",
    "total_reading_time": "float64",
    "n_fixations": "int64",
    "dyslexic": "bool",
    "word_length": "int64",
}


def get_mapping(source_type: str) -> dict:
    """
    Get column mapping for a specific data source

    Args:
        source_type: Type of data source ('extracted_features', 'participant_stats')

    Returns:
        Dictionary mapping source columns to standardized names
    """
    mappings = {
        "extracted_features": EXTRACTED_FEATURES_MAPPING,
        "participant_stats": PARTICIPANT_STATS_MAPPING,
    }

    return mappings.get(source_type, {})


def apply_mapping(df, source_type: str):
    """
    Apply column mapping to a DataFrame

    Args:
        df: Input DataFrame
        source_type: Type of data source

    Returns:
        DataFrame with renamed columns
    """
    mapping = get_mapping(source_type)

    # Only rename columns that exist in both the mapping and the DataFrame
    existing_mapping = {old: new for old, new in mapping.items() if old in df.columns}

    if existing_mapping:
        df = df.rename(columns=existing_mapping)

    return df
