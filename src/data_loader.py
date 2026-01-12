"""
Data loading and validation module for Spotify analysis pipeline.

This module handles CSV ingestion, data validation, and preprocessing
for Spotify tracks and features datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_csvs(tracks_path, features_path):
    """
    Load Spotify tracks and features CSV files.
    
    Args:
        tracks_path: Path to the tracks dataset CSV file
        features_path: Path to the features dataset CSV file
        
    Returns:
        tuple: (tracks_df, features_df) - Two pandas DataFrames
        
    Raises:
        FileNotFoundError: If either CSV file does not exist
        pd.errors.EmptyDataError: If CSV files are empty
    """
    tracks_path = Path(tracks_path)
    features_path = Path(features_path)
    
    if not tracks_path.exists():
        raise FileNotFoundError(f"Tracks dataset not found: {tracks_path}")
    
    if not features_path.exists():
        raise FileNotFoundError(f"Features dataset not found: {features_path}")
    
    try:
        tracks_df = pd.read_csv(tracks_path)
        features_df = pd.read_csv(features_path)
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"One or more CSV files are empty: {e}")
    
    print(f"✓ Loaded tracks dataset: {len(tracks_df)} rows, {len(tracks_df.columns)} columns")
    print(f"✓ Loaded features dataset: {len(features_df)} rows, {len(features_df.columns)} columns")
    
    return tracks_df, features_df


def validate_data(tracks_df, features_df):
    """
    Validate that required columns exist in the datasets.
    
    Args:
        tracks_df: Tracks DataFrame
        features_df: Features DataFrame
        
    Raises:
        ValueError: If required columns are missing
    """
    # Required columns for tracks dataset
    required_tracks_cols = [
        'name', 'popularity', 'duration_ms', 'release_date'
    ]
    
    # Required columns for features dataset
    required_features_cols = [
        'genre', 'popularity', 'duration_ms'
    ]
    
    # Check tracks columns
    missing_tracks = [col for col in required_tracks_cols if col not in tracks_df.columns]
    if missing_tracks:
        raise ValueError(f"Missing required columns in tracks dataset: {missing_tracks}")
    
    # Check features columns
    missing_features = [col for col in required_features_cols if col not in features_df.columns]
    if missing_features:
        raise ValueError(f"Missing required columns in features dataset: {missing_features}")
    
    # Check for excessive null values
    tracks_null_pct = (tracks_df.isnull().sum() / len(tracks_df) * 100)
    features_null_pct = (features_df.isnull().sum() / len(features_df) * 100)
    
    high_null_tracks = tracks_null_pct[tracks_null_pct > 50]
    high_null_features = features_null_pct[features_null_pct > 50]
    
    if not high_null_tracks.empty:
        print(f"⚠ Warning: Tracks columns with >50% null values: {high_null_tracks.to_dict()}")
    
    if not high_null_features.empty:
        print(f"⚠ Warning: Features columns with >50% null values: {high_null_features.to_dict()}")
    
    print("✓ Data validation passed")


def preprocess_tracks(tracks_df):
    """
    Preprocess tracks dataset: convert duration, parse dates, handle nulls.
    
    Args:
        tracks_df: Raw tracks DataFrame
        
    Returns:
        pd.DataFrame: Preprocessed tracks DataFrame
    """
    df = tracks_df.copy()
    
    # Convert duration from milliseconds to seconds
    if 'duration_ms' in df.columns:
        df['duration'] = df['duration_ms'].apply(lambda x: round(x / 1000) if pd.notna(x) else np.nan)
    
    # Parse release dates
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
    
    # Drop rows with missing critical values
    df = df.dropna(subset=['popularity', 'name'])
    
    print(f"✓ Preprocessed tracks: {len(df)} rows retained")
    return df


def preprocess_features(features_df):
    """
    Preprocess features dataset: handle nulls and outliers.
    
    Args:
        features_df: Raw features DataFrame
        
    Returns:
        pd.DataFrame: Preprocessed features DataFrame
    """
    df = features_df.copy()
    
    # Drop rows with missing genre or popularity
    df = df.dropna(subset=['genre', 'popularity'])
    
    print(f"✓ Preprocessed features: {len(df)} rows retained")
    return df


def get_data_summary(tracks_df, features_df):
    """
    Generate a summary of the loaded datasets.
    
    Args:
        tracks_df: Tracks DataFrame
        features_df: Features DataFrame
        
    Returns:
        dict: Summary statistics and metadata
    """
    summary = {
        'tracks_count': len(tracks_df),
        'features_count': len(features_df),
        'tracks_columns': list(tracks_df.columns),
        'features_columns': list(features_df.columns),
        'tracks_null_counts': tracks_df.isnull().sum().to_dict(),
        'features_null_counts': features_df.isnull().sum().to_dict(),
        'popularity_range_tracks': (tracks_df['popularity'].min(), tracks_df['popularity'].max()),
        'popularity_range_features': (features_df['popularity'].min(), features_df['popularity'].max()),
    }
    
    if 'release_year' in tracks_df.columns:
        summary['year_range'] = (
            tracks_df['release_year'].min(),
            tracks_df['release_year'].max()
        )
    
    if 'genre' in features_df.columns:
        summary['unique_genres'] = features_df['genre'].nunique()
    
    return summary
