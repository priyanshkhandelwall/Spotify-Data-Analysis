"""
Analysis functions for Spotify data exploration.

This module contains functions for popularity analysis, audio feature
relationships, genre trends, and temporal analysis.
"""

import pandas as pd
import numpy as np


def compute_popularity_analysis(tracks_df):
    """
    Analyze song popularity: top and least popular songs.
    
    Args:
        tracks_df: Tracks DataFrame with 'popularity', 'name', and 'artists' columns
        
    Returns:
        dict: Contains 'top_10' and 'least_10' DataFrames
    """
    # Top 10 most popular songs
    top_10 = tracks_df.nlargest(10, 'popularity')[['name', 'popularity', 'artists']].copy()
    
    # 10 least popular songs
    least_10 = tracks_df.nsmallest(10, 'popularity')[['name', 'popularity']].copy()
    
    print(f"✓ Popularity analysis complete: Top 10 and Least 10 songs identified")
    
    return {
        'top_10': top_10,
        'least_10': least_10
    }


def analyze_audio_feature_relationships(tracks_df):
    """
    Analyze relationships between audio features (correlation, regression data).
    
    Args:
        tracks_df: Tracks DataFrame with audio feature columns
        
    Returns:
        dict: Contains correlation matrix and sample data for regression plots
    """
    # Select numeric columns for correlation (exclude non-audio features)
    exclude_cols = ['key', 'mode', 'explicit', 'duration_ms', 'duration']
    numeric_cols = tracks_df.select_dtypes(include=[np.number]).columns.tolist()
    audio_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Compute correlation matrix
    correlation_matrix = tracks_df[audio_cols].corr(method='pearson')
    
    # Sample 4% of data for regression plots (to avoid overplotting)
    sample_size = max(int(0.04 * len(tracks_df)), 100)
    sample_df = tracks_df.sample(n=min(sample_size, len(tracks_df)), random_state=42)
    
    # Prepare regression data pairs
    regression_pairs = []
    
    if 'loudness' in sample_df.columns and 'energy' in sample_df.columns:
        regression_pairs.append({
            'x': 'energy',
            'y': 'loudness',
            'data': sample_df[['energy', 'loudness']].dropna()
        })
    
    if 'popularity' in sample_df.columns and 'acousticness' in sample_df.columns:
        regression_pairs.append({
            'x': 'acousticness',
            'y': 'popularity',
            'data': sample_df[['acousticness', 'popularity']].dropna()
        })
    
    print(f"✓ Audio feature analysis complete: {len(audio_cols)} features analyzed")
    
    return {
        'correlation_matrix': correlation_matrix,
        'regression_pairs': regression_pairs,
        'sample_size': len(sample_df)
    }


def analyze_genre_trends(features_df):
    """
    Analyze genre-based trends: duration and popularity by genre.
    
    Args:
        features_df: Features DataFrame with 'genre', 'duration_ms', 'popularity'
        
    Returns:
        dict: Contains genre statistics and top genres by popularity
    """
    # Average duration by genre
    genre_duration = features_df.groupby('genre')['duration_ms'].mean().sort_values(ascending=False)
    
    # Average popularity by genre
    genre_popularity = features_df.groupby('genre')['popularity'].mean().sort_values(ascending=False)
    
    # Top 10 genres by popularity
    top_genres = genre_popularity.head(10)
    
    # Genre statistics
    genre_stats = pd.DataFrame({
        'avg_duration_ms': genre_duration,
        'avg_popularity': genre_popularity,
        'track_count': features_df.groupby('genre').size()
    }).sort_values('avg_popularity', ascending=False)
    
    print(f"✓ Genre analysis complete: {len(genre_stats)} genres analyzed")
    
    return {
        'genre_stats': genre_stats,
        'top_genres': top_genres,
        'genre_duration': genre_duration,
        'genre_popularity': genre_popularity
    }


def analyze_temporal_trends(tracks_df):
    """
    Analyze temporal trends: songs per year, duration trends over time.
    
    Args:
        tracks_df: Tracks DataFrame with 'release_year' and 'duration' columns
        
    Returns:
        dict: Contains yearly statistics and trends
    """
    if 'release_year' not in tracks_df.columns:
        print("⚠ Warning: 'release_year' column not found. Skipping temporal analysis.")
        return None
    
    # Remove rows with missing years
    df = tracks_df.dropna(subset=['release_year'])
    
    # Songs per year
    songs_per_year = df['release_year'].value_counts().sort_index()
    
    # Average duration per year
    if 'duration' in df.columns:
        avg_duration_per_year = df.groupby('release_year')['duration'].mean()
    else:
        avg_duration_per_year = None
    
    # Average popularity per year
    if 'popularity' in df.columns:
        avg_popularity_per_year = df.groupby('release_year')['popularity'].mean()
    else:
        avg_popularity_per_year = None
    
    # Year range
    year_range = (int(df['release_year'].min()), int(df['release_year'].max()))
    
    print(f"✓ Temporal analysis complete: {year_range[0]}-{year_range[1]}")
    
    return {
        'songs_per_year': songs_per_year,
        'avg_duration_per_year': avg_duration_per_year,
        'avg_popularity_per_year': avg_popularity_per_year,
        'year_range': year_range
    }


def generate_summary_statistics(tracks_df, features_df):
    """
    Generate comprehensive summary statistics for reporting.
    
    Args:
        tracks_df: Tracks DataFrame
        features_df: Features DataFrame
        
    Returns:
        pd.DataFrame: Summary statistics table
    """
    summary_data = []
    
    # Tracks dataset statistics
    summary_data.append({
        'Dataset': 'Tracks',
        'Total Records': len(tracks_df),
        'Avg Popularity': round(tracks_df['popularity'].mean(), 2),
        'Avg Duration (s)': round(tracks_df['duration'].mean(), 2) if 'duration' in tracks_df.columns else 'N/A',
        'Year Range': f"{tracks_df['release_year'].min():.0f}-{tracks_df['release_year'].max():.0f}" if 'release_year' in tracks_df.columns else 'N/A'
    })
    
    # Features dataset statistics
    summary_data.append({
        'Dataset': 'Features',
        'Total Records': len(features_df),
        'Avg Popularity': round(features_df['popularity'].mean(), 2),
        'Avg Duration (s)': round(features_df['duration_ms'].mean() / 1000, 2) if 'duration_ms' in features_df.columns else 'N/A',
        'Unique Genres': features_df['genre'].nunique() if 'genre' in features_df.columns else 'N/A'
    })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("✓ Summary statistics generated")
    
    return summary_df
