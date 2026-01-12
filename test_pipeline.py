#!/usr/bin/env python3
"""
Test script to generate sample data and verify the pipeline works correctly.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_tracks_data(n_samples=1000):
    """Generate sample tracks dataset."""
    np.random.seed(42)
    
    years = np.random.randint(1960, 2024, n_samples)
    months = np.random.randint(1, 13, n_samples)
    days = np.random.randint(1, 29, n_samples)
    
    data = {
        'id': [f'track_{i}' for i in range(n_samples)],
        'name': [f'Song {i}' for i in range(n_samples)],
        'popularity': np.random.randint(0, 101, n_samples),
        'duration_ms': np.random.randint(120000, 360000, n_samples),
        'explicit': np.random.choice([0, 1], n_samples),
        'artists': [f'Artist {i % 100}' for i in range(n_samples)],
        'release_date': [f'{y}-{m:02d}-{d:02d}' for y, m, d in zip(years, months, days)],
        'danceability': np.random.uniform(0, 1, n_samples),
        'energy': np.random.uniform(0, 1, n_samples),
        'key': np.random.randint(0, 12, n_samples),
        'loudness': np.random.uniform(-60, 0, n_samples),
        'mode': np.random.choice([0, 1], n_samples),
        'speechiness': np.random.uniform(0, 1, n_samples),
        'acousticness': np.random.uniform(0, 1, n_samples),
        'instrumentalness': np.random.uniform(0, 1, n_samples),
        'liveness': np.random.uniform(0, 1, n_samples),
        'valence': np.random.uniform(0, 1, n_samples),
        'tempo': np.random.uniform(60, 200, n_samples),
    }
    
    return pd.DataFrame(data)


def generate_sample_features_data(n_samples=500):
    """Generate sample features dataset."""
    np.random.seed(43)
    
    genres = ['pop', 'rock', 'hip-hop', 'jazz', 'classical', 'electronic', 
              'country', 'r&b', 'indie', 'metal']
    
    data = {
        'genre': np.random.choice(genres, n_samples),
        'artist_name': [f'Artist {i % 50}' for i in range(n_samples)],
        'track_name': [f'Track {i}' for i in range(n_samples)],
        'track_id': [f'feature_track_{i}' for i in range(n_samples)],
        'popularity': np.random.randint(0, 101, n_samples),
        'acousticness': np.random.uniform(0, 1, n_samples),
        'danceability': np.random.uniform(0, 1, n_samples),
        'duration_ms': np.random.randint(120000, 360000, n_samples),
        'energy': np.random.uniform(0, 1, n_samples),
        'instrumentalness': np.random.uniform(0, 1, n_samples),
        'key': np.random.randint(0, 12, n_samples),
        'liveness': np.random.uniform(0, 1, n_samples),
        'loudness': np.random.uniform(-60, 0, n_samples),
        'mode': np.random.choice([0, 1], n_samples),
        'speechiness': np.random.uniform(0, 1, n_samples),
        'tempo': np.random.uniform(60, 200, n_samples),
        'valence': np.random.uniform(0, 1, n_samples),
    }
    
    return pd.DataFrame(data)


def main():
    """Generate sample data and save to data directory."""
    print("Generating sample data for testing...")
    
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Generate and save tracks data
    tracks_df = generate_sample_tracks_data(1000)
    tracks_path = data_dir / 'dataset.csv'
    tracks_df.to_csv(tracks_path, index=False)
    print(f"✓ Generated tracks dataset: {tracks_path} ({len(tracks_df)} rows)")
    
    # Generate and save features data
    features_df = generate_sample_features_data(500)
    features_path = data_dir / 'SpotifyFeatures.csv'
    features_df.to_csv(features_path, index=False)
    print(f"✓ Generated features dataset: {features_path} ({len(features_df)} rows)")
    
    print("\nSample data generated successfully!")
    print("\nYou can now run the pipeline with:")
    print("  python main.py")
    print("\nOr with custom paths:")
    print(f"  python main.py --tracks {tracks_path} --features {features_path}")


if __name__ == '__main__':
    main()
