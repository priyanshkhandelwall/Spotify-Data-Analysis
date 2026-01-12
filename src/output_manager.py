"""
Output management module for saving analysis results and reports.

This module handles saving CSV reports, summary statistics, and organizing outputs.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


def save_summary_report(summary_df, output_path):
    """
    Save summary statistics to CSV.
    
    Args:
        summary_df: DataFrame with summary statistics
        output_path: Path to save the CSV file
        
    Returns:
        str: Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary_df.to_csv(output_path, index=False)
    
    print(f"✓ Saved summary report: {output_path}")
    return str(output_path)


def save_popularity_analysis(popularity_results, output_dir):
    """
    Save popularity analysis results (top and least popular songs).
    
    Args:
        popularity_results: Dict with 'top_10' and 'least_10' DataFrames
        output_dir: Directory to save CSV files
        
    Returns:
        dict: Paths to saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save top 10 popular songs
    top_path = output_dir / 'top_10_popular_songs.csv'
    popularity_results['top_10'].to_csv(top_path, index=False)
    paths['top_10'] = str(top_path)
    print(f"✓ Saved: {top_path}")
    
    # Save least 10 popular songs
    least_path = output_dir / 'least_10_popular_songs.csv'
    popularity_results['least_10'].to_csv(least_path, index=False)
    paths['least_10'] = str(least_path)
    print(f"✓ Saved: {least_path}")
    
    return paths


def save_genre_analysis(genre_results, output_dir):
    """
    Save genre analysis results.
    
    Args:
        genre_results: Dict with genre statistics
        output_dir: Directory to save CSV files
        
    Returns:
        str: Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'genre_statistics.csv'
    genre_results['genre_stats'].to_csv(output_path)
    
    print(f"✓ Saved: {output_path}")
    return str(output_path)


def save_temporal_analysis(temporal_results, output_dir):
    """
    Save temporal analysis results.
    
    Args:
        temporal_results: Dict with temporal statistics
        output_dir: Directory to save CSV files
        
    Returns:
        dict: Paths to saved files
    """
    if temporal_results is None:
        print("⚠ Skipping temporal analysis save: no data available")
        return {}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save songs per year
    songs_path = output_dir / 'songs_per_year.csv'
    temporal_results['songs_per_year'].to_csv(songs_path, header=['count'])
    paths['songs_per_year'] = str(songs_path)
    print(f"✓ Saved: {songs_path}")
    
    # Save duration per year if available
    if temporal_results['avg_duration_per_year'] is not None:
        duration_path = output_dir / 'avg_duration_per_year.csv'
        temporal_results['avg_duration_per_year'].to_csv(duration_path, header=['avg_duration'])
        paths['avg_duration'] = str(duration_path)
        print(f"✓ Saved: {duration_path}")
    
    return paths


def save_ml_results(ml_results, output_dir):
    """
    Save machine learning model results and metrics.
    
    Args:
        ml_results: Dict with ML model results
        output_dir: Directory to save files
        
    Returns:
        dict: Paths to saved files
    """
    if ml_results is None:
        print("⚠ Skipping ML results save: no model trained")
        return {}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save model metrics
    metrics_df = pd.DataFrame([ml_results['metrics']])
    metrics_path = output_dir / 'model_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    paths['metrics'] = str(metrics_path)
    print(f"✓ Saved: {metrics_path}")
    
    # Save feature importance if available
    if ml_results['feature_importance'] is not None:
        importance_path = output_dir / 'feature_importance.csv'
        ml_results['feature_importance'].to_csv(importance_path, index=False)
        paths['feature_importance'] = str(importance_path)
        print(f"✓ Saved: {importance_path}")
    
    # Save predictions sample
    predictions_path = output_dir / 'model_predictions_sample.csv'
    ml_results['predictions'].head(100).to_csv(predictions_path, index=False)
    paths['predictions'] = str(predictions_path)
    print(f"✓ Saved: {predictions_path}")
    
    return paths


def create_analysis_manifest(analysis_results, ml_results, plot_paths, output_dir):
    """
    Create a manifest file listing all generated outputs.
    
    Args:
        analysis_results: Dict with analysis results
        ml_results: Dict with ML results
        plot_paths: Dict with plot file paths
        output_dir: Directory to save manifest
        
    Returns:
        str: Path to manifest file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'analysis_completed': True,
        'ml_model_trained': ml_results is not None,
        'outputs': {
            'plots_directory': str(output_dir / 'plots'),
            'reports_directory': str(output_dir / 'reports'),
        }
    }
    
    # Add plot counts
    total_plots = sum(len(v) if isinstance(v, list) else 1 for v in plot_paths.values() if v)
    manifest['total_plots_generated'] = total_plots
    
    # Save manifest as text file
    manifest_path = output_dir / 'analysis_manifest.txt'
    
    with open(manifest_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SPOTIFY DATA ANALYSIS PIPELINE - OUTPUT MANIFEST\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Timestamp: {manifest['timestamp']}\n")
        f.write(f"ML Model Trained: {'Yes' if manifest['ml_model_trained'] else 'No'}\n")
        f.write(f"Total Plots Generated: {manifest['total_plots_generated']}\n\n")
        f.write("-" * 60 + "\n")
        f.write("OUTPUT DIRECTORIES:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Plots: {manifest['outputs']['plots_directory']}\n")
        f.write(f"Reports: {manifest['outputs']['reports_directory']}\n\n")
        f.write("-" * 60 + "\n")
        f.write("ANALYSIS COMPONENTS COMPLETED:\n")
        f.write("-" * 60 + "\n")
        f.write("✓ Popularity Analysis\n")
        f.write("✓ Audio Feature Relationships\n")
        f.write("✓ Genre Trends Analysis\n")
        f.write("✓ Temporal Trends Analysis\n")
        if manifest['ml_model_trained']:
            f.write("✓ Baseline ML Model (Popularity Prediction)\n")
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"✓ Created analysis manifest: {manifest_path}")
    return str(manifest_path)


def save_all_outputs(summary_df, analysis_results, ml_results, plot_paths, output_base_dir):
    """
    Save all analysis outputs (reports, CSVs, manifest).
    
    Args:
        summary_df: Summary statistics DataFrame
        analysis_results: Dict with all analysis results
        ml_results: Dict with ML model results
        plot_paths: Dict with plot file paths
        output_base_dir: Base directory for all outputs
        
    Returns:
        dict: Paths to all saved outputs
    """
    output_base_dir = Path(output_base_dir)
    reports_dir = output_base_dir / 'reports'
    
    saved_paths = {
        'reports': {},
        'plots': plot_paths,
        'manifest': None
    }
    
    # Save summary report
    saved_paths['reports']['summary'] = save_summary_report(
        summary_df,
        reports_dir / 'summary.csv'
    )
    
    # Save popularity analysis
    if 'popularity' in analysis_results:
        saved_paths['reports']['popularity'] = save_popularity_analysis(
            analysis_results['popularity'],
            reports_dir
        )
    
    # Save genre analysis
    if 'genre_trends' in analysis_results:
        saved_paths['reports']['genre'] = save_genre_analysis(
            analysis_results['genre_trends'],
            reports_dir
        )
    
    # Save temporal analysis
    if 'temporal_trends' in analysis_results:
        saved_paths['reports']['temporal'] = save_temporal_analysis(
            analysis_results['temporal_trends'],
            reports_dir
        )
    
    # Save ML results
    if ml_results is not None:
        saved_paths['reports']['ml'] = save_ml_results(
            ml_results,
            reports_dir
        )
    
    # Create manifest
    saved_paths['manifest'] = create_analysis_manifest(
        analysis_results,
        ml_results,
        plot_paths,
        output_base_dir
    )
    
    print(f"\n{'=' * 60}")
    print("✓ ALL OUTPUTS SAVED SUCCESSFULLY")
    print(f"{'=' * 60}")
    print(f"Base directory: {output_base_dir}")
    print(f"Reports: {reports_dir}")
    print(f"Plots: {output_base_dir / 'plots'}")
    print(f"{'=' * 60}\n")
    
    return saved_paths
