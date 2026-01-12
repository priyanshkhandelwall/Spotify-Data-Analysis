"""
Visualization module for Spotify data analysis.

This module generates high-quality plots for popularity analysis,
audio feature relationships, genre trends, temporal trends, and ML results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style for high-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def plot_correlation_heatmap(correlation_matrix, output_path):
    """
    Generate a correlation heatmap for audio features.
    
    Args:
        correlation_matrix: Pandas DataFrame with correlation values
        output_path: Path to save the plot
        
    Returns:
        str: Path to saved plot
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        vmin=-1,
        vmax=1,
        center=0,
        cmap='RdBu_r',
        linewidths=0.5,
        linecolor='gray',
        square=True,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    
    plt.title('Audio Features Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    return str(output_path)


def plot_regression_pairs(regression_pairs, output_dir):
    """
    Generate regression plots for feature relationships.
    
    Args:
        regression_pairs: List of dicts with 'x', 'y', and 'data' keys
        output_dir: Directory to save plots
        
    Returns:
        list: Paths to saved plots
    """
    output_paths = []
    
    for pair in regression_pairs:
        x_col = pair['x']
        y_col = pair['y']
        data = pair['data']
        
        plt.figure(figsize=(10, 6))
        
        sns.regplot(
            data=data,
            x=x_col,
            y=y_col,
            scatter_kws={'alpha': 0.5, 's': 30},
            line_kws={'color': '#d62728', 'linewidth': 2},
            color='#1f77b4'
        )
        
        plt.title(f'{y_col.title()} vs {x_col.title()} Relationship',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel(x_col.title(), fontsize=12)
        plt.ylabel(y_col.title(), fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'regression_{y_col}_vs_{x_col}.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        output_paths.append(str(output_path))
        print(f"✓ Saved: {output_path}")
    
    return output_paths


def plot_popularity_distribution(tracks_df, output_path):
    """
    Plot distribution of song popularity.
    
    Args:
        tracks_df: Tracks DataFrame with 'popularity' column
        output_path: Path to save the plot
        
    Returns:
        str: Path to saved plot
    """
    plt.figure(figsize=(10, 6))
    
    sns.histplot(
        data=tracks_df,
        x='popularity',
        bins=50,
        kde=True,
        color='#2ca02c',
        edgecolor='black',
        linewidth=0.5
    )
    
    plt.title('Distribution of Song Popularity', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Popularity Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    return str(output_path)


def plot_genre_trends(genre_stats, output_dir, top_n=15):
    """
    Plot genre-based trends: duration and popularity.
    
    Args:
        genre_stats: DataFrame with genre statistics
        output_dir: Directory to save plots
        top_n: Number of top genres to display (default: 15)
        
    Returns:
        list: Paths to saved plots
    """
    output_paths = []
    
    # Top genres by popularity
    plt.figure(figsize=(10, 8))
    top_genres = genre_stats.nlargest(top_n, 'avg_popularity')
    
    sns.barplot(
        data=top_genres.reset_index(),
        y='genre',
        x='avg_popularity',
        palette='viridis',
        edgecolor='black',
        linewidth=0.5
    )
    
    plt.title(f'Top {top_n} Genres by Average Popularity', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Average Popularity', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'genre_popularity.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    output_paths.append(str(output_path))
    print(f"✓ Saved: {output_path}")
    
    # Genre duration
    plt.figure(figsize=(10, 8))
    top_duration = genre_stats.nlargest(top_n, 'avg_duration_ms')
    
    sns.barplot(
        data=top_duration.reset_index(),
        y='genre',
        x='avg_duration_ms',
        palette='coolwarm',
        edgecolor='black',
        linewidth=0.5
    )
    
    plt.title(f'Top {top_n} Genres by Average Duration', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Average Duration (ms)', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'genre_duration.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    output_paths.append(str(output_path))
    print(f"✓ Saved: {output_path}")
    
    return output_paths


def plot_temporal_trends(temporal_data, output_dir):
    """
    Plot temporal trends: songs per year and duration over time.
    
    Args:
        temporal_data: Dict with temporal analysis results
        output_dir: Directory to save plots
        
    Returns:
        list: Paths to saved plots
    """
    if temporal_data is None:
        print("⚠ Skipping temporal plots: no temporal data available")
        return []
    
    output_paths = []
    
    # Songs per year
    plt.figure(figsize=(14, 6))
    songs_per_year = temporal_data['songs_per_year']
    
    plt.bar(
        songs_per_year.index,
        songs_per_year.values,
        color='#ff7f0e',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.8
    )
    
    plt.title('Number of Songs Released Per Year', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Songs', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'songs_per_year.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    output_paths.append(str(output_path))
    print(f"✓ Saved: {output_path}")
    
    # Duration over time
    if temporal_data['avg_duration_per_year'] is not None:
        plt.figure(figsize=(14, 6))
        duration_data = temporal_data['avg_duration_per_year']
        
        plt.plot(
            duration_data.index,
            duration_data.values,
            marker='o',
            linewidth=2,
            markersize=4,
            color='#9467bd',
            alpha=0.8
        )
        
        plt.title('Average Song Duration Over Time', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Average Duration (seconds)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = Path(output_dir) / 'duration_over_time.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        output_paths.append(str(output_path))
        print(f"✓ Saved: {output_path}")
    
    return output_paths


def plot_model_predictions(predictions_df, metrics, output_path):
    """
    Plot actual vs predicted popularity for ML model.
    
    Args:
        predictions_df: DataFrame with 'actual' and 'predicted' columns
        metrics: Dict with model evaluation metrics
        output_path: Path to save the plot
        
    Returns:
        str: Path to saved plot
    """
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(
        predictions_df['actual'],
        predictions_df['predicted'],
        alpha=0.5,
        s=20,
        color='#17becf',
        edgecolors='black',
        linewidth=0.3
    )
    
    # Perfect prediction line
    min_val = min(predictions_df['actual'].min(), predictions_df['predicted'].min())
    max_val = max(predictions_df['actual'].max(), predictions_df['predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add metrics text
    metrics_text = f"R² = {metrics['r2_score']:.3f}\nRMSE = {metrics['rmse']:.3f}\nMAE = {metrics['mae']:.3f}"
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.title(f"{metrics['model']} - Actual vs Predicted Popularity",
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Actual Popularity', fontsize=12)
    plt.ylabel('Predicted Popularity', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    return str(output_path)


def plot_feature_importance(feature_importance_df, output_path):
    """
    Plot feature importance for tree-based models.
    
    Args:
        feature_importance_df: DataFrame with 'feature' and 'importance' columns
        output_path: Path to save the plot
        
    Returns:
        str: Path to saved plot or None if no data
    """
    if feature_importance_df is None or feature_importance_df.empty:
        print("⚠ Skipping feature importance plot: no data available")
        return None
    
    plt.figure(figsize=(10, 6))
    
    sns.barplot(
        data=feature_importance_df,
        y='feature',
        x='importance',
        palette='rocket',
        edgecolor='black',
        linewidth=0.5
    )
    
    plt.title('Feature Importance for Popularity Prediction', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    return str(output_path)


def generate_all_plots(tracks_df, features_df, analysis_results, ml_results, output_dir):
    """
    Generate all visualization plots and save to output directory.
    
    Args:
        tracks_df: Tracks DataFrame
        features_df: Features DataFrame
        analysis_results: Dict with analysis results
        ml_results: Dict with ML model results (or None)
        output_dir: Directory to save all plots
        
    Returns:
        dict: Paths to all generated plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_paths = {}
    
    # Popularity distribution
    plot_paths['popularity_dist'] = plot_popularity_distribution(
        tracks_df,
        output_dir / 'popularity_distribution.png'
    )
    
    # Correlation heatmap
    if 'audio_features' in analysis_results:
        plot_paths['correlation_heatmap'] = plot_correlation_heatmap(
            analysis_results['audio_features']['correlation_matrix'],
            output_dir / 'correlation_heatmap.png'
        )
        
        # Regression plots
        plot_paths['regression_plots'] = plot_regression_pairs(
            analysis_results['audio_features']['regression_pairs'],
            output_dir
        )
    
    # Genre trends
    if 'genre_trends' in analysis_results:
        plot_paths['genre_plots'] = plot_genre_trends(
            analysis_results['genre_trends']['genre_stats'],
            output_dir
        )
    
    # Temporal trends
    if 'temporal_trends' in analysis_results:
        plot_paths['temporal_plots'] = plot_temporal_trends(
            analysis_results['temporal_trends'],
            output_dir
        )
    
    # ML model plots
    if ml_results is not None:
        plot_paths['model_predictions'] = plot_model_predictions(
            ml_results['predictions'],
            ml_results['metrics'],
            output_dir / 'model_predictions.png'
        )
        
        if ml_results['feature_importance'] is not None:
            plot_paths['feature_importance'] = plot_feature_importance(
                ml_results['feature_importance'],
                output_dir / 'feature_importance.png'
            )
    
    print(f"✓ All plots generated and saved to {output_dir}")
    
    return plot_paths
