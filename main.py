#!/usr/bin/env python3
"""
Spotify Data Analysis Pipeline - Main Entry Point

This script orchestrates the complete analysis pipeline:
1. Load and validate data
2. Perform exploratory data analysis
3. Generate visualizations
4. Train baseline ML model (optional)
5. Save all outputs

Usage:
    python main.py --tracks data/dataset.csv --features data/SpotifyFeatures.csv
    python main.py --tracks data/dataset.csv --features data/SpotifyFeatures.csv --skip-ml
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import (
    load_csvs, validate_data, preprocess_tracks,
    preprocess_features, get_data_summary
)
from analysis import (
    compute_popularity_analysis, analyze_audio_feature_relationships,
    analyze_genre_trends, analyze_temporal_trends, generate_summary_statistics
)
from modeling import baseline_popularity_model
from visualization import generate_all_plots
from output_manager import save_all_outputs


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Spotify Data Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --tracks data/dataset.csv --features data/SpotifyFeatures.csv
  python main.py --tracks data/dataset.csv --features data/SpotifyFeatures.csv --skip-ml
  python main.py --tracks data/dataset.csv --features data/SpotifyFeatures.csv --model linear_regression
        """
    )
    
    parser.add_argument(
        '--tracks',
        type=str,
        default='data/dataset.csv',
        help='Path to tracks dataset CSV file (default: data/dataset.csv)'
    )
    
    parser.add_argument(
        '--features',
        type=str,
        default='data/SpotifyFeatures.csv',
        help='Path to features dataset CSV file (default: data/SpotifyFeatures.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Base directory for outputs (default: outputs)'
    )
    
    parser.add_argument(
        '--skip-ml',
        action='store_true',
        help='Skip machine learning model training'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['linear_regression', 'random_forest'],
        default='random_forest',
        help='ML model type to train (default: random_forest)'
    )
    
    return parser.parse_args()


def print_header():
    """Print pipeline header."""
    print("\n" + "=" * 70)
    print(" " * 15 + "SPOTIFY DATA ANALYSIS PIPELINE")
    print("=" * 70 + "\n")


def print_section(title):
    """Print section header."""
    print(f"\n{'-' * 70}")
    print(f"  {title}")
    print(f"{'-' * 70}\n")


def main():
    """Main pipeline execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Print header
    print_header()
    
    try:
        # ===================================================================
        # PHASE 1: DATA LOADING AND VALIDATION
        # ===================================================================
        print_section("PHASE 1: DATA LOADING AND VALIDATION")
        
        tracks_df, features_df = load_csvs(args.tracks, args.features)
        validate_data(tracks_df, features_df)
        
        # Preprocess data
        tracks_df = preprocess_tracks(tracks_df)
        features_df = preprocess_features(features_df)
        
        # Get data summary
        data_summary = get_data_summary(tracks_df, features_df)
        print(f"\nData Summary:")
        print(f"  Tracks: {data_summary['tracks_count']} records")
        print(f"  Features: {data_summary['features_count']} records")
        if 'unique_genres' in data_summary:
            print(f"  Unique Genres: {data_summary['unique_genres']}")
        if 'year_range' in data_summary:
            print(f"  Year Range: {data_summary['year_range'][0]:.0f} - {data_summary['year_range'][1]:.0f}")
        
        # ===================================================================
        # PHASE 2: EXPLORATORY DATA ANALYSIS
        # ===================================================================
        print_section("PHASE 2: EXPLORATORY DATA ANALYSIS")
        
        analysis_results = {}
        
        # Popularity analysis
        print("Running popularity analysis...")
        analysis_results['popularity'] = compute_popularity_analysis(tracks_df)
        
        # Audio feature relationships
        print("Analyzing audio feature relationships...")
        analysis_results['audio_features'] = analyze_audio_feature_relationships(tracks_df)
        
        # Genre trends
        print("Analyzing genre trends...")
        analysis_results['genre_trends'] = analyze_genre_trends(features_df)
        
        # Temporal trends
        print("Analyzing temporal trends...")
        analysis_results['temporal_trends'] = analyze_temporal_trends(tracks_df)
        
        # Generate summary statistics
        print("Generating summary statistics...")
        summary_df = generate_summary_statistics(tracks_df, features_df)
        
        # ===================================================================
        # PHASE 3: MACHINE LEARNING (OPTIONAL)
        # ===================================================================
        ml_results = None
        
        if not args.skip_ml:
            print_section("PHASE 3: BASELINE ML MODEL TRAINING")
            
            try:
                print(f"Training {args.model} model...")
                ml_results = baseline_popularity_model(tracks_df, model_type=args.model)
                
                print(f"\nModel Performance:")
                print(f"  Model: {ml_results['metrics']['model']}")
                print(f"  RMSE: {ml_results['metrics']['rmse']}")
                print(f"  MAE: {ml_results['metrics']['mae']}")
                print(f"  R² Score: {ml_results['metrics']['r2_score']}")
                
            except Exception as e:
                print(f"⚠ Warning: ML model training failed: {e}")
                print("Continuing without ML results...")
                ml_results = None
        else:
            print_section("PHASE 3: MACHINE LEARNING (SKIPPED)")
            print("ML model training skipped as requested.")
        
        # ===================================================================
        # PHASE 4: VISUALIZATION
        # ===================================================================
        print_section("PHASE 4: GENERATING VISUALIZATIONS")
        
        plots_dir = Path(args.output) / 'plots'
        plot_paths = generate_all_plots(
            tracks_df,
            features_df,
            analysis_results,
            ml_results,
            plots_dir
        )
        
        # ===================================================================
        # PHASE 5: SAVE OUTPUTS
        # ===================================================================
        print_section("PHASE 5: SAVING OUTPUTS")
        
        saved_paths = save_all_outputs(
            summary_df,
            analysis_results,
            ml_results,
            plot_paths,
            args.output
        )
        
        # ===================================================================
        # COMPLETION
        # ===================================================================
        print("\n" + "=" * 70)
        print(" " * 20 + "PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nAll outputs saved to: {Path(args.output).absolute()}")
        print(f"  - Plots: {Path(args.output).absolute() / 'plots'}")
        print(f"  - Reports: {Path(args.output).absolute() / 'reports'}")
        print(f"  - Manifest: {saved_paths['manifest']}")
        print("\n" + "=" * 70 + "\n")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check that the input file paths are correct.")
        return 1
        
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check that your datasets have the required columns.")
        return 1
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
