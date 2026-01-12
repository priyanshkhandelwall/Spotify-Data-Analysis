"""
Machine learning module for baseline popularity prediction.

This module implements simple regression models (Linear Regression and Random Forest)
to predict song popularity based on audio features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def prepare_ml_data(tracks_df, test_size=0.2, random_state=42):
    """
    Prepare data for machine learning: feature selection and train-test split.
    
    Args:
        tracks_df: Tracks DataFrame with audio features and popularity
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    # Select audio feature columns for prediction
    feature_cols = [
        'acousticness', 'danceability', 'energy', 'instrumentalness',
        'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
    ]
    
    # Filter to available columns
    available_features = [col for col in feature_cols if col in tracks_df.columns]
    
    if len(available_features) == 0:
        raise ValueError("No audio features found in dataset for modeling")
    
    # Prepare features and target
    df = tracks_df[available_features + ['popularity']].dropna()
    
    X = df[available_features]
    y = df['popularity']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"✓ ML data prepared: {len(available_features)} features, {len(X_train)} train, {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test, available_features


def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        LinearRegression: Trained model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("✓ Linear Regression model trained")
    
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """
    Train a Random Forest Regressor model.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees in the forest (default: 100)
        max_depth: Maximum depth of trees (default: 10)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        RandomForestRegressor: Trained model
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print("✓ Random Forest model trained")
    
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for reporting (default: "Model")
        
    Returns:
        dict: Evaluation metrics (RMSE, MAE, R²)
    """
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'model': model_name,
        'rmse': round(rmse, 3),
        'mae': round(mae, 3),
        'r2_score': round(r2, 3)
    }
    
    print(f"✓ {model_name} evaluation: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")
    
    return metrics


def get_feature_importance(model, feature_names, top_n=10):
    """
    Extract feature importance from tree-based models.
    
    Args:
        model: Trained tree-based model (e.g., RandomForest)
        feature_names: List of feature names
        top_n: Number of top features to return (default: 10)
        
    Returns:
        pd.DataFrame: Feature importance scores sorted by importance
    """
    if not hasattr(model, 'feature_importances_'):
        print("⚠ Warning: Model does not support feature importance")
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    print(f"✓ Feature importance extracted: Top {len(importance_df)} features")
    
    return importance_df


def baseline_popularity_model(tracks_df, model_type='random_forest'):
    """
    Train and evaluate a baseline popularity prediction model.
    
    Args:
        tracks_df: Tracks DataFrame with audio features and popularity
        model_type: Type of model to train ('linear_regression' or 'random_forest')
        
    Returns:
        dict: Contains model, metrics, feature importance, and predictions
    """
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_ml_data(tracks_df)
    
    # Train model
    if model_type == 'linear_regression':
        model = train_linear_regression(X_train, y_train)
        model_name = "Linear Regression"
    elif model_type == 'random_forest':
        model = train_random_forest(X_train, y_train)
        model_name = "Random Forest"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, model_name)
    
    # Get feature importance (if applicable)
    feature_importance = get_feature_importance(model, feature_names)
    
    # Generate predictions for visualization
    y_pred = model.predict(X_test)
    predictions_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred
    })
    
    print(f"✓ Baseline {model_name} model complete")
    
    return {
        'model': model,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'predictions': predictions_df,
        'feature_names': feature_names
    }
