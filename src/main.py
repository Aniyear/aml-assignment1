"""
Main execution script for water surface classification.
Orchestrates data loading, preprocessing, feature engineering, model training, and evaluation.
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from preprocessing import DataPreprocessor
from features import FeatureEngineer
from models import ModelFactory
from evaluation import ModelEvaluator


def generate_synthetic_dataset(n_samples=5000, random_state=42):
    """
    Generate synthetic satellite imagery dataset for water surface classification.
    Simulates Sentinel-2-like spectral bands and water indices.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Synthetic dataset with spectral bands and water surface labels
    """
    np.random.seed(random_state)
    
    # Generate spectral bands (normalized reflectance values 0-1)
    # Sentinel-2 bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2)
    
    # Water pixels (lower NIR, higher green, lower SWIR)
    n_water = n_samples // 2
    n_non_water = n_samples - n_water
    
    # Water surface characteristics
    water_data = {
        'B2_Blue': np.random.normal(0.15, 0.05, n_water),
        'B3_Green': np.random.normal(0.25, 0.08, n_water),
        'B4_Red': np.random.normal(0.12, 0.04, n_water),
        'B8_NIR': np.random.normal(0.08, 0.03, n_water),  # Low NIR for water
        'B11_SWIR1': np.random.normal(0.05, 0.02, n_water),  # Low SWIR for water
        'B12_SWIR2': np.random.normal(0.04, 0.02, n_water),
    }
    
    # Non-water surface characteristics (vegetation, soil, urban)
    non_water_data = {
        'B2_Blue': np.random.normal(0.20, 0.08, n_non_water),
        'B3_Green': np.random.normal(0.30, 0.10, n_non_water),
        'B4_Red': np.random.normal(0.35, 0.12, n_non_water),
        'B8_NIR': np.random.normal(0.45, 0.15, n_non_water),  # High NIR for vegetation
        'B11_SWIR1': np.random.normal(0.30, 0.10, n_non_water),  # Higher SWIR
        'B12_SWIR2': np.random.normal(0.25, 0.08, n_non_water),
    }
    
    # Combine data
    all_data = {}
    for key in water_data.keys():
        all_data[key] = np.concatenate([water_data[key], non_water_data[key]])
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Ensure values are in valid range [0, 1]
    for col in df.columns:
        df[col] = np.clip(df[col], 0.01, 0.99)
    
    # Calculate water indices
    fe = FeatureEngineer()
    df['NDWI'] = fe.calculate_ndwi(df['B3_Green'].values, df['B8_NIR'].values)
    df['MNDWI'] = fe.calculate_mndwi(df['B3_Green'].values, df['B11_SWIR1'].values)
    df['NDVI'] = fe.calculate_ndvi(df['B4_Red'].values, df['B8_NIR'].values)
    df['EVI'] = fe.calculate_evi(df['B2_Blue'].values, df['B4_Red'].values, df['B8_NIR'].values)
    df['NIR_Red_Ratio'] = df['B8_NIR'] / (df['B4_Red'] + np.finfo(float).eps)
    df['SWIR_NIR_Ratio'] = df['B11_SWIR1'] / (df['B8_NIR'] + np.finfo(float).eps)
    
    # Create target: water surface probability (0-1)
    # Higher NDWI and MNDWI indicate water
    water_probability = (
        0.4 * (df['NDWI'] + 1) / 2 +  # Normalize NDWI to [0, 1]
        0.4 * (df['MNDWI'] + 1) / 2 +  # Normalize MNDWI to [0, 1]
        0.2 * (1 - df['NDVI'])  # Low NDVI indicates water
    )
    water_probability = np.clip(water_probability, 0.0, 1.0)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, len(df))
    water_probability = np.clip(water_probability + noise, 0.0, 1.0)
    
    df['Water_Surface_Probability'] = water_probability
    
    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df


def main():
    """Main execution function."""
    print("=" * 80)
    print("Water Surface Classification in Kazakhstan - Assignment 1")
    print("=" * 80)
    print()
    
    # Set random seeds for reproducibility
    random_state = 42
    np.random.seed(random_state)
    
    # Create output directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Step 1: Generate or load dataset
    print("Step 1: Loading dataset...")
    data_file = 'data/water_surface_dataset.csv'
    
    if os.path.exists(data_file):
        print(f"Loading dataset from {data_file}")
        preprocessor = DataPreprocessor(random_state=random_state)
        data = preprocessor.load_data(data_file)
    else:
        print("Generating synthetic satellite imagery dataset...")
        data = generate_synthetic_dataset(n_samples=5000, random_state=random_state)
        data.to_csv(data_file, index=False)
        print(f"Dataset saved to {data_file}")
    
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {list(data.columns)}")
    print()
    
    # Step 2: Feature engineering
    print("Step 2: Feature engineering...")
    fe = FeatureEngineer()
    data = fe.create_features(data)
    print(f"Enhanced dataset shape: {data.shape}")
    print()
    
    # Step 3: Preprocessing
    print("Step 3: Preprocessing data...")
    preprocessor = DataPreprocessor(random_state=random_state)
    X, y = preprocessor.split_features_target(data, 'Water_Surface_Probability')
    
    # Scale features
    X_scaled = preprocessor.scale_features(X, fit=True)
    
    print(f"Features shape: {X_scaled.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of features: {X_scaled.shape[1]}")
    print()
    
    # Step 4: Initialize models
    print("Step 4: Initializing models...")
    model_factory = ModelFactory(random_state=random_state)
    models = model_factory.get_all_models()
    print(f"Initialized {len(models)} models:")
    for name in models.keys():
        print(f"  - {name}")
    print()
    
    # Step 5: Evaluate models
    print("Step 5: Evaluating models with 10-fold cross-validation...")
    print("-" * 80)
    
    evaluator = ModelEvaluator(cv_folds=10, random_state=random_state)
    fitted_models = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        try:
            result = evaluator.evaluate_model(model, X_scaled, y.values, model_name)
            fitted_models[model_name] = model
            print(f"  RMSE: {result['mean_rmse']:.4f} ± {result['std_rmse']:.4f}")
            print(f"  R²:   {result['mean_r2']:.4f} ± {result['std_r2']:.4f}")
        except Exception as e:
            print(f"  Error evaluating {model_name}: {e}")
        print()
    
    # Step 6: Generate results table
    print("Step 6: Generating results table...")
    results_df = evaluator.get_results_dataframe()
    results_df.to_csv('results/table1_results.csv', index=False)
    print("Results saved to results/table1_results.csv")
    print()
    print("Table 1: Model Performance Summary")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print()
    
    # Step 7: Generate visualizations
    print("Step 7: Generating visualizations...")
    evaluator.plot_model_comparison('figures/model_comparison.png')
    evaluator.plot_cv_performance('figures/cv_performance.png')
    evaluator.plot_feature_importance(fitted_models, X_scaled, 'figures/feature_importance.png')
    print()
    
    # Step 8: Summary
    print("=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Total models evaluated: {len(evaluator.results)}")
    print(f"Results saved to: results/table1_results.csv")
    print(f"Figures saved to: figures/")
    print()
    
    # Find best model
    if evaluator.results:
        best_model = min(evaluator.results, key=lambda x: x['mean_rmse'])
        print(f"Best model (lowest RMSE): {best_model['model_name']}")
        print(f"  RMSE: {best_model['mean_rmse']:.4f}")
        print(f"  R²:   {best_model['mean_r2']:.4f}")
    
    return results_df


if __name__ == "__main__":
    results = main()

