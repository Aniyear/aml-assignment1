"""
Data preprocessing module for water surface classification.
Handles data loading, cleaning, and preparation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Handles data preprocessing operations."""
    
    def __init__(self, random_state=42):
        """
        Initialize preprocessor.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        
    def load_data(self, filepath):
        """
        Load dataset from file.
        
        Parameters:
        -----------
        filepath : str
            Path to data file
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataset
        """
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.parquet'):
            return pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
    
    def split_features_target(self, data, target_column):
        """
        Split dataset into features and target.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        target_column : str
            Name of target column
            
        Returns:
        --------
        tuple
            (X, y) features and target arrays
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return X, y
    
    def scale_features(self, X_train, X_test=None, fit=True):
        """
        Scale features using StandardScaler.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        X_test : array-like, optional
            Test features
        fit : bool
            Whether to fit scaler on training data
            
        Returns:
        --------
        tuple
            Scaled features
        """
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)
            
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def train_test_split(self, X, y, test_size=0.2):
        """
        Split data into train and test sets.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target
        test_size : float
            Proportion of test set
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            shuffle=True
        )

