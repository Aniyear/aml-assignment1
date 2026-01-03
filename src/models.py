"""
Machine learning models module.
Implements all required algorithms for water surface classification.
"""

import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


class ModelFactory:
    """Factory class for creating ML models."""
    
    def __init__(self, random_state=42):
        """
        Initialize model factory.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
    
    def create_ridge_regression(self, alpha=1.0):
        """Create Ridge Regression model."""
        return Ridge(alpha=alpha, random_state=self.random_state, max_iter=1000)
    
    def create_lasso_regression(self, alpha=1.0):
        """Create Lasso Regression model."""
        return Lasso(alpha=alpha, random_state=self.random_state, max_iter=1000)
    
    def create_elastic_net(self, alpha=1.0, l1_ratio=0.5):
        """Create Elastic Net model."""
        return ElasticNet(
            alpha=alpha, 
            l1_ratio=l1_ratio, 
            random_state=self.random_state, 
            max_iter=1000
        )
    
    def create_knn_regression(self, n_neighbors=5):
        """Create k-Nearest Neighbors Regression model."""
        return KNeighborsRegressor(n_neighbors=n_neighbors)
    
    def create_extra_trees(self, n_estimators=100):
        """Create Extra Trees Regression model."""
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def create_adaboost(self, n_estimators=50):
        """Create AdaBoost Regression model."""
        return AdaBoostRegressor(
            n_estimators=n_estimators,
            random_state=self.random_state
        )
    
    def create_gradient_boosting(self, n_estimators=100):
        """Create Gradient Boosting Regression model."""
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            random_state=self.random_state,
            max_depth=3
        )
    
    def create_xgboost(self, n_estimators=100):
        """Create XGBoost Regression model."""
        return XGBRegressor(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
    
    def create_lightgbm(self, n_estimators=100):
        """Create LightGBM Regression model."""
        return LGBMRegressor(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1
        )
    
    def create_catboost(self, n_estimators=100):
        """Create CatBoost Regression model."""
        return CatBoostRegressor(
            n_estimators=n_estimators,
            random_seed=self.random_state,
            verbose=False
        )
    
    def create_hist_gradient_boosting(self, max_iter=100):
        """Create HistGradientBoosting Regression model."""
        return HistGradientBoostingRegressor(
            max_iter=max_iter,
            random_state=self.random_state
        )
    
    def get_all_models(self):
        """
        Get all models configured with default hyperparameters.
        
        Returns:
        --------
        dict
            Dictionary of model names and instances
        """
        models = {
            'Ridge Regression': self.create_ridge_regression(alpha=1.0),
            'Lasso Regression': self.create_lasso_regression(alpha=1.0),
            'Elastic Net': self.create_elastic_net(alpha=1.0, l1_ratio=0.5),
            'KNN Regression': self.create_knn_regression(n_neighbors=5),
            'Extra Trees': self.create_extra_trees(n_estimators=100),
            'AdaBoost': self.create_adaboost(n_estimators=50),
            'Gradient Boosting': self.create_gradient_boosting(n_estimators=100),
            'XGBoost': self.create_xgboost(n_estimators=100),
            'LightGBM': self.create_lightgbm(n_estimators=100),
            'CatBoost': self.create_catboost(n_estimators=100),
            'HistGradientBoosting': self.create_hist_gradient_boosting(max_iter=100)
        }
        return models

