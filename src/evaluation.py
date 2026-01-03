"""
Evaluation module for model performance assessment.
Implements cross-validation and metric calculation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Handles model evaluation and metrics calculation."""
    
    def __init__(self, cv_folds=10, random_state=42):
        """
        Initialize evaluator.
        
        Parameters:
        -----------
        cv_folds : int
            Number of cross-validation folds
        random_state : int
            Random seed for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        self.results = []
    
    def calculate_rmse(self, y_true, y_pred):
        """
        Calculate Root Mean Square Error.
        
        Parameters:
        -----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted target values
            
        Returns:
        --------
        float
            RMSE value
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def calculate_r2(self, y_true, y_pred):
        """
        Calculate R² score.
        
        Parameters:
        -----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted target values
            
        Returns:
        --------
        float
            R² score
        """
        return r2_score(y_true, y_pred)
    
    def evaluate_model(self, model, X, y, model_name):
        """
        Evaluate model using k-fold cross-validation.
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to evaluate
        X : array-like
            Features
        y : array-like
            Target
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        # Create scorers
        rmse_scorer = make_scorer(
            lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            greater_is_better=False
        )
        r2_scorer = make_scorer(r2_score)
        
        # Perform cross-validation
        cv_rmse_scores = -cross_val_score(
            model, X, y, 
            cv=self.kfold, 
            scoring=rmse_scorer, 
            n_jobs=-1
        )
        cv_r2_scores = cross_val_score(
            model, X, y, 
            cv=self.kfold, 
            scoring=r2_scorer, 
            n_jobs=-1
        )
        
        # Fit model on full data for feature count
        model.fit(X, y)
        
        # Calculate mean and std
        mean_rmse = np.mean(cv_rmse_scores)
        std_rmse = np.std(cv_rmse_scores)
        mean_r2 = np.mean(cv_r2_scores)
        std_r2 = np.std(cv_r2_scores)
        
        # Store results
        result = {
            'model_name': model_name,
            'n_features': X.shape[1],
            'n_targets': 1 if len(y.shape) == 1 else y.shape[1],
            'cv_folds': self.cv_folds,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'cv_rmse_scores': cv_rmse_scores,
            'cv_r2_scores': cv_r2_scores
        }
        
        self.results.append(result)
        return result
    
    def get_results_dataframe(self):
        """
        Get results as a pandas DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            Results summary
        """
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Algorithm': result['model_name'],
                'Number of Features': result['n_features'],
                'Number of Targets': result['n_targets'],
                'k-fold Validation': result['cv_folds'],
                'RMSE (Mean ± Std)': f"{result['mean_rmse']:.4f} ± {result['std_rmse']:.4f}",
                'R² (Mean ± Std)': f"{result['mean_r2']:.4f} ± {result['std_r2']:.4f}",
                'Mean RMSE': result['mean_rmse'],
                'Mean R²': result['mean_r2']
            })
        
        return pd.DataFrame(data)
    
    def plot_model_comparison(self, save_path='figures/model_comparison.png'):
        """
        Create model comparison plots.
        
        Parameters:
        -----------
        save_path : str
            Path to save the figure
        """
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Extract data
        model_names = [r['model_name'] for r in self.results]
        rmse_means = [r['mean_rmse'] for r in self.results]
        r2_means = [r['mean_r2'] for r in self.results]
        
        # RMSE plot
        axes[0].barh(model_names, rmse_means, color='steelblue')
        axes[0].set_xlabel('RMSE', fontsize=12)
        axes[0].set_title('Model Comparison: RMSE', fontsize=14, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # R² plot
        axes[1].barh(model_names, r2_means, color='coral')
        axes[1].set_xlabel('R² Score', fontsize=12)
        axes[1].set_title('Model Comparison: R² Score', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Model comparison plot saved to {save_path}")
    
    def plot_cv_performance(self, save_path='figures/cv_performance.png'):
        """
        Create cross-validation performance plots.
        
        Parameters:
        -----------
        save_path : str
            Path to save the figure
        """
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Prepare data for box plots
        rmse_data = []
        r2_data = []
        model_labels = []
        
        for result in self.results:
            rmse_data.append(result['cv_rmse_scores'])
            r2_data.append(result['cv_r2_scores'])
            model_labels.append(result['model_name'])
        
        # RMSE box plot
        bp1 = axes[0].boxplot(rmse_data, labels=model_labels, vert=True, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
        axes[0].set_ylabel('RMSE', fontsize=12)
        axes[0].set_title('Cross-Validation RMSE Distribution (k=10)', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # R² box plot
        bp2 = axes[1].boxplot(r2_data, labels=model_labels, vert=True, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightcoral')
        axes[1].set_ylabel('R² Score', fontsize=12)
        axes[1].set_title('Cross-Validation R² Distribution (k=10)', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"CV performance plot saved to {save_path}")
    
    def plot_feature_importance(self, models_dict, X, save_path='figures/feature_importance.png'):
        """
        Plot feature importance for tree-based models.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of fitted models
        X : array-like
            Feature array (for feature names)
        save_path : str
            Path to save the figure
        """
        tree_models = ['Extra Trees', 'AdaBoost', 'Gradient Boosting', 
                      'XGBoost', 'LightGBM', 'CatBoost', 'HistGradientBoosting']
        
        available_models = {k: v for k, v in models_dict.items() if k in tree_models}
        
        if not available_models:
            print("No tree-based models available for feature importance")
            return
        
        n_models = len(available_models)
        fig, axes = plt.subplots((n_models + 2) // 3, 3, figsize=(18, 6 * ((n_models + 2) // 3)))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (name, model) in enumerate(available_models.items()):
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = [f'Feature {i+1}' for i in range(len(importances))]
                    
                    # Get top 10 features
                    top_indices = np.argsort(importances)[-10:][::-1]
                    top_importances = importances[top_indices]
                    top_features = [feature_names[i] for i in top_indices]
                    
                    axes[idx].barh(top_features, top_importances, color='teal')
                    axes[idx].set_xlabel('Importance', fontsize=10)
                    axes[idx].set_title(f'{name} - Top 10 Features', fontsize=11, fontweight='bold')
                    axes[idx].grid(axis='x', alpha=0.3)
            except Exception as e:
                print(f"Could not plot feature importance for {name}: {e}")
        
        # Hide unused subplots
        for idx in range(len(available_models), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to {save_path}")

