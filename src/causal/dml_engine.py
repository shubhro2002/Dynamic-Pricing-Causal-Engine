import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
from sklearn.model_selection import KFold
import joblib
import os
from loguru import logger
from typing import List, Dict, Any, Optional
from core.config import settings

class DoubleMachineLearningEngine:
    """
    Implements a Double Machine Learning (DML) pipeline with cross-fitting
    to estimate continuous Average Treatment Effects (ATE) and 
    Conditional Average Treatment Effects (CATE).
    """
    def __init__(self, n_splits: int = settings.dml_cv_folds, random_state: int = settings.random_seed):
        self.n_splits = n_splits
        self.random_state = random_state
        self.ate_model = None
        self.cate_model = None
        self.cate_features: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame, treatment_col: str, outcome_col: str, 
            confounders: List[str], cate_features: Optional[List[str]] = None):
        """
        Executes the Orthogonalization process using K-Fold Cross-Fitting.
        """
        logger.info(f"Starting DML Cross-Fitting with {self.n_splits} folds...")
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        y_residuals = np.zeros(len(df))
        t_residuals = np.zeros(len(df))
        
        self.cate_features = cate_features

        for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
            X_train, X_test = df.loc[train_idx, confounders], df.loc[test_idx, confounders]
            y_train, y_test = df.loc[train_idx, outcome_col], df.loc[test_idx, outcome_col]
            t_train, t_test = df.loc[train_idx, treatment_col], df.loc[test_idx, treatment_col]
            
            # Nuisance Model 1: E[Y | X] -> Predict Demand from Covariates
            model_y = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, 
                                        random_state=self.random_state, verbose=-1)
            model_y.fit(X_train, y_train)
            y_residuals[test_idx] = y_test.to_numpy() - model_y.predict(X_test)
            
            # Nuisance Model 2: E[T | X] -> Predict Price from Covariates
            model_t = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, 
                                        random_state=self.random_state, verbose=-1)
            model_t.fit(X_train, t_train)
            t_residuals[test_idx] = t_test.to_numpy() - model_t.predict(X_test)

        logger.info("Cross-fitting complete. Fitting Final Causal Stages...")

        # Regress Y_res on T_res
        t_res_const = sm.add_constant(t_residuals)
        self.ate_model = sm.OLS(y_residuals, t_res_const).fit()
        
        if self.cate_features:
            # We interact the treatment residuals with the CATE features
            X_cate_vals = df[self.cate_features].to_numpy(dtype=float)
            interaction_terms = X_cate_vals * t_residuals[:, None]
            
            # Matrix structure: [T_res, T_res * X1, T_res * X2, ...]
            X_final = np.column_stack((t_residuals, interaction_terms))
            
            self.cate_model = sm.OLS(y_residuals, X_final).fit()
            logger.info("CATE estimation complete.")

    def get_ate_summary(self) -> Dict[str, Any]:
        """Returns the isolated Average Treatment Effect."""
        if self.ate_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Extract estimates and cast to float to prevent API serialization issues
        ate_est = float(self.ate_model.params[1])
        p_val = float(self.ate_model.pvalues[1])
        ci_lower, ci_upper = self.ate_model.conf_int()[1]
        
        return {
            "ATE_Estimate": round(ate_est, 4),
            "P_Value": round(p_val, 5),
            "CI_Lower": round(float(ci_lower), 4),
            "CI_Upper": round(float(ci_upper), 4),
            "Significant": bool(p_val < 0.05)
        }

    def predict_cate(self, X_new: pd.DataFrame) -> np.ndarray:
        """Predicts the personalized price elasticity for new customers."""
        if self.cate_model is None or self.cate_features is None:
            raise ValueError("CATE model not fitted. Provide cate_features during fit().")
            
        X_vals = X_new[self.cate_features].to_numpy(dtype=float)
        
        # Base elasticity (beta_0) + interactions
        base_elasticity = self.cate_model.params[0]
        interaction_elasticities = np.dot(X_vals, self.cate_model.params[1:])
        
        return base_elasticity + interaction_elasticities

    def save(self, filepath: str):
        """Serializes the trained engine to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"DML Engine saved successfully to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "DoubleMachineLearningEngine":
        """Loads a serialized DML engine from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        logger.info(f"Loading DML Engine from {filepath}")
        return joblib.load(filepath)