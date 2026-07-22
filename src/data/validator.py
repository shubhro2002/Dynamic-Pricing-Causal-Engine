import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from loguru import logger
from typing import Dict, Any

class PreTreatmentValidator:
    """
    Executes pre-treatment statistical validation, power analysis, 
    and common support verification for continuous causal inference.
    """
    def __init__(self, df: pd.DataFrame, random_seed: int = 42):
        self.df = df
        self.rng = np.random.default_rng(random_seed)
        
    def verify_confounding_bias(self) -> Dict[str, float]:
        logger.info("Executing Confounding Bias Verification...")
        
        # 1. Naïve OLS: Y ~ T 
        # Biased because it ignores X and W confounding the price
        X_naive = sm.add_constant(self.df['treatment_price'])
        model_naive = sm.OLS(self.df['outcome_revenue'], X_naive).fit()
        naive_ate = model_naive.params['treatment_price']
        
        # 2. Oracle OLS: Y ~ T + X + W 
        # Controls for everything, including the unobserved W (which we wouldn't have in real life)
        X_oracle = sm.add_constant(self.df[['treatment_price', 'x_loyalty', 'x_spend', 'x_device', 'w_macro']])
        model_oracle = sm.OLS(self.df['outcome_revenue'], X_oracle).fit()
        oracle_ate = model_oracle.params['treatment_price']
        
        true_ate = self.df['true_cate'].mean()
        
        results = {
            'True_ATE': round(true_ate, 4),
            'Naive_ATE_Estimate': round(naive_ate, 4),
            'Oracle_ATE_Estimate': round(oracle_ate, 4),
            'Selection_Bias': round(naive_ate - true_ate, 4)
        }
        
        logger.info(f"Bias Verification Results: {results}")
        return results
        
    def calculate_continuous_power(self, alpha: float = 0.05, power: float = 0.8) -> float:
        """
        Calculates Minimum Detectable Effect (MDE) for a continuous treatment.
        """
        logger.info("Calculating Statistical Power (MDE)...")
        n = len(self.df)
        var_t = self.df['treatment_price'].var()
        var_y = self.df['outcome_revenue'].var()
        
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        # MDE approximation for continuous regression coefficient
        mde = (z_alpha + z_beta) * np.sqrt(var_y / (n * var_t))
        
        logger.info(f"Minimum Detectable Effect (MDE): {mde:.4f}")
        return round(mde, 4)

    def check_generalized_propensity_support(self) -> Dict[str, Any]:
        """
        Fits a Generalized Propensity Score model and checks residual variance
        to ensure the Positivity Assumption is not violated.
        """
        logger.info("Verifying Common Support via GPS...")
        X_cov = sm.add_constant(self.df[['x_loyalty', 'x_spend', 'x_device']])
        
        # Model treatment assignment mechanism: T ~ X
        gps_model = sm.OLS(self.df['treatment_price'], X_cov).fit()
        residuals = gps_model.resid
        resid_var = np.var(residuals)
        
        # Shapiro-Wilk test for normality of residuals
        # Using a sub-sample to respect scipy's computational limits for Shapiro
        sample_resids = self.rng.choice(residuals, 5000, replace=False)
        stat, p_val = stats.shapiro(sample_resids)
        
        # We need residual variance > 0 to ensure we actually have exogenous price variation
        # to exploit for causal inference.
        support_satisfied = bool(resid_var > 0.5)
        
        results = {
            'Treatment_R_squared': round(gps_model.rsquared, 4),
            'Residual_Variance': round(resid_var, 4),
            'Residual_Normality_P_Value': round(p_val, 4),
            'Support_Satisfied': support_satisfied
        }
        
        logger.info(f"GPS Support Results: {results}")
        return results