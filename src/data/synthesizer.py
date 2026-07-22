import numpy as np
import pandas as pd
from loguru import logger
from typing import Tuple

class CausalDataSynthesizer:
    """
    Generates a synthetically confounded e-commerce dataset for causal inference.
    Simulates continuous dynamic pricing assignment with both observed (X) 
    and unobserved (W) confounders.
    """
    def __init__(self, n_samples: int = 50000, random_seed: int = 42):
        self.n_samples = n_samples
        self.seed = random_seed
        self.rng = np.random.default_rng(self.seed)

    def generate_data(self) -> Tuple[pd.DataFrame, float]:
        logger.info(f"Generating synthetic causal dataset with {self.n_samples} samples.")
        
        # 1. Observed Covariates (X)
        # Loyalty tier [0, 1], continuous
        x_loyalty = self.rng.uniform(0, 1, self.n_samples)          
        # Historical Spend / LTV
        x_spend = self.rng.lognormal(mean=4.0, sigma=0.5, size=self.n_samples) 
        # Device: 1 = Desktop, 0 = Mobile
        x_device = self.rng.binomial(1, 0.4, self.n_samples)        
        
        # 2. Unobserved Confounder (W)
        # Macro market demand shock (e.g., inflation surge, hidden competitor sale)
        w_macro = self.rng.normal(0, 1, self.n_samples)
        
        # 3. Treatment Assignment Mechanism (T) - Continuous Price Deviation
        # Price is dynamically driven by observed features AND the hidden macro shock
        t_base = 0.0 
        t_bias_x = 5.0 * x_loyalty + 0.05 * x_spend - 2.0 * x_device
        t_bias_w = 4.0 * w_macro
        t_noise = self.rng.normal(0, 2.0, self.n_samples)
        
        treatment_price = 100 + t_base + t_bias_x + t_bias_w + t_noise
        
        # 4. True Conditional Average Treatment Effect (CATE)
        # Heterogeneous Price Elasticity: 
        # Higher loyalty customers are less price sensitive (less negative elasticity)
        cate = -2.5 + 1.5 * x_loyalty - 0.5 * x_device
        
        # 5. Outcome Generating Mechanism (Y) - Continuous Revenue/Demand
        y_base = 200 + 0.8 * x_spend + 30.0 * x_loyalty
        y_confounded = 25.0 * w_macro
        y_noise = self.rng.normal(0, 10.0, self.n_samples)
        
        # Y(T) = Y_baseline + CATE * T + Noise
        outcome_revenue = y_base + y_confounded + (treatment_price * cate) + y_noise
        
        # 6. Calculate Analytical True ATE for validation
        # E[x_loyalty] = 0.5, E[x_device] = 0.4
        # ATE = -2.5 + 1.5*(0.5) - 0.5*(0.4) = -1.95
        true_ate = -2.5 + 1.5 * 0.5 - 0.5 * 0.4
        
        df = pd.DataFrame({
            'x_loyalty': x_loyalty,
            'x_spend': x_spend,
            'x_device': x_device,
            'w_macro': w_macro, # Kept for validation, MUST be dropped during DML training
            'treatment_price': treatment_price,
            'true_cate': cate,
            'outcome_revenue': outcome_revenue
        })
        
        logger.success(f"Data generation complete. Analytical True ATE: {true_ate:.4f}")
        return df, true_ate