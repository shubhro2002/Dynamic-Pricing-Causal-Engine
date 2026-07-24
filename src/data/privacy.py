import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict, Any, Union

class DifferentialPrivacyLayer:
    """
    Advanced Differential Privacy layer implementing Pre-Clipping, 
    Domain Compression (Log-Transform), and Weighted Budget Allocation.
    """
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)
        
    def add_laplace_noise(self, data: Union[pd.Series, np.ndarray], epsilon: float, sensitivity: float) -> Union[pd.Series, np.ndarray]:
        """Injects Laplace noise into a continuous variable."""
        scale = sensitivity / epsilon
        noise = self.rng.laplace(loc=0.0, scale=scale, size=len(data))
        return data + noise
        
    def apply_dp_to_dataset(self, df: pd.DataFrame, epsilon: float, feature_configs: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Creates a privacy-preserved copy of the dataset using advanced DP mechanics.
        """
        df_private = df.copy()
        
        for col, config in feature_configs.items():
            if col not in df_private.columns:
                continue
                
            lower_bound, upper_bound = config['bounds']
            budget_share = config['budget_share']
            use_log = config.get('use_log_transform', False)
            
            # Allocate specific epsilon for this feature
            feature_epsilon = epsilon * budget_share
            
            # 1. Pre-Clipping: Enforce bounds to limit sensitivity BEFORE DP
            processed_data = np.clip(df_private[col], lower_bound, upper_bound)
            
            # 2. Domain Compression
            if use_log:
                processed_data = np.log1p(processed_data) # log(1 + x)
                sensitivity = np.log1p(upper_bound) - np.log1p(lower_bound)
            else:
                sensitivity = upper_bound - lower_bound
                
            # 3. Apply Laplace Noise
            noisy_data = self.add_laplace_noise(processed_data, feature_epsilon, sensitivity)
            
            # Reverse Domain Compression for downstream model interpretability
            if use_log:
                # Expm1 reverses log1p. We also clip to avoid math overflows from extreme noise
                noisy_data = np.expm1(np.clip(noisy_data, 0, np.log1p(upper_bound) * 2))
                
            # Post-noise boundary enforcement (prevents impossible values like negative spend)
            df_private[col] = np.clip(noisy_data, lower_bound, None)
            
        return df_private