import pandas as pd
import numpy as np
from loguru import logger
from typing import List

class KAnonymityCohorter:
    """
    Implements k-Anonymity via Micro-Cohorting.
    Groups users into buckets of minimum size 'k' and masks their 
    individual features with the cohort's median value.
    """
    def __init__(self, k: int = 100):
        self.k = k

    def apply_cohorting(self, df: pd.DataFrame, continuous_features: List[str]) -> pd.DataFrame:
        """
        Applies quantile-based cohorting to continuous features.
        """
        logger.info(f"Applying k-Anonymity Cohorting (k={self.k}) to {continuous_features}")
        df_cohort = df.copy()
        
        for feat in continuous_features:
            # Calculate the number of bins to ensure each bin has roughly 'k' users
            num_bins = max(1, len(df) // self.k)
            
            # Use qcut to create quantile bins so they are evenly populated
            try:
                bin_labels = pd.qcut(df_cohort[feat], q=num_bins, duplicates='drop')
                
                # Replace individual raw values with the cohort's median value
                df_cohort[feat] = df_cohort.groupby(bin_labels)[feat].transform('median')
            except ValueError as e:
                logger.warning(f"Could not cleanly cohort feature {feat}: {e}")
                
        return df_cohort