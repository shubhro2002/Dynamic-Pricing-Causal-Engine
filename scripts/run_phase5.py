import sys
import os
import pandas as pd
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.causal.dml_engine import DoubleMachineLearningEngine
from src.data.cohorting import KAnonymityCohorter

def main():
    logger.info("--- Starting Phase 5: The Fix (k-Anonymity Cohorting) ---")
    
    data_path = "data/processed/synthetic_causal_data.csv"
    if not os.path.exists(data_path):
        logger.error("Data file not found. Run Phase 1 first.")
        return
        
    df_raw = pd.read_csv(data_path)
    TRUE_ATE = -1.95
    
    # We only apply cohorting to our highly sensitive, continuous variables
    sensitive_features = ['x_loyalty', 'x_spend']
    all_confounders = ['x_loyalty', 'x_spend', 'x_device', 'w_macro']
    cate_features = ['x_loyalty', 'x_device']
    
    # Test different cohort sizes
    # k=1 (Raw Data, No Privacy)
    # k=50 (High Risk, Low Anonymity)
    # k=500 (Low Risk, High Anonymity)
    cohort_sizes = [50, 500]
    
    logger.info(f"Target True ATE: {TRUE_ATE}")
    
    for k in cohort_sizes:
        logger.info(f"\n>> Testing Micro-Cohorting (k = {k} users per bucket) <<")
        
        cohorter = KAnonymityCohorter(k=k)
        df_private = cohorter.apply_cohorting(df_raw, continuous_features=sensitive_features)
        
        # Train DML on the Cohorted Data
        dml_private = DoubleMachineLearningEngine(n_splits=3, random_state=42)
        dml_private.fit(
            df=df_private,
            treatment_col='treatment_price',
            outcome_col='outcome_revenue',
            confounders=all_confounders,
            cate_features=cate_features
        )
        
        res_private = dml_private.get_ate_summary()
        ate_est = res_private['ATE_Estimate']
        
        error = abs(TRUE_ATE - ate_est)
        logger.success(f"[k={k}] Cohorted ATE Estimate: {ate_est:.4f}")
        logger.info(f"[k={k}] Absolute Error from True ATE: {error:.4f}")

    logger.info("\n--- Phase 5 Complete ---")

if __name__ == "__main__":
    main()