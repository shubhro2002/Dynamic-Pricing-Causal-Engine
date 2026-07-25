import sys
import os
import pandas as pd
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.causal.dml_engine import DoubleMachineLearningEngine
from src.data.privacy import DifferentialPrivacyLayer

def main():
    logger.info("--- Starting Phase 4: Optimized Differential Privacy ---")
    
    data_path = "data/processed/synthetic_causal_data.csv"
    if not os.path.exists(data_path):
        logger.error("Data file not found. Run Phase 1 first.")
        return
        
    df_raw = pd.read_csv(data_path)
    TRUE_ATE = -1.95
    
    # DP CONFIGURATION
    dp_configs = {
        'x_loyalty': {
            'bounds': (0.0, 1.0),       # Loyalty is naturally bounded 0 to 1
            'budget_share': 0.2,        # Takes 20% of the epsilon budget
            'use_log_transform': False
        },
        'x_spend': {
            'bounds': (0.0, 100.0),     # We aggressively clip extreme spend outliers to drop sensitivity
            'budget_share': 0.8,        # Takes 80% of budget (needs more budget because variance is high)
            'use_log_transform': True   # Compresses the domain to log(100) before noise injection
        }
    }
    
    all_confounders = ['x_loyalty', 'x_spend', 'x_device', 'w_macro']
    cate_features = ['x_loyalty', 'x_device']
    
    privacy_layer = DifferentialPrivacyLayer(random_seed=42)
    epsilon_budgets = [0.1, 1.0, 10.0]
    
    logger.info(f"Target True ATE: {TRUE_ATE}")
    
    for eps in epsilon_budgets:
        logger.info(f"\n>> Testing Optimized Privacy Budget (\u03B5 = {eps}) <<")
        
        # Apply Advanced DP
        df_private = privacy_layer.apply_dp_to_dataset(df_raw, epsilon=eps, feature_configs=dp_configs)
        
        # Train DML
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
        logger.warning(f"[\u03B5={eps}] DP Estimate: {ate_est:.4f}")
        logger.warning(f"[\u03B5={eps}] Absolute Error from True ATE: {error:.4f}")

    logger.info("\n--- Phase 4 Complete ---")

if __name__ == "__main__":
    main()