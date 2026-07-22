import sys
import os
import pandas as pd
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.causal.dml_engine import DoubleMachineLearningEngine

def main():
    logger.info("--- Starting Phase 2: Double Machine Learning ---")
    
    data_path = "data/processed/synthetic_causal_data.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}. Please run Phase 1 first.")
        return
        
    df = pd.read_csv(data_path)
    
    # We generated the True ATE analytically in Phase 1 as -1.95
    TRUE_ATE = -1.95
    logger.info(f"Target True ATE to recover: {TRUE_ATE}")
    
    # 1. Partial Confounders (Real-world scenario where macro shocks are unobserved)
    partial_confounders = ['x_loyalty', 'x_spend', 'x_device']
    
    # 2. All Confounders (Oracle scenario fulfilling the Unconfoundedness assumption)
    all_confounders = ['x_loyalty', 'x_spend', 'x_device', 'w_macro']
    
    # Features we want to personalize pricing on
    cate_features = ['x_loyalty', 'x_device']

    logger.info(">>> EXPERIMENT 1: DML with Missing Confounder (w_macro omitted) <<<")
    dml_partial = DoubleMachineLearningEngine()
    dml_partial.fit(
        df=df,
        treatment_col='treatment_price',
        outcome_col='outcome_revenue',
        confounders=partial_confounders,
        cate_features=cate_features
    )
    res_partial = dml_partial.get_ate_summary()
    logger.warning(f"DML Estimate (Missing Confounders): {res_partial['ATE_Estimate']}")
    
    logger.info(">>> EXPERIMENT 2: DML with Full Confounders (Oracle) <<<")
    dml_full = DoubleMachineLearningEngine()
    dml_full.fit(
        df=df,
        treatment_col='treatment_price',
        outcome_col='outcome_revenue',
        confounders=all_confounders,
        cate_features=cate_features
    )
    res_full = dml_full.get_ate_summary()
    logger.success(f"DML Estimate (All Confounders): {res_full['ATE_Estimate']} (95% CI: [{res_full['CI_Lower']}, {res_full['CI_Upper']}])")
    
    logger.info(">>> Validating Heterogeneous Elasticity (CATE) <<<")
    
    # Let's test a High Loyalty Desktop user vs Low Loyalty Mobile user
    test_profiles = pd.DataFrame({
        'x_loyalty': [1.0, 0.0],
        'x_device': [1.0, 0.0]
    })
    
    # True expected elasticity:
    # Profile 0: -2.5 + 1.5(1) - 0.5(1) = -1.5
    # Profile 1: -2.5 + 1.5(0) - 0.5(0) = -2.5
    predicted_cates = dml_full.predict_cate(test_profiles)
    
    logger.info(f"Profile 0 (High Loyalty, Desktop) - True: -1.5, Predicted: {predicted_cates[0]:.4f}")
    logger.info(f"Profile 1 (Low Loyalty, Mobile)   - True: -2.5, Predicted: {predicted_cates[1]:.4f}")

    model_path = "data/processed/dml_oracle_model.joblib"
    dml_full.save(model_path)
    
    logger.info("--- Phase 2 Complete ---")

if __name__ == "__main__":
    main()