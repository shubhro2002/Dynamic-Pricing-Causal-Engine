import sys
import os
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthesizer import CausalDataSynthesizer
from src.data.validator import PreTreatmentValidator
from core.config import settings

def main():
    logger.info("--- Starting Phase 1: Data Synthesis & Validation ---")
    
    # 1. Synthesize Data
    synthesizer = CausalDataSynthesizer(n_samples=50000, random_seed=settings.random_seed)
    df, true_ate = synthesizer.generate_data()
    
    # Optional: Save to disk for Phase 2
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/synthetic_causal_data.csv", index=False)
    logger.info("Saved synthetic dataset to data/processed/synthetic_causal_data.csv")
    
    # 2. Validate Statistical Properties
    validator = PreTreatmentValidator(df)
    
    bias_results = validator.verify_confounding_bias()
    mde = validator.calculate_continuous_power()
    gps_results = validator.check_generalized_propensity_support()
    
    logger.info("--- Phase 1 Complete ---")
    
    # Assertions to ensure our math holds up before we proceed to ML
    assert abs(bias_results['Selection_Bias']) > 0.5, "Data doesn't have enough confounding bias for a robust test!"
    assert gps_results['Support_Satisfied'], "Positivity assumption violated. No random variation in price."
    logger.success("All mathematical pre-conditions passed! Ready for DML.")

if __name__ == "__main__":
    main()