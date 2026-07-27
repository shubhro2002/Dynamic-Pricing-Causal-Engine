from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Centralized configuration management.
    Pydantic automatically reads these from the .env file and enforces type hints.
    """
    app_name: str = "Dynamic Pricing Causal Engine"
    app_env: str = "development"
    api_host: str = "127.0.0.1"
    app_port: int = 8000
    
    # ML & Causal Parameters
    random_seed: int = 42
    dml_cv_folds: int = 5
    
    # Privacy Parameters
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5

    # Look for a .env file in the root directory
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Instantiate a global settings object to be imported across the project
settings = Settings()