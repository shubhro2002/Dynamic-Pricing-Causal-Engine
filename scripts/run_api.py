import uvicorn
import os
import sys
from loguru import logger
from core.config import settings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    logger.info("Starting FastAPI Production Server for Causal Engine...")

    is_dev = settings.app_env.lower() == "development"
    
    # Run the uvicorn server, pointing to the app instance in src.api.main
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.app_port,
        reload=is_dev,  # Enables hot-reloading for development
        workers=1
    )