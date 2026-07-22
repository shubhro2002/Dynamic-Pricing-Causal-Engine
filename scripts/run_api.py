import uvicorn
import os
import sys
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    logger.info("Starting FastAPI Production Server for Causal Engine...")
    
    # Run the uvicorn server, pointing to the app instance in src.api.main
    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Enables hot-reloading for development
        workers=1
    )