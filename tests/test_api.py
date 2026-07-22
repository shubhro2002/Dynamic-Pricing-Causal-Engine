import requests
import json
from loguru import logger

def test_prediction():
    url = "http://127.0.0.1:8000/predict_elasticity"
    
    # Test Payload mimicking a High-Loyalty Desktop user
    payload = {
        "customer_id": "CUST-9921",
        "x_loyalty": 0.95,
        "x_spend": 55.4,
        "x_device": 1
    }
    
    logger.info(f"Sending request to API: {payload}")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes
        
        data = response.json()
        logger.success(f"API Response Successful:\n{json.dumps(data, indent=2)}")
        logger.info(f"Recommended Action for {data['customer_id']}: {data['recommended_action']}")
        
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to the API. Is the server running (python scripts/run_api.py)?")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    test_prediction()