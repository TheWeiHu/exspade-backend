import logging
import boto3
import watchtower
from datetime import datetime
import json
import os


def setup_logger(log_group_name="/spade/backend", log_stream_name="api-calls"):
    """Set up a logger that writes to CloudWatch in prod, disabled otherwise"""
    logger = logging.getLogger("spade_backend")
    
    # Check if we're in production
    is_prod = os.environ.get('ENVIRONMENT') == 'prod'
    
    if not is_prod:
        # Disable logging completely in non-prod environments
        logger.setLevel(logging.CRITICAL)
        return logger

    # Production logging setup
    logger.setLevel(logging.INFO)
    logger.handlers = []

    try:
        # Get region from environment variable or use default
        region = os.environ.get('AWS_REGION', 'us-west-2')
        
        # Configure boto3 client with region
        boto3.setup_default_session(region_name=region)
        
        cloudwatch_handler = watchtower.CloudWatchLogHandler(
            log_group=log_group_name,
            stream_name=log_stream_name,
            use_queues=True
        )
        cloudwatch_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(cloudwatch_handler)
        print(f"CloudWatch handler setup successful in region: {region}")
    except Exception as e:
        print(f"Failed to set up CloudWatch handler: {e}")

    return logger


def log_request(logger, data):
    """Log incoming request data"""
    logger.info(f"Request: {json.dumps(data)}")


def log_error(logger, error_msg, traceback_str=None):
    """Log error with optional traceback"""
    logger.error(error_msg)
    if traceback_str:
        logger.error(f"Traceback: {traceback_str}")


# Create default logger instance
logger = setup_logger()

if __name__ == "__main__":
    test_data = {
        "user_doc_format": "test",
        "user_question": "test question",
        "timestamp": datetime.utcnow().isoformat()
    }
    print("Testing request logging...")
    log_request(logger, test_data)

    # Test error logging
    print("Testing error logging...")
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_error(logger, str(e), "Test traceback")

    print("Logger test completed. Check CloudWatch logs.")
