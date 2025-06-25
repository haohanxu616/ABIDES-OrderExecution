import logging

# Configure logging with debug level and ensure a handler exists
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.addHandler(handler)
logger.handlers[0].setFormatter(logging.Formatter('{asctime} - {levelname} - {message}', style='{'))