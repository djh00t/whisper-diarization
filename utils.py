import logging
import sys

def setup_logger(debug=False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Set the logging level for 'urllib3.connectionpool' to WARNING
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    if debug:
        logger.debug("Debug logging enabled")
    return logger
