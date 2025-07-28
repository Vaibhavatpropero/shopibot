import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter(
    r"%(asctime)s - %(levelname)-7s %(threadName)-12s [%(filename)s:%(lineno)s - %(funcName)s()] - %(message)s"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
