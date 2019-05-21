import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s| %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger

def find_id():
    lo = 0
    hi = 1
    while os.path.exists(f'logs/avalanche-{hi}.log'):
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) >> 1
        if os.path.exists(f'logs/avalanche-{mid}.log'):
            lo = mid
        else:
            hi = mid
    return hi


def get_logger_avalanche():
    try:
        os.mkdir('logs')
    except:
        pass

    logger_avalanche = setup_logger('avalanche', f'logs/avalanche-{find_id()}.log')
    logger_avalanche.disabled = False
    return logger_avalanche
