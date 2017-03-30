import logging
import os

_loggers = {}

logging.basicConfig(
    format='%(levelname)s %(asctime)s %(message)s', level=logging.DEBUG)


def get_logger(logger_name, logdir, level=logging.INFO):
    if logger_name in _loggers:
        return _loggers[logger_name]
    else:
        log_file = os.path.join(logdir, logger_name)
        logger = setup_logger(logger_name, log_file, level)
        _loggers[logger_name] = logger
        return logger


def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    file_handler = logging.FileHandler(log_file, mode='w')
    logger.addHandler(file_handler)
    return logger
