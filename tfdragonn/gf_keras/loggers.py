import logging
import os
import sys

_loggers = {}

logging.basicConfig(
    format='%(levelname)s %(asctime)s %(name)s: %(message)s', level=logging.DEBUG)


def get_logger(logger_name, logdir=None, stdout=True, level=logging.INFO):
    if logger_name in _loggers:
        return _loggers[logger_name]
    else:
        if logdir is None:
            log_file = None
        else:
            log_file = os.path.join(logdir, logger_name)
        logger = setup_logger(logger_name, log_file, level)
        _loggers[logger_name] = logger
        return logger


def add_logdir(logger_name, logdir, level=logging.INFO):
    logger = get_logger(logger_name)
    log_file = os.path.join(logdir, logger_name)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(level)
    logger.addHandler(file_handler)


def setup_logger(logger_name, log_file=None, stdout=True, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    if stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)
    return logger
