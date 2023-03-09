import logging
import sys
import time
from logging.handlers import TimedRotatingFileHandler
from logging.handlers import RotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_FILE = time.strftime("%Y%m%d-%H%M%S") + "test.log"


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    # file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=500 * 1024 * 1024, backupCount=50)
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.DEBUG)  # better to have too much log than not enough

    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())

    return logger
