""" Log generation """
import logging

logging.basicConfig(level=logging.INFO)


class SimpleLogger:
    """ Simple Logger Helper """

    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.name = name

    def info(self, msg):
        """ Info Log generation """
        self.logger.info(msg)

    def warning(self, msg):
        """ Warning Log generation """
        self.logger.warning(msg)

    def error(self, msg):
        """ Error Log generation """
        self.logger.error(msg)
