import logging
import os
from termcolor import colored

class PythonLogger:
    """Simple console logger for DL training
    This is a WIP
    """

    def __init__(self, name: str = "launch", console: bool = False ):
        self.logger = logging.getLogger(name)
        self.logger.handlers.clear()
        formatter = logging.Formatter( "[%(asctime)s - %(name)s - %(levelname)s] %(message)s", datefmt="%H:%M:%S"  )
        if console:
            streamhandler = logging.StreamHandler()
            streamhandler.setFormatter(formatter)
            streamhandler.setLevel(logging.INFO)
            self.logger.addHandler(streamhandler)

        # Not sure if this works
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # Prevent parent logging

    def file_logging(self, file_name: str = "launch.log"):
        """Log to file"""
        if os.path.exists(file_name):
            try:
                os.remove(file_name)
            except FileNotFoundError:
                # ignore if already removed (can happen with multiple processes)
                pass
        formatter = logging.Formatter(    "[%(asctime)s - %(name)s - %(levelname)s] %(message)s",  datefmt="%H:%M:%S" )
        filehandler = logging.FileHandler(file_name)
        filehandler.setFormatter(formatter)
        filehandler.setLevel(logging.DEBUG)
        self.logger.addHandler(filehandler)

    def log(self, message: str):
        """Log message"""
        self.logger.info(message)

    def info(self, message: str):
        """Log info"""
        self.logger.info(colored(message, "light_blue"))

    def success(self, message: str):
        """Log success"""
        self.logger.info(colored(message, "light_green"))

    def warning(self, message: str):
        """Log warning"""
        self.logger.warning(colored(message, "light_yellow"))

    def error(self, message: str):
        """Log error"""
        self.logger.error(colored(message, "light_red"))


class RankZeroLoggingWrapper:
    """Wrapper class to only log from rank 0 process in distributed training."""

    def __init__(self, obj, dist):
        self.obj = obj
        self.dist = dist

    def __getattr__(self, name):
        attr = getattr(self.obj, name)
        if callable(attr):

            def wrapper(*args, **kwargs):
                if self.dist.rank == 0:
                    return attr(*args, **kwargs)
                else:
                    return None

            return wrapper
        else:
            return attr
