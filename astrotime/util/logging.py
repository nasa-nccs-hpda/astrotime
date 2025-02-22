import numpy as np
from typing import List, Optional, Dict, Type
import tensorflow as tf
import os, datetime
from functools import wraps
from time import time
from datetime import datetime
from termcolor import colored
import time, logging, sys, traceback
Array =  np.ndarray

def lgm() -> "LogManager":
    return LogManager.instance()

def shp(x): return list(x.shape)

def hasNaN(x: Array) -> bool:
    return np.isnan(x).any() if type(x) is np.ndarray else np.isnan(x).any()

def exception_handled(func):
    def wrapper( *args, **kwargs ):
        try:
            return func( *args, **kwargs )
        except:
            lgm().exception( f" Error in {func}:" )
            traceback.print_exc()
    return wrapper

def log_timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        try:
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()
            lgm().log( f'EXEC {f.__name__} took: {te-ts:3.4f} sec' )
            return result
        except:
            lgm().exception( f" Error in {f}:" )
    return wrap

class PythonLogger:
    """Simple console logger for DL training
    This is a WIP
    """

    def __init__(self, console: bool = False ):
        self.logger = tf.get_logger()
        self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)
        self.filehandler = None
        formatter = logging.Formatter( "[%(asctime)s - %(name)s - %(levelname)s] %(message)s", datefmt="%H:%M:%S"  )
        if console:
            streamhandler = logging.StreamHandler()
            streamhandler.setFormatter(formatter)
            streamhandler.setLevel(logging.INFO)
            self.logger.addHandler(streamhandler)

    def file_logging(self, file_name: str = "launch.log"):
        """Log to file"""
        if os.path.exists(file_name):
            try:
                os.remove(file_name)
            except FileNotFoundError:
                # ignore if already removed (can happen with multiple processes)
                pass
        formatter = logging.Formatter(    "[%(asctime)s - %(name)s - %(levelname)s] %(message)s",  datefmt="%H:%M:%S" )
        self.filehandler = logging.FileHandler(file_name)
        self.filehandler.setFormatter(formatter)
        self.filehandler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.filehandler)

    def set_level(self, level ):
        self.logger.setLevel(level)

    def get_level(self):
        return self.logger.level

    def log(self, message: str):
        """Log message"""
        self.logger.info(message)
        self.filehandler.flush()

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

    def debug(self, message: str):
        """Log error"""
        self.logger.debug(colored(message, "light_red"))

class LogManager(object):
    _instance: "LogManager" = None

    def __init__(self):
        super(LogManager, self).__init__()
        self._lid = None
        self._level = logging.INFO
        self._logger: PythonLogger = None
        self.log_dir = None
        self.log_file  = None
        self.rank = 0

    @classmethod
    def instance(cls) -> "LogManager":
        if cls._instance is None:
            logger = LogManager()
            cls._instance = logger
        return cls._instance

    def close(self):
        self._logger = None

    @classmethod
    def pid(cls):
        return os.getpid()

    def set_level(self, level ):
        self._logger.set_level(level)

    def get_level(self ):
        return self._logger.get_level()

    def init_logging(self, log_dir: str, level, **kwargs ):
        self.log_dir =  log_dir # f"{cfg().platform.cache}/logs"
        overwrite = kwargs.get("overwrite_log", True)
        self._lid = "" if overwrite else f"-{os.getpid()}"
        self.log_file = f'{self.log_dir}/astrotime{self._lid}.log' # f'{self.log_dir}/{cid()}{self._lid}.log'
        os.makedirs( os.path.dirname( self.log_file ), mode=0o777, exist_ok=True )
        print( f"\n     Logging to file: '{self.log_file}'    \n\n", flush=True)
        self._logger = PythonLogger()
        self._logger.file_logging( self.log_file )
        self._logger.set_level( level )

    @property
    def ctime(self):
        return datetime.now().strftime("%H:%M:%S")

    def console(self, msg: str, end="\n"):
        print( msg, flush=True, end=end)

    def log( self,  msg, display=False, end="\n" ):
        if self.rank < 1:
            self.info(msg,display,end)

    def info( self,  msg, display=False, end="\n" ):
        self._logger.log(msg)
        if display: self.console( msg, end=end )

    def fatal(self, msg: str, status: int = 1 ):
        self.console( msg )
        self._logger.error(msg)
        sys.exit( status )

    def debug(self, msg ):
        self._logger.debug(msg)

    def exception(self,  msg ):
        error_msg = f"\n{msg}\n{traceback.format_exc()}\n"
        self._logger.error(error_msg)
        self.console( error_msg )

    def trace(self,  msg ):
        strace = "".join(traceback.format_stack())
        self._logger.error(f"\n{msg}\n{strace}\n")