import numpy as np
import torch
from typing import List, Optional, Dict, Type
import os, datetime
from .logger import PythonLogger
from functools import wraps
from time import time
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import threading, time, logging, sys, traceback
Array = torch.Tensor | np.ndarray

def lgm() -> "LogManager":
    return LogManager.instance()

def shp(x): return list(x.shape)

def hasNaN(x: Array) -> bool:
    return np.isnan(x).any() if type(x) is np.ndarray else torch.isnan(x).any()

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

class LogManager(object):
    _instance: "LogManager" = None

    def __init__(self):
        super(LogManager, self).__init__()
        self._lid = None
        self._level = logging.INFO
        self._logger: PythonLogger = None
        self.log_file  = None

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
        self._level = level

    def init_logging(self, log_dir: str, version: str, overwrite_log):
        self._lid = "" if overwrite_log else f"-{os.getpid()}"
        self.log_file = f'{log_dir}/{version}{self._lid}.log'
        os.makedirs( os.path.dirname( self.log_file ), mode=0o777, exist_ok=True )
        self._logger = PythonLogger("main")
        self._logger.file_logging( self.log_file )
        print( f"\n     **** Logging to file: {self.log_file} ****   \n")

    @property
    def ctime(self):
        return datetime.now().strftime("%H:%M:%S")

    def console(self, msg: str, end="\n"):
        print( msg, flush=True, end=end)

    def log( self,  msg, display=False, end="\n" ):
        self.info(msg,display,end)

    def info( self,  msg, display=False, end="\n" ):
        self._logger.log(msg)
        if display: self.console( msg, end=end )

    def fatal(self, msg: str, status: int = 1 ):
        self.console( msg )
        self._logger.error(msg)
        sys.exit( status )

    def debug(self, msg ):
        if self._level == logging.DEBUG:
            self.log( msg )

    def exception(self,  msg ):
        error_msg = f"\n{msg}\n{traceback.format_exc()}\n"
        self._logger.error(error_msg)
        self.console( error_msg )


    def trace(self,  msg ):
        strace = "".join(traceback.format_stack())
        self._logger.error(f"\n{msg}\n{strace}\n")