import numpy as np
import torch
from functools import wraps
from time import time
import threading, time, sys, traceback
Array = torch.Tensor | np.ndarray
import logging
log = logging.getLogger("astrotime")

def shp(x): return list(x.shape)

def exception_handled(func):
    def wrapper( *args, **kwargs ):
        try:
            return func( *args, **kwargs )
        except:
            log.error( f" Error in {func}:" )
            log.error( traceback.format_exc(100) )
    return wrapper

def log_timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        try:
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()
            log.info( f'EXEC {f.__name__} took: {te-ts:3.4f} sec' )
            return result
        except:
            log.error( f" Error in {f}:" )
            log.error(traceback.format_exc(100))
    return wrap
