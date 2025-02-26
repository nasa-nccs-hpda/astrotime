import numpy as np
import torch
from functools import wraps
from time import time
import threading, time, sys, traceback
Array = torch.Tensor | np.ndarray
import logging
log = logging.getLogger("astrotime")

def shp(x): return list(x.shape)

def hasNaN(x: Array) -> bool:
    return np.isnan(x).any() if type(x) is np.ndarray else torch.isnan(x).any()

def exception_handled(func):
    def wrapper( *args, **kwargs ):
        try:
            return func( *args, **kwargs )
        except:
            log.error( f" Error in {func}:" )
            traceback.print_exc()
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
    return wrap
