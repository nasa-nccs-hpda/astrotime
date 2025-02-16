import numpy as np
import tensorflow as tf
from typing import List, Optional, Dict, Type, Any

def logspace(start: float, stop: float, N: int) -> np.ndarray:
	return np.pow(10.0, np.linspace(np.log10(start), np.log10(stop), N))

def shp( x ) -> List[int]:
	return list(x.shape)

def tmean(x: tf.Tensor) -> float:
	xm: tf.Tensor = tf.math.reduce_mean(x)
	return tf.squeeze(xm).numpy()

def tstd(x: tf.Tensor) -> float:
	xs: tf.Tensor = tf.math.reduce_std(x)
	return tf.squeeze(xs).numpy()