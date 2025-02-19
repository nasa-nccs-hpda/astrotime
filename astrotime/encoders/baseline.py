import random, keras, time, tensorflow as tf, numpy as np
from typing import Any, Dict, List, Optional, Tuple
from astrotime.encoders.base import Encoder
from astrotime.util.math import tmean, tstd, tmag, tnorm, shp

class ValueEncoder(Encoder):

	def __init__(self, device, series_len: int, max_series_len: int ):
		super(ValueEncoder, self).__init__( device, series_len )
		self.max_series_len = max_series_len

	def encode_dset(self, dset: Dict[str,np.ndarray]) -> Tuple[tf.Tensor,tf.Tensor]:
		with (self.device):
			y1, x1 = [], []
			for idx, (y,x) in enumerate(zip(dset['y'],dset['x'])):
				x,y = self.apply_filters(x,y,0)
				x0: int = tf.random.uniform([1], 0, self.max_series_len - self.series_len, dtype=tf.int32)[0]
				ys: tf.Tensor = tf.convert_to_tensor(y[x0:x0 + self.series_len], dtype=tf.float32)
				xs: tf.Tensor = tf.convert_to_tensor(x[x0:x0 + self.series_len], dtype=tf.float32)
				y1.append( tf.expand_dims( tnorm(ys,axis=0), 0) )
				x1.append( tf.expand_dims( xs, 0) )
			Y, X = tf.concat(y1, axis=0), tf.concat(x1, axis=0)
			if Y.ndim == 2: Y = tf.expand_dims(Y, axis=2)
			return X, Y

	def encode_batch(self, x: np.ndarray, y: np.ndarray ) -> Tuple[tf.Tensor,tf.Tensor]:
		with (self.device):
			x,y = self.apply_filters(x,y,1)
			x0: int = tf.random.uniform([1], 0, self.max_series_len - self.series_len, dtype=tf.int32)[0]
			Y: tf.Tensor = tf.convert_to_tensor(y[:,x0:x0 + self.series_len], dtype=tf.float32)
			X: tf.Tensor = tf.convert_to_tensor(x[:,x0:x0 + self.series_len], dtype=tf.float32)
			Y = tnorm(Y,axis=1)
			if Y.ndim == 2: Y = tf.expand_dims(Y, axis=2)
			return X, Y