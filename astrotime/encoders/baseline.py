import random, keras, time, tensorflow as tf, numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Any, Dict, List, Optional, Tuple
from astrotime.encoders.base import Encoder
from astrotime.util.math import tmean, tstd, tmag, tnorm

class ValueEncoder(Encoder):

	def __init__(self, device, series_len: int):
		super(ValueEncoder, self).__init__( device )
		self.series_len = series_len
		self.slmax = 6000

	def encode_dset(self, dset: Dict[str,np.ndarray]) -> Tuple[tf.Tensor,tf.Tensor]:
		t0 = time.time()
		with (self.device):
			y1, x1 = [], []
			for idx, (y,x) in enumerate(zip(dset['y'],dset['x'])):
				x0: int = tf.random.uniform([1], 0, self.slmax - self.series_len, dtype=tf.int32)[0]
				ys: tf.Tensor = tf.convert_to_tensor(y[x0:x0 + self.series_len], dtype=tf.float32)
				xs: tf.Tensor = tf.convert_to_tensor(x[x0:x0 + self.series_len], dtype=tf.float32)
				y1.append( tf.expand_dims( tnorm(ys,axis=0), 0) )
				x1.append( tf.expand_dims( xs, 0) )
			Y, X = self.apply_filters( tf.concat(y1, axis=0), tf.concat(x1, axis=0) )
			if Y.ndim == 2: Y = tf.expand_dims(Y, axis=2)
			print(f" Completed encoding in {(time.time()-t0)/60.0:.2f}m: ")
			print(f" --> X{X.shape}, Y{Y.shape}: (mean={tmean(Y):.5f}, std={tstd(Y):.5f}, mag={tmag(Y):.5f})")
			return X, Y