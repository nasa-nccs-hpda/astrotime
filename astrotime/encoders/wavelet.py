import random, time, numpy as np, tensorflow as tf
from typing import Any, Dict, List, Optional, Tuple
from astrotime.encoders.base import Encoder
from astrotime.transforms.wwz import wwz
from astrotime.util.math import logspace, shp
from astrotime.util.math import tmean, tstd, tmag, tnorm

class WaveletEncoder(Encoder):

	def __init__(self, device: str, series_len: int, nfreq: int , fbounds: Tuple[float,float], fscale: str, nfeatures: int, max_series_len: int ):
		super(WaveletEncoder, self).__init__( device, series_len )
		self.fbeg, self.fend = fbounds
		self.nfreq = nfreq
		self.fscale = fscale
		self.freq: tf.Tensor = self.create_freq()
		self.slmax = 6000
		self.nfeatures = nfeatures
		self.chan_first = False
		self.batch_size = 100
		self.max_series_len = max_series_len

	def create_freq(self) -> tf.Tensor:
		fspace = logspace if (self.fscale == "log") else np.linspace
		return tf.convert_to_tensor( fspace( self.fbeg, self.fend, self.nfreq ), dtype=tf.float32 )

	def encode_dset(self, dset: Dict[str,np.ndarray]) -> Tuple[tf.Tensor,tf.Tensor]:
		t0 = time.time()
		with (self.device):
			amps, phases, coeffs = [], [], ([], [], [])
			y1, x1, wwz_start_time, wwz_end_time = [], [], time.time(), time.time()
			for idx, (y,x) in enumerate(zip(dset['y'],dset['x'])):
				x, y = self.apply_filters(x,y,0)
				x0: int = tf.random.uniform( [1], 0, self.max_series_len-self.series_len, dtype=tf.int32 )[0]
				ys: tf.Tensor = tf.convert_to_tensor( y[x0:x0+self.series_len], dtype=tf.float32 )
				xs: tf.Tensor = tf.convert_to_tensor( x[x0:x0+self.series_len], dtype=tf.float32)
				y1.append( tf.expand_dims( tnorm(ys, 0), 0 ) )
				x1.append( tf.expand_dims( xs, 0 ) )
				if idx % self.batch_size == self.batch_size-1:
					Y, X = tf.concat(y1,axis=0), tf.concat(x1,axis=0)
					# print(f"Encoding wwz batch: x{shp(x)}, y{shp(y)} -> X{X.shape} Y{Y.shape}")
					amp, phase, cs = wwz(Y, X, self.freq, X[:,self.series_len//2] )
					amps.append( amp )
					phases.append( phase )
					for coeff, c in zip(coeffs, cs): coeff.append( c )
					y1, x1 = [], []
			amp, phase, coeff =  tf.concat(amps,0), tf.concat(phases,0), [ tf.concat(c,0) for c in coeffs ]
			features = [amp,phase]+coeff
			dim = 1 if self.chan_first else 2
			encoded_dset = tf.stack( features[:self.nfeatures], axis=dim )
			print(f" Completed encoding in {(time.time()-t0)/60.0:.2f}m: amp{amp.shape}({tf.reduce_mean(amp):.2f},{tf.math.reduce_std(amp):.2f}), phase{phase.shape}({tf.reduce_mean(phase):.2f},{tf.math.reduce_std(phase):.2f}), coeff{coeff[0].shape}({tf.reduce_mean(coeff[0]):.2f},{tf.math.reduce_std(coeff[0]):.2f})")
			print(f" --> X{self.freq.shape} Y{encoded_dset.shape}({tf.reduce_mean(encoded_dset):.2f},{tf.math.reduce_std(encoded_dset):.2f})")
			return self.freq, encoded_dset

	def encode_batch(self, x: np.ndarray, y: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
		with (self.device):
			x, y = self.apply_filters(x, y, 1)
			x0: int = 0 # tf.random.uniform([1], 0, self.max_series_len - self.series_len, dtype=tf.int32)[0]
			Y: tf.Tensor = tf.convert_to_tensor(y[:, x0:x0 + self.series_len], dtype=tf.float32)
			X: tf.Tensor = tf.convert_to_tensor(x[:, x0:x0 + self.series_len], dtype=tf.float32)
			Y = tnorm(Y, axis=1)
			#print( f"Encoding wwz batch: x{shp(x)}, y{shp(y)} -> X{X.shape} Y{Y.shape}")
			amp, phase, cs = wwz(Y, X, self.freq, X[:, self.series_len // 2])
			features = [amp,phase]+list(cs)
			dim = 1 if self.chan_first else 2
			WWZ = tf.stack( features[:self.nfeatures], axis=dim )
			# print(f" --> f{self.freq.shape} WWZ{WWZ.shape}")
			return self.freq, WWZ