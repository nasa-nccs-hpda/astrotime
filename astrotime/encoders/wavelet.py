import random, time, numpy as np, tensorflow as tf
from typing import Any, Dict, List, Optional, Tuple
from astrotime.encoders.base import Encoder
from astrotime.transforms.wwz import wwz
from astrotime.util.math import logspace, shp
from astrotime.util.math import tmean, tstd, tmag, tnorm
import keras

def norm_concat( xs: List[tf.Tensor]) -> tf.Tensor:
	xf = tf.concat( xs, axis=0 )
	print( f"norm_concat: {xf.shape}")
	return tnorm( xf, 1 )

class WaveletEncoder(Encoder):

	def __init__(self, device: str, series_len: int = 1000, fbounds: Tuple[float,float] = (0.1,10.0), nfreq: int = 1000, fscale: str = "log" ):
		super(WaveletEncoder, self).__init__( device )
		self.series_len = series_len
		self.fbeg, self.fend = fbounds
		self.nfreq = nfreq
		self.fscale = fscale
		self.freq: tf.Tensor = self.create_freq()
		self.slmax = 6000
		self.nfeatures = 5
		self.chan_first = False
		self.batch_size = 100

	def create_freq(self) -> tf.Tensor:
		fspace = logspace if (self.fscale == "log") else np.linspace
		return tf.convert_to_tensor( fspace( self.fbeg, self.fend, self.nfreq ), dtype=tf.float32 )

	def encode_dset(self, dset: Dict[str,np.ndarray]) -> Tuple[tf.Tensor,tf.Tensor]:
		t0 = time.time()
		with (self.device):
			amps, phases, coeffs = [], [], ([], [], [])
			y1, x1, wwz_start_time, wwz_end_time = [], [], time.time(), time.time()
			for idx, (y,x) in enumerate(zip(dset['y'],dset['x'])):
				x0: int = tf.random.uniform( [1], 0, self.slmax - self.series_len, dtype=tf.int32 )[0]
				ys: tf.Tensor = tf.convert_to_tensor( y[x0:x0+self.series_len], dtype=tf.float32 )
				xs: tf.Tensor = tf.convert_to_tensor( x[x0:x0+self.series_len], dtype=tf.float32)
				y1.append( tf.expand_dims( tnorm(ys, 0), 0 ) )
				x1.append( tf.expand_dims( xs, 0 ) )
				if idx % self.batch_size == self.batch_size-1:
					Y, X = tf.concat(y1,axis=0), tf.concat(x1,axis=0)
					amp, phase, cs = wwz(Y, X, self.freq, X[:,self.series_len//2] )
					amps.append( amp )
					phases.append( phase )
					for coeff, c in zip(coeffs, cs): coeff.append( c )
					y1, x1 = [], []
			amp, phase, coeff = norm_concat(amps), norm_concat(phases), [ norm_concat(c) for c in coeffs ]
			features = [amp,phase]+coeff
			dim = 1 if self.chan_first else 2
			encoded_dset = tf.stack( features[:self.nfeatures], axis=dim )
			print(f" Completed encoding in {(time.time()-t0)/60.0:.2f}m: amp{amp.shape}({tf.reduce_mean(amp):.2f},{tf.math.reduce_std(amp):.2f}), phase{phase.shape}({tf.reduce_mean(phase):.2f},{tf.math.reduce_std(phase):.2f}), coeff{coeff[0].shape}({tf.reduce_mean(coeff[0]):.2f},{tf.math.reduce_std(coeff[0]):.2f})")
			print(f" --> X{self.freq.shape} Y{encoded_dset.shape}({tf.reduce_mean(encoded_dset):.2f},{tf.math.reduce_std(encoded_dset):.2f})")
			return self.freq, encoded_dset