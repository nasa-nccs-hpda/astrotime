import random, time, numpy as np, tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from typing import Any, Dict, List, Optional, Tuple
from astrotime.encoders.base import Encoder
from astrotime.transforms.wwz import wwz
from astrotime.util.math import logspace, shp
import keras

class WaveletEncoder(Encoder):

	def __init__(self, device: str, series_len: int = 1000, fbounds: Tuple[float,float] = (0.1,10.0), nfreq: int = 1000, fscale: str = "log" ):
		super().__init__()
		self.device = tf.device(device)
		self.series_len = series_len
		self.fbeg, self.fend = fbounds
		self.nfreq = nfreq
		self.fscale = fscale
		self.freq: np.ndarray = self.create_freq()
		self.slmax = 6000
		self.nfeatures = 5
		self.chan_first = True
		self.batch_size = 1000

	def create_freq(self) -> np.ndarray:
		fspace = logspace if (self.fscale == "log") else np.linspace
		return fspace( self.fbeg, self.fend, self.nfreq )

	def encode_dset(self, dset: Dict[str,tf.Tensor]) -> tf.Tensor:
		with (self.device):
			t0 = time.time()
			amps, phases, coeffs = [], [], ([], [], [])
			y1, x1, wwz_start_time, wwz_end_time = [], [], time.time(), time.time()
			for idx, (y,x) in enumerate(zip(dset['y'],dset['x'])):
				x0: int = tf.random.uniform( [1], 0, self.slmax - self.series_len, dtype=tf.int32 )[0]
				ys: tf.Tensor = tf.convert_to_tensor( y[x0:x0+self.series_len], dtype=tf.float32 )
				xs: tf.Tensor = tf.convert_to_tensor( x[x0:x0+self.series_len], dtype=tf.float32)
				y1.append( tf.expand_dims( keras.utils.normalize(ys, order=1), 0 ) )
				x1.append( tf.expand_dims( xs, 0 ) )
				if idx % self.batch_size == self.batch_size-1:
					wwz_start_time = time.time()
					Y, X = tf.concat(y1,axis=0), tf.concat(x1,axis=0)
					print(f" **wavelet: encoding batch {idx // self.batch_size} of {len(dset['y']) // self.batch_size}, x{shp(X)}, y{shp(Y)}, load-time={wwz_start_time-wwz_end_time:.2f}s")
					amp, phase, cs = wwz(Y, X, self.freq, X[:,self.series_len//2] )
					amps.append( amp )
					phases.append( phase )
					for coeff, c in zip(coeffs, cs): coeff.append( c )
					wwz_end_time = time.time()
					y1, x1 = [], []
					print(f" ---> Encoded amp{shp(amp)}, phase{shp(phase)}, C{shp(cs[0])}, encode-time={wwz_end_time-wwz_start_time:.2f}s")
		amp, phase, coeff = tf.concat(amps,axis=0), tf.concat(phases,axis=0), [ tf.concat(c,axis=0) for c in coeffs ]
		print( f" **wavelet: amp{shp(amp)} phase{shp(phase)} coeffs: {shp(coeff[0])} {shp(coeff[1])} {shp(coeff[2])}")
		features = [amp,phase]+coeff
		dim = 1 if self.chan_first else 2
		encoded_dset = tf.stack( features[:self.nfeatures], axis=dim )
		print(f" Completed encoding in ={time.time()-t0:.2f}s")
		return encoded_dset

# result = np.array(val_Xs)
#
# print(f"WaveletEncoder: dset keys = {list(dset.keys())}")
# ydata: np.ndarray = dset['y']
# print( f"  --> ydata{ydata.shape} y{ydata[0].shape} y{ydata[100].shape} y{ydata[1000].shape}")
# tr = self.series_len // 2
# t0 = random.randrange(tr, self.slmax - tr)
#
# return y

#
#
# def create_batch(self, xbatch: xa.Dataset, **kwargs ) -> Dict[str,torch.Tensor]:
# 	cnn_kernel_size = cnn_series_length( self.series_length, self.kernel_size, self.stride )
# 	kr = cnn_kernel_size // 2
# 	y: torch.Tensor = torch.FloatTensor(xbatch['y'].values).to(self.device)
# 	t: torch.Tensor = torch.FloatTensor(xbatch['t'].values).to(self.device)
# 	tindx = kwargs.get('tindx', random.randrange(kr, t.shape[1] - kr))
# 	target: torch.Tensor = torch.FloatTensor(xbatch[self.target_var].values).to(self.device)
# 	trng = [tindx - kr, tindx + kr]
# 	y: torch.Tensor = y[:, trng[0]:trng[1]]
# 	t: torch.Tensor = t[:, trng[0]:trng[1]]
# 	tau: torch.Tensor = ( t[:,kr] + t[:,kr+1] )/2
# 	dim = 1 if self.chan_first else 2
# 	amp, phase, coeff = wwz( y, t, self.freq, tau )
# 	features = [amp,phase]+list(coeff)
# 	batch: torch.Tensor = torch.stack( features[:self.nfeatures], dim=dim )
# 	# print( f"WaveletEncoder[nf={self.nfeatures}]: y{shp(y)} t{shp(t)} amp{shp(amp)} phase{shp(phase)} coeff-0{shp(coeff[0])} batch{shp(batch)} ")
# 	return dict( batch=batch, target=target, tau=tau )