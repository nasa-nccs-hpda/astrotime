import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Any, Dict, List, Optional, Tuple
from astrotime.encoders.base import Encoder
from astrotime.transforms.wwz import wwz
from astrotime.util.math import logspace

class WaveletEncoder(Encoder):

	def __init__(self, series_len: int = 1000, fbounds: Tuple[float,float] = (0.1,10.0), nfreq: int = 1000, fscale: str = "log" ):
		super().__init__()
		self.series_len = series_len
		self.fbeg, self.fend = fbounds
		self.nfreq = nfreq
		self.fscale = fscale
		self.freq: np.ndarray = self.create_freq()

	def create_freq(self) -> np.ndarray:
		fspace = logspace if (self.fscale == "log") else np.linspace
		return fspace( self.fbeg, self.fend, self.nfreq )

	def encode_dset(self, batch_data: Dict[str,np.ndarray]) -> np.ndarray:
		y: np.ndarray = batch_data['y']
		print( f"WaveletEncoder: y{y.shape}")

		return y

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