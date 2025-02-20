import random, time, numpy as np
import torch
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple
from astrotime.encoders.base import Encoder
from astrotime.transforms.wwz_tf import wwz
from astrotime.util.math import logspace, shp
from astrotime.util.math import tmean, tstd, tmag, tnorm

class WaveletEncoder(Encoder):

	def __init__(self, device: str, series_len: int, nfreq: int , fbounds: Tuple[float,float], fscale: str, nfeatures: int, max_series_len: int ):
		super(WaveletEncoder, self).__init__( device, series_len )
		self.fbeg, self.fend = fbounds
		self.nfreq = nfreq
		self.fscale = fscale
		self.freq: Tensor = self.create_freq()
		self.slmax = 6000
		self.nfeatures = nfeatures
		self.chan_first = False
		self.batch_size = 100
		self.max_series_len = max_series_len

	def create_freq(self) -> Tensor:
		fspace = logspace if (self.fscale == "log") else np.linspace
		return torch.from_numpy( fspace( self.fbeg, self.fend, self.nfreq ) ).to(self.device)

	def encode_dset(self, dset: Dict[str,np.ndarray]) -> Tuple[Tensor,Tensor]:
		t0 = time.time()
		with (self.device):
			amps, phases, coeffs = [], [], ([], [], [])
			y1, x1, wwz_start_time, wwz_end_time = [], [], time.time(), time.time()
			for idx, (y,x) in enumerate(zip(dset['y'],dset['x'])):
				x, y = self.apply_filters(x,y,dim=0)
				x0: int = random.randint(0, self.max_series_len - self.series_len)
				ys: Tensor = torch.from_numpy( y[x0:x0 + self.series_len] ).to(self.device)
				xs: Tensor = torch.from_numpy( x[x0:x0 + self.series_len] ).to(self.device)
				y1.append(torch.unsqueeze(tnorm(ys, dim=0), dim=0))
				x1.append(torch.unsqueeze(xs, dim=0))
				if idx % self.batch_size == self.batch_size-1:
					Y, X = torch.concatenate(y1,dim=0), torch.concatenate(x1,dim=0)
					amp, phase, cs = wwz(Y, X, self.freq, X[:,self.series_len//2] )
					amps.append( amp )
					phases.append( phase )
					for coeff, c in zip(coeffs, cs): coeff.append( c )
					y1, x1 = [], []
			amp, phase, coeff =  torch.concatenate(amps,dim=0), torch.concatenate(phases,dim=0), [ torch.concatenate(c,dim=0) for c in coeffs ]
			features = [amp,phase]+coeff
			dim = 1 if self.chan_first else 2
			encoded_dset = torch.stack( features[:self.nfeatures], dim=dim )
			print(f" Completed encoding in {(time.time()-t0)/60.0:.2f}m: amp{amp.shape}({torch.mean(amp):.2f},{torch.std(amp):.2f}), phase{phase.shape}({torch.mean(phase):.2f},{torch.std(phase):.2f}), coeff{coeff[0].shape}({torch.mean(coeff[0]):.2f},{torch.std(coeff[0]):.2f})")
			print(f" --> X{self.freq.shape} Y{encoded_dset.shape}({torch.mean(encoded_dset):.2f},{torch.std(encoded_dset):.2f})")
			return self.freq, encoded_dset

	def encode_batch(self, x: np.ndarray, y: np.ndarray) -> Tuple[Tensor, Tensor]:
		with (self.device):
			x, y = self.apply_filters(x, y, 1)
			x0: int = random.randint(0, self.max_series_len - self.series_len)
			Y: Tensor = torch.from_numpy(y[:, x0:x0 + self.series_len] ).to(self.device)
			X: Tensor = torch.from_numpy(x[:, x0:x0 + self.series_len] ).to(self.device)
			Y = tnorm(Y, dim=1)
			amp, phase, cs = wwz(Y, X, self.freq, X[:, self.series_len // 2])
			features = [amp,phase]+list(cs)
			dim = 1 if self.chan_first else 2
			WWZ = torch.stack( features[:self.nfeatures], dim=dim )
			return self.freq, WWZ