import random, time, numpy as np
import torch
from torch import Tensor, device
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Optional, Tuple
from astrotime.encoders.base import Encoder
from astrotime.transforms.wwz import wwz
from astrotime.util.math import logspace, shp
from astrotime.util.math import tmean, tstd, tmag, tnorm

class WaveletEncoder(Encoder):

	def __init__(self, device: device, cfg: DictConfig ):
		super(WaveletEncoder, self).__init__( device, cfg )
		self.freq: Tensor = self.create_freq()
		self.chan_first = True

	def create_freq(self) -> Tensor:
		fspace = logspace if (self.cfg.fscale == "log") else np.linspace
		print( f"create_freq: fbounds = {self.cfg.freq_start} {self.cfg.freq_end}")
		return torch.FloatTensor( fspace( self.cfg.freq_start, self.cfg.freq_end, self.cfg.nfreq ) ).to(self.device)

	def encode_dset(self, dset: Dict[str,np.ndarray], **kwargs) -> Tuple[Tensor,Tensor]:
		t0 = time.time()
		concat_size = kwargs.get('concat_size',100)
		amps, phases, coeffs = [], [], ([], [], [])
		y1, x1, wwz_start_time, wwz_end_time = [], [], time.time(), time.time()
		for idx, (y,x) in enumerate(zip(dset['y'],dset['x'])):
			x, y = self.apply_filters(x,y,dim=0)
			x0: int = random.randint(0, self.cfg.max_series_length - self.series_length)
			ys: Tensor = torch.FloatTensor( y[x0:x0 + self.series_length] ).to(self.device)
			xs: Tensor = torch.FloatTensor( x[x0:x0 + self.series_length] ).to(self.device)
			y1.append(torch.unsqueeze(tnorm(ys, dim=0), dim=0))
			x1.append(torch.unsqueeze(xs, dim=0))
			if idx % concat_size == concat_size-1:
				Y, X = torch.concatenate(y1,dim=0), torch.concatenate(x1,dim=0)
				amp, phase, cs = wwz(Y, X, self.freq, X[:,self.series_length//2] )
				amps.append( amp )
				phases.append( phase )
				for coeff, c in zip(coeffs, cs): coeff.append( c )
				y1, x1 = [], []
		amp, phase, coeff =  torch.concatenate(amps,dim=0), torch.concatenate(phases,dim=0), [ torch.concatenate(c,dim=0) for c in coeffs ]
		features = [amp,phase]+coeff
		dim = 1 if self.chan_first else 2
		encoded_dset = torch.stack( features[:self.cfg.nfeatures], dim=dim )
		print(f" Completed encoding in {(time.time()-t0)/60.0:.2f}m: amp{amp.shape}({torch.mean(amp):.2f},{torch.std(amp):.2f}), phase{phase.shape}({torch.mean(phase):.2f},{torch.std(phase):.2f}), coeff{coeff[0].shape}({torch.mean(coeff[0]):.2f},{torch.std(coeff[0]):.2f})")
		print(f" --> X{self.freq.shape} Y{encoded_dset.shape}({torch.mean(encoded_dset):.2f},{torch.std(encoded_dset):.2f})")
		return self.freq, encoded_dset

	def encode_batch(self, x: np.ndarray, y: np.ndarray) -> Tuple[Tensor, Tensor]:
		x, y = self.apply_filters(x, y, 1)
		x0: int = random.randint(0, self.cfg.max_series_length - self.series_length)
		Y: Tensor = torch.FloatTensor(y[:, x0:x0 + self.series_length] ).to(self.device)
		X: Tensor = torch.FloatTensor(x[:, x0:x0 + self.series_length] ).to(self.device)
		Y = tnorm(Y, dim=1)
		amp, phase, cs = wwz(Y, X, self.freq, X[:, self.series_length // 2])
		features = [amp,phase]+list(cs)
		dim = 1 if self.chan_first else 2
		WWZ = torch.stack( features[:self.cfg.nfeatures], dim=dim )
		return self.freq, WWZ