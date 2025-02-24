import random, time, numpy as np
import torch, math
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
				amp, phase, cs = wwz(Y, X, self.freq, X[:,self.series_length//2], self.device )
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
		amp, phase, cs = wwz(Y, X, self.freq, X[:, self.series_length // 2], self.device)
		features = [amp,phase]+list(cs)
		dim = 1 if self.chan_first else 2
		WWZ = torch.stack( features[:self.cfg.nfeatures], dim=dim )
		return self.freq, WWZ


class WaveletEncoderLayer(torch.nn.Module):

	def __init__(self, cfg, device: device):
		torch.nn.Module.__init__(self)
		self.requires_grad_(False)
		self.device = device
		self.cfg = cfg
		self.nts = cfg.series_length
		self.nf = cfg.nfreq
		self.nb = cfg.batch_size
		self.C = 1 / (8 * math.pi ** 2)
		fspace = logspace if (self.cfg.fscale == "log") else np.linspace
		self.freq = torch.FloatTensor( fspace( self.cfg.freq_start, self.cfg.freq_end, self.cfg.nfreq ) ).to(self.device)
		self.ones: Tensor = torch.ones( self.nb, self.nf, self.nts, device=self.device)

	def forward(self, ys: torch.Tensor, ts: torch.Tensor ):
		tau = 0.5 * (ts[:, self.nts / 2] + ts[:, self.nts / 2 + 1])
		tau: Tensor = tau[:, None, None]
		omega = self.freq * 2.0 * math.pi
		omega_: Tensor = omega[None, :, None]  # broadcast-to(self.nb,self.nf,self.nts)
		ts: Tensor = ts[:, None, :]  # broadcast-to(self.nb,self.nf,self.nts)
		ys: Tensor = ys[:, None, :]  # broadcast-to(self.nb,self.nf,self.nts)
		dt: Tensor = (ts - tau)
		dz: Tensor = omega_ * dt
		weights: Tensor = torch.exp(-self.C * dz ** 2)
		sum_w: Tensor = torch.sum(weights, dim=-1)

		def w_prod(xs: Tensor, ys: Tensor) -> Tensor:
			return torch.sum(weights * xs * ys, dim=-1) / sum_w

		theta: Tensor = omega_ * ts
		sin_basis: Tensor = torch.sin(theta)
		cos_basis: Tensor = torch.cos(theta)


		sin_one: Tensor = w_prod(sin_basis, self.ones)
		cos_one: Tensor = w_prod(cos_basis, self.ones)
		sin_cos: Tensor = w_prod(sin_basis, cos_basis)
		sin_sin: Tensor = w_prod(sin_basis, sin_basis)
		cos_cos: Tensor = w_prod(cos_basis, cos_basis)

		numerator: Tensor = 2 * (sin_cos - sin_one * cos_one)
		denominator: Tensor = (cos_cos - cos_one ** 2) - (sin_sin - sin_one ** 2)
		time_shift: Tensor = torch.atan2(numerator, denominator) / (2 * omega)  # Eq. (S5)
		time_shift_: Tensor = time_shift[:, :, None]  # broadcast-to(self.nb,self.nf,self.nts)

		sin_shift: Tensor = torch.sin(omega_ * (ts - time_shift_))
		cos_shift: Tensor = torch.cos(omega_ * (ts - time_shift_))
		sin_tau_center: Tensor = torch.sin(omega * (time_shift - tau[:, :, 0]))
		cos_tau_center: Tensor = torch.cos(omega * (time_shift - tau[:, :, 0]))

		ys_cos_shift: Tensor = w_prod(ys, cos_shift)
		ys_sin_shift: Tensor = w_prod(ys, sin_shift)
		ys_one: Tensor = w_prod(ys, self.ones)
		cos_shift_one: Tensor = w_prod(cos_shift, self.ones)
		sin_shift_one: Tensor = w_prod(sin_shift, self.ones)

		A: Tensor = 2 * (ys_cos_shift - ys_one * cos_shift_one)
		B: Tensor = 2 * (ys_sin_shift - ys_one * sin_shift_one)

		a0: Tensor = ys_one
		a1: Tensor = cos_tau_center * A - sin_tau_center * B  # Eq. (S6)
		a2: Tensor = sin_tau_center * A + cos_tau_center * B  # Eq. (S7)

		wwp: Tensor = a1 ** 2 + a2 ** 2
		phase: Tensor = torch.atan2(a2, a1)
		coeff: Tuple[Tensor, Tensor, Tensor] = (a0, a1, a2)
		return wwp, phase, coeff