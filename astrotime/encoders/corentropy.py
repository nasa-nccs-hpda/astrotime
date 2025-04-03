import random, time, numpy as np
import torch, math
from typing import List, Tuple, Mapping
from torch import Tensor, device, nn
from .embedding import EmbeddingLayer
from astrotime.util.math import logspace, tnorm
from astrotime.util.logging import elapsed

def clamp( idx: int ) -> int: return max( 0, idx )

def embedding_space( cfg, device: device ) -> Tuple[np.ndarray,Tensor]:
	fspace = logspace if (cfg.fscale == "log") else np.linspace
	nspace = fspace( cfg.base_freq_range[0], cfg.base_freq_range[1], cfg.nfreq )
	tspace = torch.FloatTensor( nspace ).to(device)
	return nspace, tspace

class CorentropyLayer(EmbeddingLayer):

	def __init__(self, cfg, embedding_space: Tensor, device: device):
		EmbeddingLayer.__init__(self, cfg, embedding_space, device)
		self.nfreq = cfg.nfreq
		self.msigma: float = 1.0
		self.init_log(f"CorentropyLayer: nfreq={self.nfreq} ")

	@property
	def cy0(self) -> float: return 1 / math.sqrt( 2*np.pi*self.msigma)

	@property
	def cy1(self) -> float: return 1 / 2*self.msigma**2

	def embed(self, ts: torch.Tensor, ys: torch.Tensor ) -> Tensor:
		t0 = time.time()
		self.init_log(f"CorentropyLayer shapes: ts{list(ts.shape)} ys{list(ys.shape)}")
		L: int =  ys.shape[1]
		T0: Tensor = ts[:, :, None]
		T1: Tensor = ts[:, None, :]
		Y0: Tensor = ys[:, :, None]
		Y1: Tensor = ys[:, None, :]
		DY: Tensor = Y1 - Y0
		DT: Tensor = T1 - T0
		GY:  Tensor = self.cy0 * torch.exp( -self.cy1 * DY**2  )
		NGY: Tensor = torch.sum(GY, dim=(1,2))/L**2
		CGY: Tensor = GY - NGY[:, None, None]





		ones: Tensor = torch.ones( ys.shape[0], self.nfreq, slen, device=self.device)
		tau = 0.5 * (ts[:, slen // 2] + ts[:, slen // 2 + 1])
		tau: Tensor = tau[:, None, None]
		omega = self.embedding_space * 2.0 * math.pi
		omega_: Tensor = omega[None, :, None]  # broadcast-to(self.batch_size,self.nfreq,slen)
		ts: Tensor = ts[:, None, :]  # broadcast-to(self.batch_size,self.nfreq,slen)
		dt: Tensor = (ts - tau)
		self.init_log(f" tau{list(tau.shape)} dt{list(dt.shape)} ones{list(ones.shape)}")
		dz: Tensor = omega_ * dt
		weights: Tensor = torch.exp(-self.C * dz ** 2) if (self.cfg.decay_factor > 0.0) else 1.0
		sum_w: Tensor = torch.sum(weights, dim=-1) if (self.cfg.decay_factor > 0.0) else 1.0

		def w_prod( x0: Tensor, x1: Tensor) -> Tensor:
			return torch.sum(weights * x0 * x1, dim=-1) / sum_w

		pw1: Tensor = torch.sin(dz)
		pw2: Tensor = torch.cos(dz)
		self.init_log(f" --> pw0{list(ones.shape)} pw1{list(pw1.shape)} pw2{list(pw2.shape)}  ")

		p0: Tensor = w_prod(ys, ones)
		p1: Tensor = w_prod(ys, pw1)
		p2: Tensor = w_prod(ys, pw2)
		self.init_log(f" --> p0{list(p0.shape)} p1{list(p1.shape)} p2{list(p2.shape)}")

		rv: Tensor = torch.concat( (p0[:, None, :], p1[:, None, :], p2[:, None, :]), dim=1)
		self.init_log(f" Completed embedding in {elapsed(t0):.5f} sec: result{list(rv.shape)}")
		self.init_state = False
		return rv

	def magnitude(self, embedding: Tensor) -> Tensor:
		return torch.sqrt( torch.sum( embedding**2, dim=1 ) )

	@property
	def nfeatures(self) -> int:
		return 3

	@property
	def output_series_length(self):
		return self.cfg.nfreq
