import random, time, numpy as np
import torch, math
from typing import List, Tuple, Mapping
from torch import Tensor, device, nn
from .embedding import EmbeddingLayer
from astrotime.util.math import logspace, tnorm
from astrotime.util.logging import elapsed

def clamp( idx: int ) -> int: return max( 0, idx )

def periods( cfg, device: device ) -> Tensor:
	fspace = logspace if (cfg.fscale == "log") else np.linspace
	nF: np.ndarray = fspace( cfg.base_freq_range[0], cfg.base_freq_range[1], cfg.nfreq )
	tF: Tensor = torch.FloatTensor( nF ).to(device)
	return torch.flip(1/tF)

class CorentropyLayer(EmbeddingLayer):

	def __init__(self, cfg, embedding_space: Tensor, device: device):
		EmbeddingLayer.__init__(self, cfg, embedding_space, device)
		self.P: Tensor = periods( cfg, device )
		self.ysigma: float = 1.0
		self.tsigma: float = 1.0
		self.init_log(f"CorentropyLayer: nfreq={self.nfreq} ")

	@property
	def cYn(self) -> float: return 1 /( math.sqrt(2*np.pi) * self.ysigma )

	@property
	def cYs(self) -> float: return 1 / 2*self.ysigma**2

	@property
	def cTn(self) -> float: return 1 /( math.sqrt(2*np.pi) * self.tsigma )

	@property
	def cTs(self) -> float: return 2 / self.tsigma**2

	def embed(self, ts: torch.Tensor, ys: torch.Tensor ) -> Tensor:
		self.init_log(f"CorentropyLayer shapes: ts{list(ts.shape)} ys{list(ys.shape)}")
		if ts.ndim == 1: ts = ts[None,:]
		if ys.ndim == 1: ys = ys[None, :]
		L: int =  ys.shape[1]
		cLn: float = 1.0/L**2
		T0:  Tensor = ts[:, :, None]
		T1:  Tensor = ts[:, None, :]
		Y0:  Tensor = ys[:, :, None]
		Y1:  Tensor = ys[:, None, :]
		DY:  Tensor = Y1 - Y0
		DT:  Tensor = T1 - T0
		GY:  Tensor = self.cYn * torch.exp( -self.cYs * DY**2  )
		NGY: Tensor = cLn * torch.sum(GY,dim=(1,2))
		CGY: Tensor = GY - NGY[:, None, None]
		UTP: Tensor = torch.sin(  DT * (np.pi/self.P) ) ** 2
		GT:  Tensor = self.cTn * torch.exp( -self.cTs * UTP )
		delt:Tensor = ts[:,-1] - ts[:,0]
		W:   Tensor = 0.54 + 0.46 * torch.cos( np.pi * DT / delt )
		V:   Tensor = CGY * GT * W
		return self.ysigma * cLn * torch.sum(V,dim=(1,2))

	def magnitude(self, embedding: Tensor) -> Tensor:
		return torch.sqrt( torch.sum( embedding**2, dim=1 ) )

	@property
	def nfeatures(self) -> int:
		return 1

	@property
	def output_series_length(self):
		return self.P.shape[0]
