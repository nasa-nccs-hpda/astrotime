
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
	return torch.flip(1/tF,[0])

# Pytorch GPU implementation of Algorithms from:
# An Information Theoretic Algorithm for Finding Periodicities in Stellar Light Curves
# Pablo Huijse, Pabl Estevez, Pavlos Protopapas, Pablo Zegers, Jose C. Prıncipe
# arXiv:1212.2398v1: IEEE TRANSACTIONS ON SIGNAL PROCESSING, VOL. 1, NO. 1, JANUARY 2012

class CorentropyLayer(EmbeddingLayer):

	def __init__(self, cfg, embedding_space: Tensor, device: device):
		EmbeddingLayer.__init__(self, cfg, embedding_space, device)
		self.P: Tensor = 1/embedding_space
		self.ysf: float = self.cfg.ysf
		self.tsf: float = self.cfg.tsf
		self.init_log(f"CorentropyLayer: nfreq={self.nfreq} ")

	@property
	def cYn(self) -> float: return 1 /( math.sqrt(2*np.pi) * self.ysf )

	@property
	def cYs(self) -> float: return 1 / 2*self.ysf**2

	def get_ykernel(self,ys: torch.Tensor) -> Tensor:
		L: int =  ys.shape[0]
		cLn: float = 1.0/L**2
		Y0:  Tensor = ys[:, None]                                        # [L,L]
		Y1:  Tensor = ys[None, :]                                        # [L,L]
		DY:  Tensor = Y1 - Y0                                               # [L,L]
		GY:  Tensor = self.cYn * torch.exp( -self.cYs * DY**2  )            # [L,L]     Eqn 3
		NGY: Tensor = cLn * torch.sum(GY)                                   #           Eqn 5
		return GY - NGY                                                     # [L,L]

	def get_tkernel(self,ts: torch.Tensor) -> Tensor:
		T0:  Tensor = ts[ :, None]                                        # [L,L]
		T1:  Tensor = ts[ None, :]                                        # [L,L]
		DT:  Tensor = T1 - T0                                             # [L,L]
		PP: Tensor = self.P[None,None,:]                                  # [L,L,F]
		tsig = self.tsf * PP                                              # [L,L,F]
		UTP: Tensor = torch.sin(  DT[:,:,None] * (np.pi/PP) )**2          # [L,L,F]
		cTn: Tensor = 1 / ( math.sqrt(2 * np.pi) * tsig )                 # [L,L,F]
		GT:  Tensor = cTn * torch.exp( -tsig * UTP )                      # [L,L,F]   Eqn 10
		return GT

	def get_W(self, ts: torch.Tensor) -> Tensor:
		T0: Tensor = ts[:, None]                                             # [L,L]
		T1: Tensor = ts[None, :]                                             # [L,L]
		DT: Tensor = T1 - T0                                                 # [L,L]
		delt:float = (ts[-1] - ts[0]).item()
		W:   Tensor = 0.54 + 0.46*torch.cos( np.pi*DT / delt )               # [L,L]     Eqn 12
		return W

	def embed_series(self, ts: torch.Tensor, ys: torch.Tensor ) -> Tensor:
		self.init_log(f"CorentropyLayer shapes: ts{list(ts.shape)} ys{list(ys.shape)}")
		ykernel: Tensor = self.get_ykernel(ys)                               # [L,L]
		L: int = ykernel.shape[0]
		cLn: float = 1.0 / L ** 2
		tkernel: Tensor = self.get_tkernel(ts)                               # [L,L,F]
		W: Tensor = self.get_W(ts)                                           #  [L,L]
		V: Tensor = ykernel[:,:,None] * tkernel * W[:,:,None]                # [L,L,F]
		return self.ysf * cLn * torch.sum(V,dim=(0,1))                       # [F]       Eqn 11

	def embed(self, ts: torch.Tensor, ys: torch.Tensor) -> Tensor:
		if ys.ndim == 1 and ts.ndim == 1:
			return self.embed_series(ts,ys)
		elif ys.ndim == 2 and ts.ndim == 2:
			series = zip( torch.unbind(ts, dim=0), torch.unbind(ys, dim=0) )
			return torch.stack([ self.embed_series(t,y) for t,y in series ], dim=0)
		else:
			raise Exception( f"Unsupported input shapes: ts{list(ts.shape)} ys{list(ys.shape)}" )

	def magnitude(self, embedding: Tensor) -> Tensor:
		return embedding

	@property
	def nfeatures(self) -> int:
		return 1

	@property
	def output_series_length(self):
		return self.P.shape[0]
