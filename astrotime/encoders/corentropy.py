
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
		self.ysigma: float = cfg.ysigma
		self.tsigma: float = cfg.tsigma
		self.init_log(f"CorentropyLayer: nfreq={self.nfreq} ")

	@property
	def cYn(self) -> float: return 1 /( math.sqrt(2*np.pi) * self.ysigma )

	@property
	def cYs(self) -> float: return 1 / 2*self.ysigma**2

	@property
	def cTn(self) -> float: return 1 /( math.sqrt(2*np.pi) * self.tsigma )

	@property
	def cTs(self) -> float: return 2 / self.tsigma**2

	def get_ykernel(self,ys: torch.Tensor) -> Tensor:
		if ys.ndim == 1: ys = ys[None, :]                                  # [B,L]
		L: int =  ys.shape[1]
		cLn: float = 1.0/L**2
		Y0:  Tensor = ys[:, :, None]                                        # [B,L,L]
		Y1:  Tensor = ys[:, None, :]                                        # [B,L,L]
		DY:  Tensor = Y1 - Y0                                               # [B,L,L]
		GY:  Tensor = self.cYn * torch.exp( -self.cYs * DY**2  )            # [B,L,L]     Eqn 3
		NGY: Tensor = cLn * torch.sum(GY,dim=(1,2))                         # [B]         Eqn 5
		return GY - NGY[:, None, None]                                      # [B,L,L]

	def get_tkernel(self,ts: torch.Tensor) -> Tensor:
		if ts.ndim == 1: ts = ts[None,:]                                   # [B,L]
		T0:  Tensor = ts[:, :, None]                                        # [B,L,L]
		T1:  Tensor = ts[:, None, :]                                        # [B,L,L]
		DT:  Tensor = T1 - T0                                               # [B,L,L]
		PP: Tensor = self.P[None,None,None,:]                               # [B,L,L,F]
		UTP: Tensor = torch.sin(  DT[:,:,:,None] * (np.pi/PP) )**2          # [B,L,L,F]
		GT:  Tensor = self.cTn * torch.exp( -self.cTs * UTP )               # [B,L,L,F]   Eqn 10
		return GT

	def get_W(self, ts: torch.Tensor) -> Tensor:
		if ts.ndim == 1: ts = ts[None, :]                                       # [B,L]
		T0: Tensor = ts[:, :, None]                                             # [B,L,L]
		T1: Tensor = ts[:, None, :]                                             # [B,L,L]
		DT: Tensor = T1 - T0                                                    # [B,L,L]
		delt:Tensor = ts[:,-1] - ts[:,0]                                        # [B]
		W:   Tensor = 0.54 + 0.46*torch.cos( np.pi*DT / delt[:,None,None] )     # [B,L,L]     Eqn 12
		return W

	def embed(self, ts: torch.Tensor, ys: torch.Tensor ) -> Tensor:
		self.init_log(f"CorentropyLayer shapes: ts{list(ts.shape)} ys{list(ys.shape)}")
		ykernel: Tensor = self.get_ykernel(ys)                               # [B,L,L]
		L: int = ykernel.shape[1]
		cLn: float = 1.0 / L ** 2
		tkernel: Tensor = self.get_tkernel(ts)                               # [B,L,L,F]
		W: Tensor = self.get_W(ts)                                           #  [B]
		V: Tensor = ykernel[:,:,:,None] * tkernel * W[:,:,:,None]            # [B,L,L,F]
		return self.ysigma * cLn * torch.sum(V,dim=(1,2))                    # [B,F]       Eqn 11

	def magnitude(self, embedding: Tensor) -> Tensor:
		return embedding

	@property
	def nfeatures(self) -> int:
		return 1

	@property
	def output_series_length(self):
		return self.P.shape[0]
