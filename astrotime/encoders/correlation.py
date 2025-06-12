import random, time, torch, numpy as np
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor, device
from omegaconf import DictConfig, OmegaConf
from astrotime.util.math import shp
from .embedding import EmbeddingLayer
from astrotime.util.math import tnorm
from astrotime.util.logging import elapsed
from torch import Tensor
import logging, math
from .wavelet import WaveletAnalysisLayer
log = logging.getLogger()

def delta( x: Tensor ) -> Tensor:
	dx: Tensor = (x[1:] - x[:-1]) / (x[-1] - x[0])
	dX = torch.zeros_like(x)
	dX[0] = dx[0]
	dX[1:] = dx
	return dX

class PolyEmbeddingLayer(EmbeddingLayer):

	def __init__(self, cfg, device: device):
		EmbeddingLayer.__init__(self,cfg,device)

	def embed(self, ts: torch.Tensor, ys: torch.Tensor) -> Tensor:
		print(f"     MODEL INPUT T: ts{list(ts.shape)}: ({ts.min().item():.2f}, {ts.max().item():.2f}, {ts.mean().item():.2f}, {ts.std().item():.2f}) ")
		print(f"     MODEL INPUT Y: ys{list(ys.shape)}: ({ys.min().item():.4f}, {ys.max().item():.4f}, {ys.mean().item():.4f}, {ys.std().item():.4f}) ")
		return ys

	@property
	def nfeatures(self) -> int:
		return 1

class CorrelationEmbedding(EmbeddingLayer):

	def __init__(self, cfg, embedding_space: Tensor, device: device):
		EmbeddingLayer.__init__(self, cfg, embedding_space, device)
		self.nfreq = cfg.nfreq
		self.C = cfg.decay_factor / (8 * np.pi ** 2)
		self.init_log(f"WaveletSynthesisLayer: nfreq={self.nfreq} ")
		self.decay_factor = cfg.decay_factor
		self.chan_first = True
		self.lag_step = cfg.lag_step

	def h(self, time_lag: float): return self.decay_factor * time_lag / 4

	def B(self, time_lag: float): return 1 / torch.sqrt( 2*np.pi*self.h(time_lag) )

	def A(self, time_lag: float): return 1 / ( 2*self.h(time_lag)**2 )

	@property
	def nfeatures(self):
		return 1

	def embed_series(self, ts: Tensor, ys: Tensor ) -> Tensor:
		self.init_log(f"WaveletSynthesisLayer shapes:")
		dt: float = (ts[1:] - ts[:-1]).median().item()
		for f in self.embedding_space:
			lag = 1/f
			tmax = ts[:-1] - lag
			itmax: int = (ts<tmax).nonzero()[0].item()
			T, Y = ts[:itmax],ys[:itmax]
			T1 = T + lag

class CorrelationAnalysisLayer(WaveletAnalysisLayer):

	def __init__(self, cfg, embedding_space: Tensor, device: device):
		WaveletAnalysisLayer.__init__(self, cfg, embedding_space, device)
		self.nfreq_oct: int   = cfg.nfreq_oct
		self.base_freq: float = cfg.base_freq
		self.noctaves: int    = cfg.noctaves
		self.nfreq: int       = self.nfreq_oct * self.noctaves

	def magnitude(self, embedding: Tensor, **kwargs) -> np.ndarray:
		t0 = time.time()
		mag: Tensor = torch.sqrt( torch.sum( embedding**2, dim=1 ) )
		cmag: Tensor = torch.corrcoef(mag)
		self.log.info(f"Completed folding magnitude in {elapsed(t0):.5f} sec: mag{list(cmag.shape)}")
		return cmag.to('cpu').numpy()

	def get_target_freq( self, target_period: float ) -> float:
		f0 = 1/target_period
		return f0

class AutoCorrelationLayer(EmbeddingLayer):

	def __init__(self, name: str,  cfg, embedding_space: Tensor, device: device):
		EmbeddingLayer.__init__(self, name, cfg, embedding_space, device)
		self.init_log(f"AutoCorrelationLayer: nfreq={self.nfreq} ")
		self.subbatch_size: int = cfg.get('subbatch_size',-1)
		self.noctaves: int = self.cfg.noctaves
		self.nfreq_oct: int = self.cfg.nfreq_oct

	@property
	def xdata(self) -> Tensor:
		return self._embedding_space

	@property
	def output_series_length(self) -> int:
		return self.nf

	def sbatch(self, ts: torch.Tensor, ys: torch.Tensor, subbatch: int) -> tuple[Tensor,Tensor]:
		sbr = [ subbatch*self.subbatch_size, (subbatch+1)*self.subbatch_size ]
		return ts[sbr[0]:sbr[1]], ys[sbr[0]:sbr[1]]

	def embed(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs) -> Tensor:
		if ys.ndim == 1:
			return self.embed_subbatch( ts[None,:], ys[None,:] )
		elif self.subbatch_size <= 0:
			return self.embed_subbatch( ts, ys )
		else:
			nsubbatches = math.ceil(ys.shape[0]/self.subbatch_size)
			subbatches = [ self.embed_subbatch( *self.sbatch(ts,ys,i), **kwargs ) for i in range(nsubbatches) ]
			result = torch.concat( subbatches, dim=0 )
			return result

	def embed_subbatch(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs ) -> Tensor:
		t0 = time.time()
		f: Tensor = self._embedding_space
		omega = 2.0 * math.pi * f
		omega_: Tensor = omega[None, :, None]  # broadcast-to(self.batch_size,self.nfreq,slen)
		ts: Tensor = ts[:, None, :]  # broadcast-to(self.batch_size,self.nfreq,slen)
		dz: Tensor = omega_ * ts

		self.init_log(f"AutoCorrelationLayer shapes: ts{list(ts.shape)} ys{list(ys.shape)} dz{list(dz.shape)}")

		def w_prod( x0: Tensor, x1: Tensor, dim=-1) -> Tensor:
			return torch.sum( x0 * x1, dim=dim)

		pw1: Tensor = torch.sin(dz)  # B, F, SLEN
		pw2: Tensor = torch.cos(dz)
		p1: Tensor = w_prod(ys, pw1)
		p2: Tensor = w_prod(ys, pw2)
		mag: Tensor =  torch.sqrt( p1**2 + p2**2 ) # B, F,
		self.init_log(f" --> mag{list(mag.shape)} pw1{list(pw1.shape)} p1{list(p1.shape)}  omega_{list(omega_.shape)}")

		odf: Tensor = omega*delta(f)
		p: Tensor = torch.flip( 1/f, [0] ) # P
		dz: Tensor = odf[:, None] * p[None, :] # F, P
		pw1: Tensor = torch.sin(dz) # F, P
		pw2: Tensor = torch.cos(dz) # F, P
		p1: Tensor = w_prod( mag[:, :, None], pw1[None, :, :], 1 ) # B, P
		p2: Tensor = w_prod( mag[:, :, None], pw2[None, :, :], 1 ) # B, P
		embedding: Tensor =  torch.sqrt( p1**2 + p2**2 ) # B, P
		self.init_log(f" --> mag{list(mag.shape)}  p{list(p.shape)} dz{list(dz.shape)} pw1{list(pw1.shape)} pw2{list(pw2.shape)}")
		self.init_log(f" Completed embedding{list(embedding.shape)} in {elapsed(t0):.5f} sec: nfeatures={embedding.shape[1]}")
		self.init_state = False
		return embedding

	@property
	def nf(self):
		return self._embedding_space.shape[0]

	@property
	def nfeatures(self):
		return 1

	def magnitude(self, embedding: Tensor) -> np.ndarray:
		self.init_log(f" -> Embedding magnitude{embedding.shape}")
		return embedding.cpu().numpy()








