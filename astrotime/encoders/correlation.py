import random, time, torch, numpy as np
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor, device
from omegaconf import DictConfig, OmegaConf
from astrotime.util.math import shp
from .embedding import EmbeddingLayer
from astrotime.util.math import tnorm
from astrotime.util.logging import elapsed
from torch import Tensor
import logging
from .wavelet import WaveletAnalysisLayer
log = logging.getLogger()

class PolyEmbeddingLayer(EmbeddingLayer):

	def __init__(self, cfg, device: device):
		EmbeddingLayer.__init__(self,cfg,device)

	def embed(self, ts: torch.Tensor, ys: torch.Tensor ) -> Tensor:
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








