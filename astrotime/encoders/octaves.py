import random, time, numpy as np
import torch, math
from typing import List, Tuple, Mapping
from torch import Tensor, device, nn
from omegaconf import DictConfig, OmegaConf
from .embedding import EmbeddingLayer
from astrotime.util.math import log2space, tnorm
from astrotime.util.logging import elapsed

def clamp( idx: int ) -> int: return max( 0, idx )

def embedding_space( cfg, device: device ) -> Tuple[np.ndarray,Tensor]:
	lspace = log2space( cfg.base_freq, cfg.base_freq*pow(2,cfg.noctaves), cfg.nfreq_oct*cfg.noctaves )
	tspace = torch.FloatTensor( lspace ).to(device)
	return lspace, tspace

def wavelet_analysis_projection( ts: np.ndarray, ys: np.ndarray, fspace: np.ndarray, cfg: DictConfig, device ) -> np.ndarray:
	t: Tensor = torch.from_numpy( ts[None,:] if ts.ndim == 1 else ts )
	y: Tensor = torch.from_numpy( ys[None,:] if ys.ndim == 1 else ys )
	embedding_space: Tensor = torch.from_numpy(fspace)
	proj = OctaveAnalysisLayer( cfg, embedding_space, device )
	embedding = proj.embed( t, y )
	return proj.magnitude( embedding )

class OctaveAnalysisLayer(EmbeddingLayer):

	def __init__(self, name, cfg, embedding_space: Tensor, device: device):
		EmbeddingLayer.__init__(self, name, cfg, embedding_space, device)
		self.C = cfg.decay_factor / (8 * math.pi ** 2)
		self.init_log(f"WaveletAnalysisLayer: nfreq={self.nfreq} ")
		self.nfreq_oct: int   = cfg.nfreq_oct
		self.base_freq: float = cfg.base_freq
		self.noctaves: int    = cfg.noctaves
		self.nfreq: int       = embedding_space.shape[0]
		self.fold: bool = False

	@property
	def nfeatures(self) -> int:
		return 3

	@property
	def output_series_length(self):
		return self.nfreq

	def process_event(self, **kwargs):
		self.log.info(f"OctaveAnalysisLayer.process_event: event={kwargs}")
		if (kwargs["id"] == "KeyEvent") and (kwargs["key"] == "ctrl+o"):
			self.fold = not self.fold

	def embed(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs) -> Tensor:
		t0 = time.time()
		self.init_log(f"WaveletAnalysisLayer shapes: ts{list(ts.shape)} ys{list(ys.shape)}")
		slen: int = self.series_length if (self.series_length > 0) else ys.shape[1]
		ones: Tensor = torch.ones( ys.shape[0], self.nfreq, slen, device=self.device)
		tau = 0.5 * (ts[:, slen // 2] + ts[:, slen // 2 + 1])
		tau: Tensor = tau[:, None, None]
		omega = self._embedding_space * 2.0 * math.pi
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

	def fold_op(self, mag: np.ndarray) -> np.ndarray:
		norm: np.ndarray = np.ones(mag.shape[1])
		for i in range(1, self.noctaves):
			octave = mag[:, i * self.nfreq_oct:]
			mag[:, :octave.shape[1]] += octave
			norm[:octave.shape[1]] += 1
		mag = mag / norm[None, :]
		return mag

	def magnitude(self, embedding: Tensor, **kwargs) -> np.ndarray:
		mag: np.ndarray = torch.sqrt( torch.sum( embedding**2, dim=1 ) ).to('cpu').numpy()
		self.log.info(f"magnitude: mag{list(mag.shape)}, fold={self.fold}")
		return self.fold_op(mag) if self.fold else mag

	def get_target_freq( self, target_period: float ) -> float:
		f0 = 1/target_period
		return f0

