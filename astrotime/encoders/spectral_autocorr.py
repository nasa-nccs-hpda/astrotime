
import random, time, numpy as np
import torch, math
from typing import List, Tuple, Mapping
from torch import Tensor, device, nn, matmul
from omegaconf import DictConfig, OmegaConf
from astrotime.util.math import log2space, tnorm
from astrotime.util.logging import elapsed
from astrotime.encoders.octaves import OctaveAnalysisLayer

def clamp( idx: int ) -> int: return max( 0, idx )

def spectral_space( cfg, device: device ) -> Tuple[np.ndarray,Tensor]:
	lspace = log2space( cfg.base_freq, cfg.base_freq*pow(2,cfg.noctaves), cfg.nfreq_oct*cfg.noctaves )
	tspace = torch.FloatTensor( lspace ).to(device)
	return lspace, tspace

def harmonics_space( cfg, device: device ) -> Tuple[np.ndarray,Tensor]:
	lspace = log2space( cfg.base_freq, cfg.base_freq*pow(2,cfg.nharmonics), cfg.nfreq_oct*cfg.nharmonics )
	tspace = torch.FloatTensor( lspace ).to(device)
	return lspace, tspace

def harmonics_filter( ts: np.ndarray, ys: np.ndarray, cfg: DictConfig, device ) -> np.ndarray:
	t: Tensor = torch.from_numpy( ts[None,:] if ts.ndim == 1 else ts )
	y: Tensor = torch.from_numpy( ys[None,:] if ys.ndim == 1 else ys )
	proj = HarmonicsFilterLayer( cfg, device )
	embedding = proj.embed( t, y )
	return proj.magnitude( embedding )

class HarmonicsFilterLayer(OctaveAnalysisLayer):

	def __init__(self, cfg, device: device):
		OctaveAnalysisLayer.__init__(self, cfg, harmonics_space(cfg, device)[1], device)
		self.fspace: np.ndarray = None

	def embed1(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs ) -> Tensor:
		alpha = 0.01
		spectral_features: torch.Tensor = super(HarmonicsFilterLayer, self).embed( ts, ys, **kwargs)
		spectral_projection: torch.Tensor = torch.sqrt(torch.sum(spectral_features ** 2, dim=1))
		self.fspace, sspace = spectral_space(self.cfg, self.device)
		harmonics: torch.Tensor = torch.stack( [sspace*ih for ih in range(1,6)], dim=1)
		df: torch.Tensor = (self._embedding_space[:,None,None] - harmonics[None,:,:]) # /harmonics[None,:,:]
		tdf = df[:,0,2][:df.shape[1]]
		W: torch.Tensor = torch.exp(-alpha*df**2).sum(dim=2)
		tW = W[:,0][:W.shape[1]]
		hfilter: torch.Tensor = matmul(spectral_projection,W)/self._embedding_space.shape[0]
		print( f"SpectralAutocorrelationLayer: harmonics{list(harmonics.shape)} df{list(df.shape)}  spectral_projection{list(spectral_projection.shape)}  sspace{list(sspace.shape)}, hspace{list(self._embedding_space.shape)}, W{list(W.shape)}  hfilter{list(hfilter.shape)} ")
		print(f"harmonics[0]: {harmonics[0].cpu().tolist()}")
		return tdf # tW # hfilter

	def embed(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs ) -> Tensor:
		alpha = 50.0
		print(f"SpectralAutocorrelationLayer:")
		spectral_features: torch.Tensor = super(HarmonicsFilterLayer, self).embed( ts, ys, **kwargs)
		spectral_projection: torch.Tensor = torch.sqrt(torch.sum(spectral_features ** 2, dim=1))
		self.fspace, sspace = spectral_space(self.cfg, self.device)
		f: torch.Tensor = sspace
		harmonics: torch.Tensor = torch.stack( [f * ih for ih in range(1, 6)], dim=1)
		df = (self._embedding_space[:,None,None] - harmonics[None,:,:])/self._embedding_space[:,None]
		W: torch.Tensor = torch.exp(-(df*alpha)**2).sum(dim=2)


		print(f" ----- embedding_space{list(self._embedding_space.shape)}: {self._embedding_space.min():.3f} -> {self._embedding_space.max():.3f}")
		print(f" ----- df{list(df.shape)}: {df.min():.3f} -> {df.max():.3f}")
		print(f" ----- W{list(W.shape)}: {W.min():.5f} -> {W.max():.5f}")
		print(f" ----- spectral_projection{list(spectral_projection.shape)}: {spectral_projection.min():.5f} -> {spectral_projection.max():.5f}")
		print(f" ----- f{list(f.shape)}: {f.min():.5f} -> {f.max():.5f}")
		print(f" ----- sspace{list(sspace.shape)}: {sspace.min():.5f} -> {sspace.max():.5f}")


		# harmonics: torch.Tensor = torch.stack( [sspace*ih for ih in range(1,6)], dim=1)
		# df: torch.Tensor = (self._embedding_space[:,None,None] - harmonics[None,:,:]) # /harmonics[None,:,:]
		# tdf = df[:,0,2][:df.shape[1]]
		# W: torch.Tensor = torch.exp(-alpha*df**2).sum(dim=2)
		# tW = W[:,0][:W.shape[1]]
		# hfilter: torch.Tensor = matmul(spectral_projection,W)/self._embedding_space.shape[0]
		# print( f"SpectralAutocorrelationLayer: harmonics{list(harmonics.shape)} df{list(df.shape)}  spectral_projection{list(spectral_projection.shape)}  sspace{list(sspace.shape)}, hspace{list(self._embedding_space.shape)}, W{list(W.shape)}  hfilter{list(hfilter.shape)} ")
		# print(f"harmonics[0]: {harmonics[0].cpu().tolist()}")
		return W[:sspace.shape[0]] # tW # hfilter

	def magnitude(self, embedding: Tensor, **kwargs) -> np.ndarray:
		mag: np.ndarray = embedding.cpu().numpy()
		return mag.squeeze()

	@property
	def xdata(self) -> np.ndarray:
		return self.fspace


