
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
		self.f0 = None
		self.alpha = 200.0
		self.nharmonics = 6

	def gaussian_harmonics0(self, f: float ) -> torch.Tensor:
		espace: torch.Tensor = self._embedding_space[:,None]
		harmonics = torch.FloatTensor([f * ih for ih in range(1, self.nharmonics + 1)]).to(self.device)
		df = self.alpha*(espace-harmonics)/f
		W: torch.Tensor = torch.exp(-df**2).sum(dim=1)
		return W

	def gaussian_harmonics1(self, f: torch.Tensor ) -> torch.Tensor:
		espace: torch.Tensor = self._embedding_space
		harmonics = torch.stack([f * ih for ih in range(1, self.nharmonics + 1)])
		df = self.alpha*(espace-harmonics)/f
		W: torch.Tensor = torch.exp(-df**2).sum(dim=1)
		return W

	def embed1(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs ) -> Tensor:
		self.log.info(f"SpectralAutocorrelationLayer:")
		spectral_features: torch.Tensor = super(HarmonicsFilterLayer, self).embed( ts, ys, **kwargs)
		spectral_projection: torch.Tensor = torch.sqrt(torch.sum(spectral_features ** 2, dim=1)).squeeze()
		self.fspace, sspace = spectral_space(self.cfg, self.device)
		W = self.gaussian_harmonics1( sspace )
		hfilter = torch.dot(W,spectral_projection)
		return hfilter[:self.fspace.shape[0]]

	def embed(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs) -> Tensor:
		self.log.info(f"SpectralAutocorrelationLayer:")
		spectral_features: torch.Tensor = super(HarmonicsFilterLayer, self).embed( ts, ys, **kwargs)
		spectral_projection: torch.Tensor = torch.sqrt(torch.sum(spectral_features ** 2, dim=1)).squeeze()
		self.fspace, sspace = spectral_space(self.cfg, self.device)
		hfilter = []
		for f in self.fspace:
			W = self.gaussian_harmonics0( f )
			hfilter.append( torch.dot(W,spectral_projection) )
		return torch.FloatTensor(hfilter[:self.fspace.shape[0]]).to(self.device)

#	"crtl-mouse-press", x = event.xdata, y = event.ydata, ax = event.inaxes
	def process_event(self, **kwargs ):
		self.log.info(f"           *** ---- HarmonicsFilterLayer.process_event: {kwargs} ")
		if kwargs["id"] == "crtl-mouse-press":
			self.f0 = kwargs["x"]
			self.log.info(f"           *** ---- set f0 = {self.f0} ")

	def magnitude(self, embedding: Tensor, **kwargs) -> np.ndarray:
		mag: np.ndarray = embedding.cpu().numpy()
		return mag.squeeze()

	@property
	def xdata(self) -> np.ndarray:
		return self.fspace


