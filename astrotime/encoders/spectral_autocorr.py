
import random, time, numpy as np
import torch, math
from typing import List, Tuple, Mapping
from torch import Tensor, device, nn
from omegaconf import DictConfig, OmegaConf
from astrotime.util.math import log2space, tnorm
from astrotime.util.logging import elapsed
from astrotime.util.stats import autocorrelation
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

def spectral_autocorrelation( ts: np.ndarray, ys: np.ndarray, fspace: np.ndarray, cfg: DictConfig, device ) -> np.ndarray:
	t: Tensor = torch.from_numpy( ts[None,:] if ts.ndim == 1 else ts )
	y: Tensor = torch.from_numpy( ys[None,:] if ys.ndim == 1 else ys )
	proj = SpectralAutocorrelationLayer( cfg, device )
	embedding = proj.embed( t, y )
	return proj.magnitude( embedding )

class SpectralAutocorrelationLayer(OctaveAnalysisLayer):

	def __init__(self, cfg, device: device):
		OctaveAnalysisLayer.__init__(self, cfg, harmonics_space(cfg, device)[1], device)

	def embed(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs ) -> Tensor:
		spectral_features: torch.Tensor = super(SpectralAutocorrelationLayer, self).embed( ts, ys, **kwargs)
		spectral_projection: torch.Tensor = torch.sqrt(torch.sum(spectral_features ** 2, dim=1))
		print(f"---- SpectralAutocorrelationLayer: sp{list(spectral_projection.shape)}, mean={spectral_projection.mean():.2f} std={spectral_projection.std():.5f} ")
		return spectral_projection

	def magnitude(self, embedding: Tensor, **kwargs) -> np.ndarray:
		mag: np.ndarray = embedding.cpu().numpy()
		return mag.squeeze()


