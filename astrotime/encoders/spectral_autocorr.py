
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

def embedding_space( cfg, device: device ) -> Tuple[np.ndarray,Tensor]:
	lspace = log2space( cfg.base_freq, cfg.base_freq*pow(2,cfg.noctaves), cfg.nfreq_oct*cfg.noctaves )
	tspace = torch.FloatTensor( lspace ).to(device)
	return lspace, tspace

def spectral_autocorrelation( ts: np.ndarray, ys: np.ndarray, fspace: np.ndarray, cfg: DictConfig, device ) -> np.ndarray:
	t: Tensor = torch.from_numpy( ts[None,:] if ts.ndim == 1 else ts )
	y: Tensor = torch.from_numpy( ys[None,:] if ys.ndim == 1 else ys )
	embedding_space: Tensor = torch.from_numpy(fspace)
	proj = SpectralAutocorrelationLayer( cfg, embedding_space, device )
	embedding = proj.embed( t, y )
	return proj.magnitude( embedding )

class SpectralAutocorrelationLayer(OctaveAnalysisLayer):

	def __init__(self, cfg, embedding_space: Tensor, device: device):
		OctaveAnalysisLayer.__init__(self, cfg, embedding_space, device)

	def embed(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs ) -> Tensor:
		spectral_features: torch.Tensor = super(SpectralAutocorrelationLayer, self).embed( ts, ys, **kwargs)
		spectral_projection: torch.Tensor = torch.sqrt(torch.sum(spectral_features ** 2, dim=1))
		sa: torch.Tensor = autocorrelation( spectral_projection )
		self.log.info(f"---- SpectralAutocorrelationLayer: sp{spectral_projection.shape}, mean={spectral_projection.mean():.2f} -> sa{sa.shape}, mean={sa.mean():.2f}, #nan={np.count_nonzero(np.isnan(sa))} --- ")
		return sa

	def magnitude(self, embedding: Tensor, **kwargs) -> np.ndarray:
		mag: np.ndarray = embedding.to('cpu').numpy()
		return mag.squeeze()


