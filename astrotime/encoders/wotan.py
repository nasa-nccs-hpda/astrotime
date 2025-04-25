import random, time, numpy as np
import torch, math
from typing import List, Tuple, Mapping
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, device, nn
from .base import Transform
from wotan import flatten
from astrotime.util.math import logspace, tnorm
from astrotime.util.logging import elapsed

def clamp( idx: int ) -> int: return max( 0, idx )


def detrend( ts: np.ndarray, ys: np.ndarray, cfg: DictConfig, device ) -> np.ndarray:
	t: Tensor = torch.from_numpy( ts[None,:] if ts.ndim == 1 else ts )
	y: Tensor = torch.from_numpy( ys[None,:] if ys.ndim == 1 else ys )
	transform = DetrendTransform( cfg,  device )
	detrended: Tensor = transform.embed( t, y )
	return transform.magnitude( detrended )

class DetrendTransform(Transform):

	def __init__(self, cfg: DictConfig, device: device):
		Transform.__init__(self, cfg, device)
		self._xdata = None
# detrend_window_length = 0.5
# detrend_method = 'biweight'

	def embed(self, ts: torch.Tensor, ys: torch.Tensor) -> Tensor:
		self._xdata = ts
		flatten_lc, trend_lc = flatten(ts.flatten(), ys.flatten(), window_length=self.cfg.detrend_window_length, method=self.cfg.detrend_method, return_trend=True)
		return torch.from_numpy(flatten_lc) # torch.stack( [torch.from_numpy(flatten_lc), torch.from_numpy(trend_lc)],  dim=1 )

	def magnitude(self, embedding: Tensor) -> np.ndarray:
		return embedding[0].cpu().numpy()

	@property
	def xdata(self) -> np.ndarray:
		return self._xdata.cpu().numpy()

	@property
	def nfeatures(self) -> int:
		return 2

	@property
	def output_series_length(self):
		return self.cfg.nfreq

	def get_target_freq( self, target_period: float ) -> float:
		f0 = 1/target_period
		return f0