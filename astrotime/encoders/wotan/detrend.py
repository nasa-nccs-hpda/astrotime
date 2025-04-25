import random, time, numpy as np
import torch, math
from typing import List, Tuple, Mapping
from omegaconf import DictConfig
from torch import Tensor, device
from astrotime.encoders.base import Transform
from astrotime.encoders.wotan.flatten import flatten

def clamp( idx: int ) -> int: return max( 0, idx )

def detrend( ts: np.ndarray, ys: np.ndarray, cfg: DictConfig ) -> np.ndarray:
	return flatten( ts.flatten(), ys.flatten(), window_length=cfg.detrend_window_length, method=cfg.detrend_method, return_trend=True)

class DetrendTransform(Transform):

	def __init__(self, cfg: DictConfig, device: device):
		Transform.__init__(self, cfg, device)
		self._xdata = None

	def embed(self, ts: torch.Tensor, ys: torch.Tensor) -> Tensor:
		self._xdata = ts
		x,y = ts.cpu().numpy().flatten(), ys.cpu().numpy().flatten()
		flatten_lc, trend_lc = flatten(x,y, window_length=self.cfg.detrend_window_length, method=self.cfg.detrend_method, return_trend=True)
		return torch.from_numpy(flatten_lc) # torch.stack( [torch.from_numpy(flatten_lc), torch.from_numpy(trend_lc)],  dim=1 )

	def magnitude(self, embedding: Tensor) -> np.ndarray:
		return embedding.cpu().numpy()

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