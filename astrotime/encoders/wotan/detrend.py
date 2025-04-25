import random, time, numpy as np
import torch, math
from typing import List, Tuple, Mapping
from omegaconf import DictConfig
from torch import Tensor, device
from astrotime.encoders.base import Transform
from astrotime.encoders.wotan.flatten import flatten

def clamp( idx: int ) -> int: return max( 0, idx )

def detrend( ts: np.ndarray, ys: np.ndarray, cfg: DictConfig ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
	return flatten( ts.flatten(), ys.flatten(), window_length=cfg.detrend_window_length, method=cfg.detrend_method )

class DetrendTransform(Transform):

	def __init__(self, cfg: DictConfig, device: device):
		Transform.__init__(self, cfg, device)
		self._xdata: np.ndarray = None

	def embed(self, ts: torch.Tensor, ys: torch.Tensor) -> Tensor:
		x,y = ts.cpu().numpy().flatten(), ys.cpu().numpy().flatten()
		fvals = flatten(x,y, window_length=self.cfg.detrend_window_length, method=self.cfg.detrend_method )
		self._xdata = fvals[0]
		return torch.from_numpy( np.stack(fvals, axis=1) )

	def magnitude(self, embedding: Tensor) -> np.ndarray:
		return embedding[:,3].cpu().numpy()

	@property
	def xdata(self) -> np.ndarray:
		return self._xdata

	@property
	def nfeatures(self) -> int:
		return 1

	@property
	def output_series_length(self):
		return self.cfg.nfreq

	def get_target_freq( self, target_period: float ) -> float:
		f0 = 1/target_period
		return f0