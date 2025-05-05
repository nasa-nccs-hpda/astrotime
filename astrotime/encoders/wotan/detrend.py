import random, time, numpy as np
import torch, math
from typing import List, Tuple, Mapping
from omegaconf import DictConfig
from torch import Tensor, device
from astrotime.encoders.base import Transform

def clamp( idx: int ) -> int: return max( 0, idx )

def detrend( ts: np.ndarray, ys: np.ndarray, cfg: DictConfig ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
	from astrotime.encoders.wotan.flatten import flatten
	time1, flux1, flatten_lc, trend_lc = flatten( ts.flatten(), ys.flatten(), window_length=cfg.detrend_window_length, method='biweight')
	return time1, flux1, flatten_lc, trend_lc

class DetrendTransform(Transform):

	def __init__(self, name: str, cfg: DictConfig, device: device):
		Transform.__init__(self, name, cfg, device)
		self._xdata: np.ndarray = None
		self._trends: List[np.ndarray] = []

	def embed(self, ts: torch.Tensor, ys: torch.Tensor) -> Tensor:
		x,y = ts.cpu().numpy().flatten(), ys.cpu().numpy().flatten()
		self.log.info(f"DetrendTransform input: time{x.shape}, range=({x.min():.3f}->{x.max():.3f})")
		time1, flux1, flatten_lc, trend_lc   = detrend( x, y, self.cfg )
		self._xdata = time1
		self._trends = [trend_lc, trend_lc]
		self.log.info(f"   ******* detrended: time{self._xdata.shape}, range=({self._xdata.min():.3f}->{self._xdata.max():.3f})")
		return torch.from_numpy( flatten_lc )

	def magnitude(self, embedding: Tensor) -> np.ndarray:
		return embedding.cpu().numpy()

	@property
	def xdata(self) -> np.ndarray:
		return self._xdata

	def trend(self, idx: int) -> np.ndarray:
		return self._trends[idx]

	@property
	def nfeatures(self) -> int:
		return 1

	@property
	def output_series_length(self):
		return self.cfg.nfreq

	def get_target_freq( self, target_period: float ) -> float:
		f0 = 1/target_period
		return f0