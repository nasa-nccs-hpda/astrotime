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
		self._trends: List[np.ndarray] = []

	def embed(self, ts: torch.Tensor, ys: torch.Tensor) -> Tensor:
		x,y = ts.cpu().numpy().flatten(), ys.cpu().numpy().flatten()
		self.log.info(f"DetrendTransform input: time{x.shape}, range=({x.min():.3f}->{x.max():.3f})")
		time1, flux1, flatten_lc, trend_lc   = flatten( x, y, window_length=self.cfg.bw_winlen, method='biweight' )
		time2, flux2, flatten_lc1, trend_lc1 = flatten( time1, flatten_lc, method='pspline', break_tolerance=self.cfg.spline_minbrk )
		self._xdata = time2
		self._trends = [ trend_lc1, trend_lc ]
		self.log.info(f"   ******* detrended: time{self._xdata.shape}, range=({self._xdata.min():.3f}->{self._xdata.max():.3f})")
		return torch.from_numpy( flatten_lc1 )

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