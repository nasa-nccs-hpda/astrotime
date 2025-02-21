from typing import Any, Dict, List, Optional, Tuple, Mapping
import numpy as np
from omegaconf import DictConfig
from torch import Tensor, device
from astrotime.transforms.filters import TrainingFilter
from astrotime.transforms.filters import RandomDownsample

class Encoder:

	def __init__(self, device: device, series_length: int, cfg: DictConfig ):
		self.device: device = device
		self.cfg = cfg
		self._series_len: int = series_length
		self.filters: List[TrainingFilter] = []
		if cfg.sparsity > 0.0:
			self.add_filter( RandomDownsample(sparsity=cfg.sparsity) )

	@property
	def series_length(self):
		return self._series_len

	def encode_dset(self, dset: Mapping[str,np.ndarray]) -> Tensor:
		raise NotImplementedError()

	def encode_batch(self, x: np.ndarray, y: np.ndarray) -> Tuple[Tensor, Tensor]:
		raise NotImplementedError()

	def add_filter(self, tfilter: TrainingFilter ):
		self.filters.append( tfilter )

	def apply_filters(self, x: np.ndarray, y: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
		for f in self.filters:
			x, y = f.apply( x, y, dim )
		return x, y