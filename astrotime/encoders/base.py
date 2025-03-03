from typing import List, Tuple, Mapping
import logging, numpy as np
from omegaconf import DictConfig
from torch import Tensor, device
from astrotime.trainers.filters import TrainingFilter
from astrotime.trainers.filters import RandomDownsample

class Encoder:

	def __init__(self, cfg: DictConfig, device: device  ):
		self.device: device = device
		self.cfg = cfg
		self.filters: List[TrainingFilter] = []
		self.log = logging.getLogger()
		if cfg.sparsity > 0.0:
			self.add_filter( RandomDownsample(sparsity=cfg.sparsity) )

	@property
	def nfeatures(self) -> int:
		raise NotImplementedError("Expansion.nfeatures() not implemented")

	@property
	def input_series_length(self):
		return self.cfg.series_length

	@property
	def output_series_length(self):
		return self.cfg.series_length

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