from typing import List, Tuple, Mapping, Optional
import logging, torch, numpy as np
from omegaconf import DictConfig
from torch import Tensor, device
from astrotime.trainers.filters import TrainingFilter
from astrotime.trainers.filters import RandomDownsample

class Transform(torch.nn.Module):

	def __init__(self, name: str, cfg: DictConfig, device: device):
		torch.nn.Module.__init__(self)
		self.name = name
		self.requires_grad_(False)
		self.device = device
		self.cfg: DictConfig = cfg
		self.log = logging.getLogger()

	def process_event(self, **kwargs ):
		pass

	@property
	def xdata(self, **kwargs ) -> np.ndarray:
		raise NotImplementedError("Transform.xdata not implemented")

	def embed(self, xs: Tensor, ys: Tensor, **kwargs) -> Tensor:
		raise NotImplementedError("Transform.embed() not implemented")

	def magnitude(self, embedding: Tensor) -> np.ndarray:
		raise NotImplementedError("Transform.magnitude() not implemented")

class Encoder:

	def __init__(self, cfg: DictConfig, device: device  ):
		self.device: device = device
		self.cfg = cfg
		self.filters: List[TrainingFilter] = []
		self.log = logging.getLogger()
		self.time_scale = cfg.time_scale
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

	def encode_batch(self, x: np.ndarray, y: np.ndarray) -> Tuple[Tensor, Tensor]:
		raise NotImplementedError()

	def encode_periodic_batch(self, batch: Mapping[str,np.ndarray]) -> Optional[Mapping[str,np.ndarray]]:
		raise NotImplementedError()

	def add_filter(self, tfilter: TrainingFilter ):
		self.filters.append( tfilter )

	def apply_filters(self, x: np.ndarray, y: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
		for f in self.filters:
			x, y = f.apply( x, y, dim )
		return x, y