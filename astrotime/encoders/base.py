from typing import Any, Dict, List, Optional, Tuple, Mapping
import numpy as np
from torch import Tensor, device
from astrotime.transforms.filters import TrainingFilter

class Encoder:

	def __init__(self, device: device, series_len: int):
		self.device: device = device
		self._series_len: int = series_len
		self.filters: List[TrainingFilter] = []

	@property
	def series_len(self):
		return self._series_len

	def encode_dset(self, dset: Mapping[str,np.ndarray]) -> Tensor:
		raise NotImplementedError()

	def encode_batch(self, x: np.ndarray, y: np.ndarray) -> Tuple[Tensor, Tensor]:
		raise NotImplementedError()

	def add_filters(self, filters: List[TrainingFilter] ):
		self.filters.extend( filters )

	def apply_filters(self, x: np.ndarray, y: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
		for f in self.filters:
			x, y = f.apply( x, y, dim )
		return x, y