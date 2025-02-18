from typing import Any, Dict, List, Optional, Tuple, Mapping
import tensorflow as tf, numpy as np
from astrotime.transforms.filters import TrainingFilter

class Encoder:

	def __init__(self, device: str, series_len: int):
		self.device = tf.device(device)
		self._series_len: int = series_len
		self.filters: List[TrainingFilter] = []

	@property
	def series_len(self):
		return self._series_len

	def encode_dset(self, dset: Mapping[str,np.ndarray]) -> tf.Tensor:
		raise NotImplementedError()

	def encode_batch(self, x: np.ndarray, y: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
		raise NotImplementedError()

	def add_filters(self, filters: List[TrainingFilter] ):
		self.filters.extend( filters )

	def apply_filters(self, x: np.ndarray, y: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
		for f in self.filters:
			x, y = f.apply( x, y, axis )
		return x, y