from typing import Any, Dict, List, Optional, Tuple
import tensorflow as tf, numpy as np
from astrotime.transforms.filters import TrainingFilter


class Encoder:

	def __init__(self, device: str):
		self.device = tf.device(device)
		self.filters: List[TrainingFilter] = []

	def encode_dset(self, dset: Dict[str,np.ndarray]) -> tf.Tensor:
		raise NotImplementedError()

	def add_filters(self, filters: List[TrainingFilter] ):
		self.filters.extend( filters )

	def apply_filters(self, x: np.ndarray, y: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
		for f in self.filters:
			x, y = f.apply( x, y, axis )
		return x, y