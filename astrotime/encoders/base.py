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

	def apply_filters(self, x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
		for f in self.filters:
			x, y = f.apply( x, y )
		return x, y