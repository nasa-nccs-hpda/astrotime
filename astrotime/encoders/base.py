from typing import Any, Dict, List, Optional
import tensorflow as tf, numpy as np


class Encoder:

	def __init__(self, device: str):
		self.device = tf.device(device)

	def encode_dset(self, dset: Dict[str,np.ndarray]) -> tf.Tensor:
		raise NotImplementedError()