from typing import Any, Dict, List, Optional
import tensorflow as tf


class Encoder:

	def __init__(self, device: str):
		self.device = tf.device(device)

	def encode_dset(self, dset: Dict[str,tf.Tensor]) -> tf.Tensor:
		raise NotImplementedError()