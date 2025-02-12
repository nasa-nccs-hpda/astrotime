from typing import Any, Dict, List, Optional
import numpy as np


class Encoder:

	def __init__(self):
		pass

	def encode_dset(self, batch_data: Dict[str,np.ndarray]) -> np.ndarray:
		raise NotImplementedError()