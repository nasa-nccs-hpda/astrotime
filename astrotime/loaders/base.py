import numpy as np
from typing import List, Optional, Dict, Type

class DataLoader:

	def __init__(self):
		pass

	def get_dataset( self, dset_idx: int ) -> Dict[ str, np.ndarray]:
		raise NotImplementedError()