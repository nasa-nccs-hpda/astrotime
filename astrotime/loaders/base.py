import numpy as np, xarray as xa
from typing import List, Optional, Dict, Type, Tuple
from astrotime.util.logging import lgm, exception_handled, log_timing, shp

class DataLoader:

	def __init__(self):
		pass

	def get_dataset( self, dset_idx: int ) -> Dict[ str, np.ndarray]:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_dataset' method")

	def get_batch(self, batch_index: int) -> xa.Dataset:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_batch' method")

	@property
	def nbatches(self) -> int:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'nbatches' property")

	@property
	def batch_size(self) -> int:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'batch_size' property")

	@property
	def nelements(self) -> int:
		return self.nbatches * self.batch_size
