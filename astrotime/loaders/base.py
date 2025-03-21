import numpy as np, xarray as xa
from astrotime.util.series import TSet
from typing import List, Optional, Dict, Type, Tuple
import logging
log = logging.getLogger("astrotime")

class DataLoader:

	def __init__(self):
		pass

	def get_dataset( self, dset_idx: int ) -> xa.Dataset:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_dataset' method")

	def get_batch(self, tset: TSet, batch_index: int) -> xa.Dataset:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_batch' method")

	def get_element(self, dset_idx: int, element_index) -> xa.DataArray:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_batch' method")

	def nbatches(self, tset: TSet) -> int:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'nbatches' method")

	@property
	def batch_size(self) -> int:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'batch_size' property")

	@property
	def dset_idx(self) -> int:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'dset_idx' property")

	def nelements( self, tset: TSet=TSet.Train ) -> int:
		return self.nbatches(tset) * self.batch_size
