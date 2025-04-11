import numpy as np, xarray as xa
from astrotime.util.series import TSet
from typing import List, Optional, Dict, Type, Tuple, Union
import logging

class DataLoader:

	def __init__(self):
		self.log = logging.getLogger("astrotime")

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

class IterativeDataLoader:

	def __init__(self):
		self.log = logging.getLogger("astrotime")

	def get_dataset( self, *args ) -> xa.Dataset:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_dataset' method")

	def initialize(self, tset: TSet) -> xa.Dataset:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'initialize' method")

	def get_next_batch(self) -> Optional[Dict[str,np.ndarray]]:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_next_batch' method")

	def get_batch( self, dset_idx: int, batch_index ) -> Optional[Dict[str,np.ndarray]]:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_batch' method")

	def get_element(self, dset_idx: int, element_index) -> Optional[Dict[str,Union[np.ndarray,float]]]:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_element' method")

	def get_dataset_element(self, dset_idx: int, element_index, **kwargs) -> xa.Dataset:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_dataset_element' method")

	@property
	def batch_size(self) -> int:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'batch_size' property")

	@property
	def dset_idx(self) -> int:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'dset_idx' property")

	@property
	def nbatches(self) -> int:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'nbatches' property")

	@property
	def nelements(self) -> int:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'nelements' property")

	@property
	def ndsets(self) -> int:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'ndsets' property")
