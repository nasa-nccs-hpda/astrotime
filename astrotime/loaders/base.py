import numpy as np, xarray as xa
from astrotime.util.series import TSet
from typing import List, Optional, Dict, Type, Tuple, Union
import logging
from omegaconf import DictConfig, OmegaConf

class DataLoader:

	def __init__(self):
		self.log = logging.getLogger()

	def get_dataset( self, dset_idx: int ) -> xa.Dataset:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_dataset' method")

	def get_batch(self, tset: TSet, batch_index: int) -> xa.Dataset:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_batch' method")

	def get_element(self, dset_idx: int, element_index) -> Optional[Dict[str,Union[np.ndarray,float]]]:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_element' method")

	def get_dataset_element(self, dset_idx: int, element_index, **kwargs) -> xa.Dataset:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_dataset_element' method")

	def nbatches(self, tset: TSet) -> int:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'nbatches' method")

	@property
	def batch_size(self) -> int:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'batch_size' property")

	@property
	def dset_idx(self) -> int:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'dset_idx' property")

	@property
	def nelements( self ) -> int:
		return self.nbatches(TSet.Train) * self.batch_size

RDict = Dict[str,Union[List[str],int,np.ndarray]]

class ElementLoader:

	def __init__(self, cfg: DictConfig, archive: int=0, **kwargs ):
		super().__init__()
		self.log = logging.getLogger()
		self.cfg = cfg
		self.rootdir = cfg.dataset_root
		self.dset = cfg.source
		self.files_per_archive: int = cfg.files_per_archive
		self.archive: int = archive
		self.data = None

	def set_archive(self, archive: int):
		if archive != self.archive:
			self.archive = archive
			self.data = None

	def load_data(self):
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'load_data' method")

	@property
	def nelem(self):
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'nelem' method")

	def load_element( self, elem_index: int ) -> Optional[RDict]:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'load_element' method")


class IterativeDataLoader:

	def __init__(self):
		self.log = logging.getLogger()
		self.params: Dict[str,float] = {}

	def set_params(self, params: Dict[str,float] ):
		self.params = params

	def get_dataset( self, *args ) -> xa.Dataset:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_dataset' method")

	def initialize(self, tset: TSet) -> xa.Dataset:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'initialize' method")

	def init_epoch(self):
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'init_epoch' method")

	def get_next_batch(self) -> Optional[RDict]:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_next_batch' method")

	def get_batch( self, dset_idx: int, batch_index ) -> Optional[RDict]:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_batch' method")

	def get_element(self, dset_idx: int, element_index, **kwargs) -> Optional[RDict]:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_element' method")

	def get_single_element(self, dset_idx: int, element_index, **kwargs) -> Optional[RDict]:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_single_element' method")

	def update_test_mode(self):
		pass

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
