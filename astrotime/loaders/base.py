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

class Loader:

	def __init__(self, cfg: DictConfig, **kwargs ):
		self.log = logging.getLogger()
		self.cfg = cfg


	def init_epoch(self, tset: TSet = TSet.Train ):
		pass

	def initialize(self):
		pass

	def get_next_batch( self ) -> Optional[RDict]:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_next_batch' method")

	def add_octave_data(self, octave_data: List[Tuple[int,int,int]] ):
		pass

class ElementLoader(Loader):

	def __init__(self, cfg: DictConfig, **kwargs ):
		super().__init__(cfg, **kwargs)
		self.rootdir = cfg.dataset_root
		self.dset = cfg.source
		self.ifile: int = kwargs.get('file',0)
		self.file_index = -1
		self.data = None
		self.tset: TSet = None
		self.batch_offset = 0
		self.params: Dict[str, float] = {}

	def set_params(self, params: Dict[str, float]):
		self.params = params

	@property
	def file_size(self):
		return self.cfg.file_size

	@property
	def nfiles(self):
		return self.cfg.nfiles

	@property
	def ntfiles(self):
		return self.cfg.nfiles-1

	def init_epoch(self, tset: TSet = TSet.Train):
		self.ifile = 0
		self.data = None
		self.batch_offset = 0
		self.set_tset(tset)
		self.load_data()

	def set_tset(self, tset: TSet):
		self.tset = tset
		return self

	def set_file(self, file_idx: int):
		if file_idx != self.ifile:
			self.ifile = file_idx
			self.data = None

	def load_data(self):
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'load_data' method")

	@property
	def nelements(self) -> int:
		return self.nfiles * self.file_size

	@property
	def nelem(self):
		return self.file_size

	def get_element( self, elem_index: int ) -> Optional[RDict]:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'load_element' method")

	def get_next_batch( self ) -> Optional[RDict]:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_next_batch' method")

class IterativeDataLoader(Loader):

	def __init__(self, cfg: DictConfig,  **kwargs):
		super().__init__(cfg, **kwargs)
		self.params: Dict[str,float] = {}

	def set_params(self, params: Dict[str,float] ):
		self.params = params

	def get_dataset( self, *args ) -> xa.Dataset:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_dataset' method")

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
	def nfiles(self) -> int:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'nfiles' property")
