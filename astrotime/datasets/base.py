import numpy as np, xarray as xa
from astrotime.util.series import TSet
from typing import List, Optional, Dict, Type, Tuple, Union
import logging, random
from glob import glob
from omegaconf import DictConfig, OmegaConf
RDict = Dict[str,Union[List[str],int,np.ndarray]]
from torch.utils.data import IterableDataset, get_worker_info

class AstrotimeDataset(IterableDataset):

	def __init__(self, cfg: DictConfig ):
		super(AstrotimeDataset).__init__()
		self.log = logging.getLogger()
		self.cfg = cfg
		self.num_workers: int = 1
		self.ifile: int = -1
		self.ielement = -1
		self.worker_idx: int = 0
		self.tset: TSet = None
		self.rootdir: str = cfg.dataset_root
		self.nfiles: int = cfg.nfiles
		self._files: List[str] = None
		self.file_sort: List[int] = None

	def get_file_sort(self, tset: TSet):
		if   tset == TSet.Train:      flist = list(range(self.nfiles-1))
		elif tset == TSet.Validation: flist = [self.nfiles]
		elif tset == TSet.Update:     flist = list(range(self.nfiles))
		else: raise ValueError(f"Unknown tset: {tset}")
		return flist[self.worker_idx::self.num_workers]

	def update_element(self):
		self.ielement += 1
		if self.ielement >= self.nelem_in_file():
			self.update_file()
			self.ielement = 0

	def update_file(self):
		self.ifile += 1
		if self.ifile >= len(self.file_sort):
			raise StopIteration
		self._load_next_file()

	@property
	def file_paths( self ) -> List[str]:
		if self._files is None:
			self._files = glob( self.cfg.dataset_files, root_dir=self.rootdir )
		return self._files

	def update_worker_info(self):
		worker_info = get_worker_info()
		self.num_workers = worker_info.num_workers
		self.worker_idx = worker_info.id

	def __iter__(self):
		self.update_worker_info()
		self.initialize()
		return self

	def __next__(self) -> RDict:
		element = self.get_next_element()
		if element is None:
			raise StopIteration
		return element

	def init_epoch(self, tset: TSet = TSet.Train):
		self.ifile = -1
		self.ielement = -1
		self.tset = tset
		self.file_sort = self.get_file_sort(tset)
		random.shuffle(self.file_sort)
		self.update_file()

	def initialize(self):
		pass

	def get_next_element( self ) -> Optional[RDict]:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_next_batch' method")

	def _load_next_file( self ):
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the '_load_cache_dataset' method")

	def nelem_in_file(self):
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'nelem_in_file' method")




