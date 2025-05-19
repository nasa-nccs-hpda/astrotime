import time, os, math, numpy as np, xarray as xa, random
from astrotime.loaders.base import IterativeDataLoader, RDict
from astrotime.loaders.pcross import PlanetCrossingDataGenerator
from typing import List, Optional, Dict, Type, Union, Tuple
import pandas as pd
from glob import glob
from omegaconf import DictConfig, OmegaConf
from astrotime.util.series import TSet

class SyntheticLoader(IterativeDataLoader):

	def __init__(self, cfg: DictConfig, **kwargs ):
		super().__init__()
		self.cfg = cfg
		self.files = None
		self.nfiles = None
		self.sector_index = 0
		self.sector_shuffle = None
		self.series_length = cfg.series_length
		self.loaded_sector = None
		self.sector_batch_offset = None
		self.dataset: Optional[xa.Dataset] = None
		self.train_data: Dict[str,np.ndarray] = {}
		self.tset: TSet = None
		self._nbatches = -1

	@property
	def current_sector(self):
		return self.sector_shuffle[self.sector_index]

	def initialize(self, tset: TSet, **kwargs ):
		self.tset = tset
		self.files = glob(f"{self.cfg.source}_*.nc", root_dir=self.cfg.dataset_root)
		self.nfiles = len(self.files)
		self.sector_shuffle = list(range(self.nfiles))
		self.init_epoch()

	def init_epoch(self):
		self.sector_batch_offset = 0
		self.sector_index = 0
		self._nbatches = -1
		random.shuffle(self.sector_shuffle)

	def get_next_batch( self ) -> Optional[RDict]:
		ibatch = self.sector_batch_offset//self.cfg.batch_size
		if (self._nbatches > 0) and ( ibatch>= self._nbatches-1):
			self.sector_index = self.sector_index + 1
			if self.sector_index == self.nfiles:
				raise StopIteration
			self.sector_batch_offset = 0
			self.log.info(f"Init Dataset: sector={self.current_sector}, sector_batch_offset={self.sector_batch_offset}")

		if self.current_sector >= 0:
			self.load_sector(self.current_sector)
			batch_start = self.sector_batch_offset
			batch_end   = batch_start+self.cfg.batch_size
			result: RDict = { k: self.train_data[k][batch_start:batch_end] for k in ['t','y','p','sn'] }
			result['offset'] = batch_start
			result['sector'] = self.current_sector
			self.sector_batch_offset = batch_end
			return result
		return None

	def get_batch( self, sector_index: int, batch_index: int ) -> Optional[Dict[str,np.ndarray]]:
		self.load_sector(sector_index)
		batch_start = self.sector_batch_offset*batch_index
		batch_end   = batch_start+self.cfg.batch_size
		result = { k: self.train_data[k][batch_start:batch_end] for k in ['t','y','p','sn'] }
		result['offset'] = batch_start
		result['sector'] = self.current_sector
		return result

	def get_element( self, sector_index: int, element_index: int ) -> Optional[Dict[str,Union[np.ndarray,float]]]:
		self.load_sector(sector_index)
		element_data = { k: self.train_data[k][element_index] for k in ['t','y','p','sn'] }
		return element_data

	@property
	def nbatches(self) -> int:
		ne = self.nelements
		return -1 if (ne == -1) else ne//self.cfg.batch_size

	@property
	def nelements(self) -> int:
		if self.current_sector >= 0:
			self.load_sector(self.current_sector)
			return self.train_data['t'].shape[0]
		return -1

	@property
	def batch_size(self) -> int:
		return self.cfg.batch_size

	@property
	def dset_idx(self) -> int:
		return self.current_sector

	def file_path( self, sector_index: int ) -> str:
		return f"{self.cfg.dataset_root}/{self.cfg.source}_{sector_index}.nc"

	def _load_cache_dataset( self, sector_index ):
		t0 = time.time()
		if self.loaded_sector != sector_index:
			self.refresh()
		if self.dataset is None:
			dspath: str = self.file_path(sector_index)
			if os.path.exists(dspath):
				self.dataset = xa.open_dataset( dspath, engine="netcdf4" )
				self.log.info( f"Opened cache dataset from {dspath} in in {time.time()-t0:.3f} sec, nvars = {len(self.dataset.data_vars)}")
			else:
				self.log.info( f"Cache file not found: {dspath}")
		return self.dataset

	def size(self, sector_index) -> int:
		self.load_sector(sector_index)
		return len(self.dataset.data_vars)

	def get_dataset(self, dset_idx: int) -> xa.Dataset:
		self.load_sector(dset_idx)
		return self.dataset

	def load_sector( self, sector: int, **kwargs ) -> bool:
		if (self.loaded_sector != sector) or (self.dataset is None):
			self._load_cache_dataset(sector)
			self.loaded_sector = sector
			self.update_training_data()
			return True
		return False

	def get_elem_slice(self,ielem: int):
		cy: xa.DataArray = self.dataset[f"s0{ielem}"]
		ct: xa.DataArray = self.dataset[f"t0{ielem}"]
		cz = np.stack([ct,cy],axis=0)
		elem = cz[:,:self.series_length] if (cz.shape[1] >= self.series_length) else None
		period = cy.attrs["period"]
		stype = cy.attrs["type"]
		return elem, period, stype

	def get_largest_block( self, TIC: str ) -> np.ndarray:
		threshold = self.cfg.block_gap_threshold
		ctime: np.ndarray = self.dataset[TIC+".time"].values.squeeze()
		cy: np.ndarray = self.dataset[TIC+".y"].values.squeeze()
		diff: np.ndarray = np.diff(ctime)
		break_indices: np.ndarray = np.nonzero(diff > threshold)[0]
		cz = np.stack([ctime,cy],axis=0)
		if break_indices.size == 0:
			bz = cz
		elif break_indices.size == 1:
			bz = cz[:,0:break_indices[0]] if (break_indices[0] >= ctime.size//2) else cz[:,break_indices[0]:]
		else:
			zblocks: List[np.ndarray] = np.array_split(cz, break_indices,axis=1)
			bsizes: np.array = np.array([break_indices[0]] + np.diff(break_indices).tolist() + [ctime.size - break_indices[-1]])
			idx_largest_block: np.ndarray = np.argmax(bsizes)
			bz: np.array = zblocks[ idx_largest_block ]
		return bz

	def get_batch_element(self, bz: np.ndarray) -> np.ndarray:
		center = bz.shape[1] // 2
		bdata = bz[:,center-self.series_length//2:center+self.series_length//2]
		return bdata

	def update_training_data(self):
		self.log.info(f"update_training_data(sector={self.loaded_sector})")
		periods, stypes, elems  = [], [], []
		for ielem in range(self.cfg.file_size):
			eslice, period, stype = self.get_elem_slice(ielem)
			if eslice is not None:
				elems.append(eslice)
				periods.append(period)
				stypes.append(stype)
		z = np.stack(elems,axis=0)
		self.train_data['t'] = z[:,0,:]
		self.train_data['y'] = z[:,1,:]
		self.train_data['period'] = np.array(periods)
		self.train_data['stype'] = np.array(stypes)
		self._nbatches = math.ceil( self.train_data['t'].shape[0] / self.cfg.batch_size )

	def refresh(self):
		self.dataset = None

class SyntheticOctavesLoader(SyntheticLoader):

	def __init__(self, cfg: DictConfig, **kwargs ):
		super().__init__(cfg, **kwargs)
		self.nfreq: int = cfg.series_length
		self.base_freq: float = cfg.base_freq
		self.noctaves: int = cfg.noctaves


