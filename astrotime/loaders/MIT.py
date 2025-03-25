import time, os, numpy as np, xarray as xa
from astrotime.loaders.base import IterativeDataLoader
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.util.logging import exception_handled
import pandas as pd
from glob import glob
from omegaconf import DictConfig, OmegaConf
from astrotime.util.series import TSet
import logging

class MITLoader(IterativeDataLoader):

	def __init__(self, cfg: DictConfig ):
		super().__init__()
		self.cfg = cfg
		self.sector_range = cfg.sector_range
		self.max_period = cfg.max_period
		self.current_sector = None
		self.sector_batch_offset = None
		self.dataset: Optional[xa.Dataset] = None
		self.tset: TSet = None
		self._TICS = None

	def initialize(self, tset: TSet):
		self.tset = tset
		self.sector_batch_offset = 0
		self.current_sector = self.sector_range[0] if tset == TSet.Train else self.sector_range[1]

	@exception_handled
	def get_next_batch( self ) -> xa.Dataset:
		result, t0 = None, time.time()
		if self.current_sector >= 0:
			self.load_sector(self.current_sector)
			batch_end = min( self.sector_batch_offset+self.cfg.batch_size, self.dataset.sizes['elem'])
			result = self.dataset.isel( elem=slice(self.sector_batch_offset,batch_end))  # get_largest_block
			if batch_end == self.dataset.sizes['elem']:
				if self.tset == TSet.Validation:
					self.current_sector = -1
				else:
					self.sector_batch_offset = 0
					self.current_sector = self.current_sector+1
					if self.current_sector == self.sector_range[1]:
						self.current_sector = -1
	#	self.log.info( f" ----> BATCH-{sector_index}.{batch_index}: bstart={bstart}, batch_size={self.cfg.batch_size}, batches_per_file={self.batches_per_sector}, y{result['y'].shape} t{result['t'].shape} p{result['p'].shape}, dt={time.time()-t0:.4f} sec")
		return result

	@property
	def batch_size(self) -> int:
		return self.cfg.batch_size

	def get_period_range(self, sector_index: int ) -> Tuple[float,float]:
		self.load_sector(sector_index)
		periods: List[float] = []
		for elem, TIC in enumerate( self.TICS(sector_index) ):
			dvar: xa.DataArray = self.dataset.data_vars[TIC + ".y"]
			periods.append(dvar.attrs["period"])
		period = np.array(periods)
		pmin, pmax = period.min(), period.max()
		print(f"\n ** periods: range=({pmin:.2f},{pmax:.2f}) median={np.median(period):.2f}")
		return pmin, pmax

	@property
	def dset_idx(self) -> int:
		return self.current_sector

	def TICS( self, sector_index: int ) -> List[str]:
		self.load_sector(sector_index)
		return self._TICS

	def _read_TICS(self, sector_index: int ):
		bls_dir = f"{self.cfg.dataset_root}/sector{sector_index}/bls"
		files = glob("*.bls", root_dir=bls_dir )
		print( f"Get TICS from {bls_dir}, nfiles: {len(files)}")
		self._TICS = [ f.split('.')[0] for f in files ]

	def bls_file_path( self, sector_index: int, TIC: str ) -> str:
		return f"{self.cfg.dataset_root}/sector{sector_index}/bls/{TIC}.bls"

	def lc_file_path( self, sector_index: int, TIC: str ) -> str:
		return f"{self.cfg.dataset_root}/sector{sector_index}/lc/{TIC}.txt"

	def cache_path( self, sector_index: int ) -> str:
		os.makedirs(self.cfg.cache_path, exist_ok=True)
		return f"{self.cfg.cache_path}/sector-{sector_index}.nc"

	def _load_cache_dataset( self, sector_index ):
		t0 = time.time()
		if self.current_sector != sector_index:
			self.refresh()
		if self.dataset is None:
			self.current_sector = sector_index
			dspath: str = self.cache_path(sector_index)
			if os.path.exists(dspath):
				self.dataset = xa.open_dataset( dspath, engine="netcdf4" )
				print( f"Opened cache dataset from {dspath} in in {time.time()-t0:.3f} sec, nvars = {len(self.dataset.data_vars)}")
			else:
				print( f"Cache file not found: {dspath}")
		return self.dataset

	def size(self, sector_index) -> int:
		self.load_sector(sector_index)
		return len(self.dataset.data_vars)

	def nelements(self, tset: TSet = TSet.Train) -> int:
		return self.size(self.current_sector)

	def load_sector( self, sector: int ):
		t0 = time.time()
		if (self.current_sector != sector) or (self.dataset is None):
			self._read_TICS(sector)
			self._load_cache_dataset(sector)
			if self.dataset is None:
				xarrays: Dict[str,xa.DataArray] = {}
				for iT, TIC in enumerate(self._TICS):
					if iT % 50 == 0: print(".",end="",flush=True)
					data_file = self.bls_file_path(sector,TIC)
					dfbls = pd.read_csv( data_file, header=None, names=['Header', 'Data'] )
					dfbls = dfbls.set_index('Header').T
					period: float = np.float64(dfbls['per'].values[0])
					sn: float = np.float64(dfbls['sn'].values[0])
					dflc = pd.read_csv( self.lc_file_path(sector,TIC), header=None, sep='\s+')
					xarrays[ TIC + ".time" ] = xa.DataArray( dflc[0].values, dims=TIC+".obs" )
					xarrays[ TIC + ".y" ]    = xa.DataArray( dflc[1].values, dims=TIC+".obs", attrs=dict(sn=sn,period=period) )
				self.dataset = xa.Dataset( xarrays )
				t1 = time.time()
				print(f" Loaded files in {t1-t0:.3f} sec")
				self.dataset.to_netcdf( self.cache_path(sector), engine="netcdf4" )
				print(f" Saved files in {time.time()-t1:.3f} sec")

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
			print( f" get_largest_block({TIC}): zblocks{len(zblocks)} bsizes{bsizes.shape} idx_largest_block={idx_largest_block}")
			bz: np.array = zblocks[ idx_largest_block ]
		return bz

	def get_training_data(self, sector_index: int) -> np.ndarray:
		TICs: List[str] = self.TICS( sector_index )
		elems = []
		for TIC in TICs:
			cy: xa.DataArray = self.dataset[TIC + ".y"]
			p = cy.attrs["period"]
			if p <= self.max_period:
				bz: np.ndarray = self.get_largest_block(TIC)
				elems.append(bz)
		z = np.stack(elems,axis=0)
		fdropped = (z.shape[0]-len(TICs))/len(TICs)
		print( f"get_training_data({sector_index}): z{z.shape}, max_period={self.max_period:.2f}, dropped {fdropped*100:.2f}%")
		return z

	def refresh(self):
		self.dataset = None

	def get_dataset( self, sector: int ) -> xa.Dataset:
		self.load_sector( sector )
		return self.dataset

	def get_element(self, sector: int, element_index ) -> xa.DataArray:
		self.load_sector(sector)
		elements: List[xa.DataArray] = list(self.dataset.data_vars.values())
		return elements[element_index]