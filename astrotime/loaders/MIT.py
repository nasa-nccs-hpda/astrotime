import time, os, numpy as np, xarray as xa
from astrotime.loaders.base import DataLoader
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.util.logging import exception_handled
import pandas as pd
from glob import glob
from omegaconf import DictConfig, OmegaConf
from astrotime.util.series import TSet
import logging

class MITLoader(DataLoader):

	def __init__(self, cfg: DictConfig ):
		super().__init__()
		self.cfg = cfg
		self.sector_range = cfg.sector_range
		self.current_sector = self.sector_range[0]
		self.dataset: xa.DataSet = None

	@property
	def dset_idx(self) -> int:
		return self.current_sector

	def TICS( self, sector_index: int ) -> List[str]:
		bls_dir = f"{self.cfg.dataset_root}/sector{sector_index}/bls"
		files = glob("*.bls", root_dir=bls_dir )
		print( f"Get TICS from {bls_dir}, nfiles: {len(files)}")
		return [ f.split('.')[0] for f in files ]

	def bls_file_path( self, sector_index: int, TIC: str ) -> str:
		return f"{self.cfg.dataset_root}/sector{sector_index}/bls/{TIC}.bls"

	def lc_file_path( self, sector_index: int, TIC: str ) -> str:
		return f"{self.cfg.dataset_root}/sector{sector_index}/lc/{TIC}.txt"

	def cache_path( self, sector_index: int ) -> str:
		os.makedirs(self.cfg.cache_path, exist_ok=True)
		return f"{self.cfg.cache_path}/sector-{sector_index}.zarr"

	def load_cache_dataset( self, sector_index ) -> Optional[xa.Dataset]:
		t0 = time.time()
		self.current_sector = sector_index
		dspath: str = self.cache_path(sector_index)
		if os.path.exists(dspath):
			result = xa.open_dataset( dspath, engine="zarr" )
			print( f"Opened cache dataset from {dspath} in in {time.time()-t0:.3f} sec, nvars = {len(result.data_vars)}")
			return result
		else:
			print( f"Cache file not found: {dspath}")

	def size(self, sector_index) -> int:
		self.load_sector(sector_index)
		return len(self.dataset.data_vars)

	def nelements(self, tset: TSet = TSet.Train) -> int:
		return self.size(self.current_sector)

	def load_sector( self, sector: int, refresh=False ):
		t0 = time.time()
		if refresh: self.dataset = None
		if (self.current_sector != sector) or (self.dataset is None):
			if not refresh: self.dataset = self.load_cache_dataset(sector)
			if self.dataset is None:
				TICS: List[str] = self.TICS(sector)
				xarrays: Dict[str,xa.DataArray] = {}
				print(f"Loading {len(TICS)} TIC files for sector {sector}:  ",end="")
				for iT, TIC in enumerate(TICS):
					if iT % 100 == 0: print(".",end="",flush=True)
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
				print(f"Loaded sector {sector} files in {t1-t0:.3f} sec")
				self.dataset.to_zarr( self.cache_path(sector) )
				print(f"Saved sector {sector} files in {time.time()-t1:.3f} sec")

	def get_dataset( self, sector: int, refresh=False ) -> xa.Dataset:
		self.load_sector( sector, refresh )
		return self.dataset

	def get_element(self, sector: int, element_index ) -> xa.DataArray:
		self.load_sector(sector)
		elements: List[xa.DataArray] = list(self.dataset.data_vars.values())
		return elements[element_index]