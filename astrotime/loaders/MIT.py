import time, os, numpy as np, xarray as xa
from astrotime.loaders.base import IterativeDataLoader
from astrotime.loaders.pcross import PlanetCrossingDataGenerator
from typing import List, Optional, Dict, Type, Union, Tuple
import pandas as pd
from enum import Enum
from glob import glob
from omegaconf import DictConfig, OmegaConf
from astrotime.util.series import TSet


class MITLoader(IterativeDataLoader):

	TestModes: List = [ "default", 'sinusoid', 'planet_crossing' ]

	def __init__(self, cfg: DictConfig, **kwargs ):
		super().__init__()
		self.cfg = cfg
		self.sector_range = cfg.sector_range
		self.series_length = cfg.series_length
		self.period_range: Optional[Tuple[float,float]] = None
		self.current_sector = None
		self.sector_batch_offset = None
		self.dataset: Optional[xa.Dataset] = None
		self.train_data: Dict[str,np.ndarray] = {}
		self.synthetic = PlanetCrossingDataGenerator(cfg)
		self.tset: TSet = None
		self._nbatches = -1
		self.test_mode_index = self.TestModes.index( cfg.test_mode )
		self.ymax = None
		self._TICS = None

	def initialize(self, tset: TSet, **kwargs ):
		self.tset = tset
		self.period_range =  self.get_period_range()
		self._read_TICS(self.current_sector)
		self.init_epoch()

	def init_epoch(self):
		self.sector_batch_offset = 0
		self.current_sector = self.sector_range[0] if self.tset == TSet.Train else self.sector_range[1]
		self._nbatches = -1

	def update_test_mode(self):
		self.test_mode_index = (self.test_mode_index + 1) % len(self.TestModes)

	def get_period_range(self) -> Optional[Tuple[float,float]]:
		max_period = self.cfg.get('max_period',None)
		if max_period is not None:
			return 0, max_period
		return 0, float('inf')

	@property
	def ndsets(self) -> int:
		return self.sector_range[1]-self.sector_range[0] + 1

	def get_next_batch( self ) -> Optional[Dict[str,np.ndarray]]:
		self.log.info(f"MITLoader.get_next_batch: sector = {self.current_sector}, nbatches={self._nbatches}, sector_batch_offset={self.sector_batch_offset}, batch_size={self.cfg.batch_size}")
		if (self._nbatches > 0) and (self.sector_batch_offset//self.cfg.batch_size >= self._nbatches-1):
			if self.tset == TSet.Validation:
				self.current_sector = -1
			else:
				self.current_sector = self.current_sector + 1
				self.sector_batch_offset = 0
				if self.current_sector == self.sector_range[1]:
					self.current_sector = self.sector_range[0]
					return None
				self.log.info( f"Init Dataset: sector={self.current_sector}, sector_range={self.sector_range}, nbatches={self._nbatches} sector_batch_offset={self.sector_batch_offset}")
		if self.current_sector >= 0:
			if self.load_sector(self.current_sector):
				self.update_training_data()
			batch_start = self.sector_batch_offset
			batch_end   = batch_start+self.cfg.batch_size
			result = { k: self.train_data[k][batch_start:batch_end] for k in ['t','y','p'] }
			self.sector_batch_offset = batch_end
			self.log.info( f"  *** get_next_batch: batch({batch_start}->{batch_end}), t{self.train_data['t'].shape}->bt{result['t'].shape}, y{self.train_data['y'].shape}->by{result['y'].shape}, p{self.train_data['p'].shape}->bp{result['p'].shape}  ")
			if self.test_mode_index == 2:
				result = self.synthetic.process_batch( result )
			return result

	def get_batch( self, sector_index: int, batch_index: int ) -> Optional[Dict[str,np.ndarray]]:
		if self.load_sector(sector_index):
			self.update_training_data()
		batch_start = self.sector_batch_offset*batch_index
		batch_end   = batch_start+self.cfg.batch_size
		result = { k: self.train_data[k][batch_start:batch_end] for k in ['t','y','p'] }
		if self.test_mode_index == 2:
			result = self.synthetic.process_batch(result)
		return result

	def get_element( self, sector_index: int, element_index: int ) -> Optional[Dict[str,Union[np.ndarray,float]]]:
		if self.load_sector(sector_index):
			self.update_training_data()
		result = { k: self.train_data[k][element_index] for k in ['t','y','p'] }
		return result

	def get_dataset_element( self, sector_index: int, TIC: str, **kwargs ) -> xa.Dataset:
		self.load_sector(sector_index, **kwargs)
		if     self.test_mode_index == 0: return xa.Dataset( { k: self.dataset.data_vars[TIC+"."+k] for k in ['time','y'] } )
		elif   self.test_mode_index == 1: return self.get_sinusoid_element(sector_index,TIC)
		elif   self.test_mode_index == 2: return self.get_pcross_element(sector_index, TIC)
		else: raise Exception(f"Unknown test mode {self.test_mode_index}")

	def get_sinusoid_element( self, sector_index: int, TIC: str, **kwargs ) -> xa.Dataset:
		self.load_sector(sector_index, **kwargs)
		time: xa.DataArray = self.dataset.data_vars[TIC+".time"]
		y: xa.DataArray = self.dataset.data_vars[TIC+".y"]
		sinusoid: np.ndarray = np.sin( 2*np.pi*time.values / y.attrs["period"] )
		return xa.Dataset( dict(  time=time, y=y.copy( data=sinusoid ) ) )

	def get_pcross_element( self, sector_index: int, TIC: str, **kwargs ) -> xa.Dataset:
		self.load_sector(sector_index, **kwargs)
		time: xa.DataArray = self.dataset.data_vars[TIC+".time"]
		y: xa.DataArray = self.dataset.data_vars[TIC + ".y"]
		signal: xa.DataArray = self.synthetic.get_element( time, y )
		return xa.Dataset( dict( time=time, y=signal ) )

	@property
	def nbatches(self) -> int:
		ne = self.nelements
		return -1 if (ne == -1) else ne//self.cfg.batch_size

	@property
	def nelements(self) -> int:
		if self.current_sector >= 0:
			if self.load_sector(self.current_sector):
				self.update_training_data()
			return self.train_data['t'].shape[0]
		return -1

	@property
	def batch_size(self) -> int:
		return self.cfg.batch_size

	@property
	def dset_idx(self) -> int:
		return self.current_sector

	def TICS( self, sector_index: int ) -> List[str]:
		self.load_sector(sector_index)
		return self._TICS

	def _read_TICS(self, sector_index: int ):
		bls_dir = f"{self.cfg.dataset_root}/sector{sector_index}/bls"
		files = glob("*.bls", root_dir=bls_dir )
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

	def get_dataset(self, dset_idx: int) -> xa.Dataset:
		self.load_sector(dset_idx)
		return self.dataset

	def load_sector( self, sector: int, **kwargs ) -> bool:
		t0 = time.time()
		refresh = kwargs.get('refresh',False)
		if (self.current_sector != sector) or (self.dataset is None):
			print(f"Loading sector {sector}")
			self._read_TICS(sector)
			if refresh: self.refresh()
			else:       self._load_cache_dataset(sector)
			if self.dataset is None:
				xarrays: Dict[str,xa.DataArray] = {}
				ymax = 0.0
				for iT, TIC in enumerate(self._TICS):
					if iT % 50 == 0: print(".",end="",flush=True)
					data_file = self.bls_file_path(sector,TIC)
					dfbls = pd.read_csv( data_file, header=None, names=['Header', 'Data'] )
					dfbls = dfbls.set_index('Header').T
					period: float = np.float64(dfbls['per'].values[0])
					if self.in_range(period):
						sn: float = np.float64(dfbls['sn'].values[0])
						dflc = pd.read_csv( self.lc_file_path(sector,TIC), header=None, sep='\s+')
						nan_mask = ~np.isnan(dflc[1].values)
						t, y = dflc[0].values[nan_mask], dflc[1].values[nan_mask]
						ym = y.max()
						if ym > ymax: ymax = ym
						xarrays[ TIC + ".time" ] = xa.DataArray( t, dims=TIC+".obs" )
						xarrays[ TIC + ".y" ]    = xa.DataArray( y, dims=TIC+".obs", attrs=dict(sn=sn,period=period) )
				self.dataset = xa.Dataset( xarrays, attrs=dict(ymax=ymax) )
				t1 = time.time()
				self.log.info(f" Loaded sector {sector} files in {t1-t0:.3f} sec")
				self.dataset.to_netcdf( self.cache_path(sector), engine="netcdf4" )
			self.ymax = self.dataset.attrs["ymax"]
			return True
		return False

	def get_elem_slice(self,TIC: str):
		ctime: np.ndarray = self.dataset[TIC+".time"].values.squeeze()
		cy: np.ndarray = self.dataset[TIC+".y"].values.squeeze()
		cz = np.stack([ctime,cy],axis=0)
		if cz.shape[1] >= self.series_length:
			return cz[:,:self.series_length]

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

	def in_range(self, p: float) -> bool:
		if self.period_range is None: return True
		return (p >= self.period_range[0]) and (p <= self.period_range[1])

	def update_training_data(self):
		self.log.info("\nupdate_training_data\n")
		elems = []
		periods = []
		for TIC in self._TICS:
			cy: xa.DataArray = self.dataset[TIC + ".y"]
			p = cy.attrs["period"]
			if self.in_range(p):
				# bz: np.ndarray = self.get_largest_block(TIC)
				# if bz.shape[1] >= self.series_length:
				# 	elems.append( self.get_batch_element(bz) )
				# 	periods.append(p)
				eslice = self.get_elem_slice(TIC)
				if eslice is not None:
					elems.append(eslice)
					periods.append(p)
		z = np.stack(elems,axis=0)
		self.train_data['t'] = z[:,0,:]
		self.train_data['y'] = z[:,1,:]
		self.train_data['p'] = np.array(periods)
		fdropped = (len(self._TICS)-z.shape[0])/len(self._TICS)
		self._nbatches = self.train_data['t'].shape[0] // self.cfg.batch_size
		self.log.info( f"get_training_data: nbatches={self._nbatches}, t{self.train_data['t'].shape}, y{self.train_data['y'].shape}, p{self.train_data['p'].shape}, dropped {fdropped*100:.2f}%")

	def refresh(self):
		self.dataset = None


class MITOctavesLoader(MITLoader):

	def __init__(self, cfg: DictConfig, **kwargs ):
		super().__init__(cfg, **kwargs)
		self.nfreq: int = cfg.series_length
		self.base_freq: float = cfg.base_freq
		self.noctaves: int = cfg.noctaves

	def get_period_range(self) -> Optional[Tuple[float,float]]:
		f0 = self.base_freq
		f1 = f0 * pow(2,self.noctaves)
		return 1/f1, 1/f0


