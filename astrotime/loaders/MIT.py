import time, os, math, numpy as np, xarray as xa, random
from astrotime.loaders.base import IterativeDataLoader, RDict
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
		self.sector_index = -1
		self.sector_batch_offset = None
		self.sector_shuffle = list( range(self.sector_range[0],self.sector_range[1]) )
		self.snr_min = cfg.get('snr_min',0.0)
		self.snr_max = cfg.get('snr_max', 1e6)
		self.period_range: Tuple[float,float] = None
		self.current_sector = None
		self.loaded_sector = None
		self.dataset: Optional[xa.Dataset] = None
		self.train_data: Dict[str,np.ndarray] = {}
		self.synthetic = PlanetCrossingDataGenerator(cfg)
		self.tset: TSet = None
		self.test_mode_index = self.TestModes.index( cfg.test_mode )
		self.refresh = kwargs.get('refresh',cfg.refresh)
		self._TICS = None

	def initialize(self, tset: TSet, **kwargs ):
		self.tset = tset
		self.period_range = self.get_period_range()
		self._read_TICS(self.current_sector)
		self.init_epoch()

	def init_epoch(self):
		self.sector_index = -1
		self.sector_batch_offset = None
		random.shuffle(self.sector_shuffle)

	def update_test_mode(self):
		self.test_mode_index = (self.test_mode_index + 1) % len(self.TestModes)

	def get_period_range(self) -> Tuple[float,float]:
		f0 = self.cfg.base_freq
		f1 = f0 + f0 * 2**self.cfg.noctaves
		return 1/f1, 1/f0

	@property
	def ndsets(self) -> int:
		return self.sector_range[1]-self.sector_range[0] + 1

	def get_next_batch( self ) -> Optional[RDict]:
		if self.sector_batch_offset is None:
			self.sector_batch_offset = 0
			if self.tset == TSet.Validation:
				self.current_sector = -1
			else:
				self.sector_index = self.sector_index + 1
				if self.sector_index == len(self.sector_shuffle):
					raise StopIteration
				self.current_sector = self.sector_shuffle[self.sector_index]
				self.log.info(f"Init Dataset: sector={self.current_sector}, sector_batch_offset={self.sector_batch_offset}")

		if self.current_sector >= 0:
			self.load_sector(self.current_sector)
			result: RDict = self.get_training_batch( self.sector_batch_offset )
			self.sector_batch_offset = result.pop('batch_end')+1
			if self.sector_batch_offset >= len(self._TICS):
				self.sector_batch_offset = None
			if self.test_mode_index == 2:
				result = self.synthetic.process_batch( result, **self.params )
			return result
		return None

	def get_element( self, sector_index: int, element_index: int ) -> Optional[Dict[str,Union[np.ndarray,float]]]:
		self.load_sector(sector_index)
		element_data = self.get_training_element( element_index )
		if   self.test_mode_index == 1:
			element_data['y'] = np.sin( 2*np.pi*element_data['t'] / element_data['p'] )
		elif   self.test_mode_index == 2:
			element_data['y'] = self.synthetic.signal(element_data['t'], element_data['p'], **self.params)
		return element_data

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
		if self.loaded_sector != sector_index: self.dataset = None
		if self.dataset is None:
			dspath: str = self.cache_path(sector_index)
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
		t0 = time.time()
		if (self.loaded_sector != sector) or (self.dataset is None):
			self._read_TICS(sector)
			print(f" Loading sector {sector}, loaded_sector={self.loaded_sector}, #TICS={len(self._TICS)}, refresh={self.refresh}")
			if self.refresh: self.dataset = None
			else:       self._load_cache_dataset(sector)
			if self.dataset is None:
				ymax = 0.0
				elems = []
				for iT, TIC in enumerate(self._TICS):
					data_file = self.bls_file_path(sector,TIC)
					lc_file = self.lc_file_path(sector, TIC)
					dfbls = pd.read_csv( data_file, header=None, names=['Header', 'Data'] )
					dfbls = dfbls.set_index('Header').T
					period: float = np.float64(dfbls['per'].values[0])
					if os.path.exists(lc_file):
						sn: float = np.float64(dfbls['sn'].values[0])
						dflc = pd.read_csv( lc_file, header=None, sep='\s+')
						nan_mask = ~np.isnan(dflc[1].values)
						t, y = dflc[0].values[nan_mask], dflc[1].values[nan_mask]
						if y.size > 0:
							ym = y.max()
							if ym > ymax: ymax = ym
							signal = dict( t=xa.DataArray( name=TIC + ".time", data=t, dims=TIC+".obs" ),
										   y = xa.DataArray( name=TIC + ".y", data=y, dims=TIC+".obs", attrs=dict(sn=sn,period=period) ) )
							elems.append( (int(y.shape[0]),signal,TIC) )

				xarrays: Dict[str, xa.DataArray] = {}
				elems.sort(key=lambda x: x[0])
				for elem in elems:
					edata = elem[1]
					xarrays[ edata['y'].name ] = edata['y']
					xarrays[ edata['t'].name ] = edata['t']
				self.dataset = xa.Dataset( xarrays, attrs=dict(ymax=ymax) )
				self._TICS = [ elem[2] for elem in elems ]
				t1 = time.time()
				self.log.info(f" Loaded sector {sector} files in {t1-t0:.3f} sec")
				self.dataset.to_netcdf( self.cache_path(sector), engine="netcdf4" )
			self.loaded_sector = sector
			return True
		return False

	def get_elem_slice(self, ielem: int, series_length: int = -1 ) -> Optional[Tuple[np.ndarray,float,float,str]]:
		TIC = self._TICS[ielem]
		dst: xa.DataArray = self.dataset[TIC+".time"]
		dsy: xa.DataArray = self.dataset[TIC+".y"]
		period = dsy.attrs["period"]
		snr = dsy.attrs["sn"]
		nanmask = ~np.isnan(dsy.values)
		ct, cy = dst.values[nanmask], dsy.values[nanmask]
		cz: np.ndarray = np.stack([ct,cy],axis=0)
		if series_length == -1:
			series_length = cz.shape[1]
		if not self.in_range(period):
			print(f"Dropping elem-{TIC}: period={period} out of range={self.period_range}")
			return None
		elif (snr<self.snr_min) or (snr>self.snr_max):
			return None
		else:
			TD = ct[-1] - ct[0]
			if period > TD:
				print(f"Dropping elem-{TIC}: period={period:.3f} > TD={TD:.3f}, maxP={self.period_range[1]:.3f}, series_length={series_length}")
				return None
			else:
				# if 2*period > TD:
				# 	peak_idx: int = np.argmin(cy)
				# 	TP = ct[peak_idx] - ct[0]
				# 	i0 = 0 if (TP > period) else min( max( peak_idx - 10, 0 ), ct.shape[0]-series_length )
				# else:
				self.log.debug(f"Elem-{ielem}: series_length={series_length}, ct.shape[0]={ct.shape[0]}, period={period}, TD={TD}")
				i0: int = random.randint(0, ct.shape[0]-series_length)
				elem: np.ndarray = cz[:,i0:i0+series_length]
				return elem, period, snr, TIC

	def in_range(self, p: float) -> bool:
		if self.period_range is None: return True
		return (p >= self.period_range[0]) and (p <= self.period_range[1])

	def get_training_batch(self, batch_start: int) -> Dict[str,np.ndarray]:
		self.log.info(f"\nupdate_training_data(sector={self.loaded_sector}), period_range={self.period_range}\n")
		elems, ielem, series_length = [], 0, -1
		periods, sns, tics  = [], [], []
		self.log.debug(f"get_training_batch({batch_start})")
		for ielem in range(batch_start,len(self._TICS)):
			eslice = self.get_elem_slice(ielem,series_length)
			if eslice is not None:
				elem, period, sn, TIC = eslice
				elems.append(elem)
				periods.append(period)
				sns.append(sn)
				tics.append(TIC)
				series_length = elem.shape[1]
			if len(elems) >= self.cfg.batch_size: break
		z = np.stack(elems,axis=0)
		train_data = dict( batch_end=ielem, t=z[:,0,:], y = z[:,1,:], period = np.array(periods), sn = np.array(sns), sector=self.current_sector, TICS=np.array(tics) )
		self.log.info( f"get_training_batch({batch_start}), t{train_data['t'].shape}, y{train_data['y'].shape}, p{train_data['period'].shape}")
		return train_data

	def get_training_element(self, element_index: int) -> Dict[str,np.ndarray]:
		self.log.info(f"\nupdate_training_data(sector={self.loaded_sector}), period_range={self.period_range}\n")
		TIC = self._TICS[element_index]
		eslice = self.get_elem_slice(TIC)
		train_data = None
		if eslice is not None:
			elem, period, sn = eslice
			train_data = dict( t=elem[0], y=elem[1], period=period, sn=sn, sector=self.current_sector, TIC=TIC)
		return train_data


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


