import time, os, math, numpy as np, xarray as xa, random
from astrotime.loaders.base import IterativeDataLoader, RDict
from astrotime.loaders.pcross import PlanetCrossingDataGenerator
from typing import List, Optional, Dict, Type, Union, Tuple
import pandas as pd
from astrotime.loaders.base import ElementLoader, RDict
from glob import glob
from omegaconf import DictConfig, OmegaConf
from astrotime.util.series import TSet

class MITLoader(IterativeDataLoader):

	TestModes: List = [ "default", 'sinusoid', 'planet_crossing' ]

	def __init__(self, cfg: DictConfig, tset: TSet, **kwargs ):
		super().__init__(cfg)
		self.sector_range = cfg.sector_range
		self.sector_index: int = -1
		self.tset = tset
		self.sector_batch_offset: int = None
		self.sector_shuffle: List[int] = list( range(self.sector_range[0],self.sector_range[1]) )
		self.snr_min: float = cfg.get('snr_min',0.0)
		self.snr_max: float = cfg.get('snr_max', 1e6)
		self.max_series_length: int = cfg.get('max_series_length', 80000 )
		self.period_range: Tuple[float,float] = None
		self.current_sector: int = None
		self.loaded_sector: int = None
		self.dataset: Optional[xa.Dataset] = None
		self.train_data: Dict[str,np.ndarray] = {}
		self.synthetic = PlanetCrossingDataGenerator(cfg)
		self.tset: TSet = None
		self.test_mode_index: int = self.TestModes.index( cfg.test_mode )
		self.refresh: bool = kwargs.get('refresh',cfg.refresh)
		self._TICS: List[str]  = None

	def preprocess(self):
		self.refresh = True
		for isector in self.sector_shuffle:
			print( f"Processing sector {isector}")
			self.load_sector(isector)
		print(f"\n --- Done --- \n")

	def initialize(self,  **kwargs ):
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
			if result is None:
				self.sector_batch_offset = None
			else:
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
				self._TICS = self.dataset.attrs['TICS']
				self.log.info( f"Opened cache dataset from {dspath} in in {time.time()-t0:.3f} sec, nvars = {len(self.dataset.data_vars)}")
			else:
				self.log.info( f"Cache file not found: {dspath}")

	def size(self, sector_index) -> int:
		self.load_sector(sector_index)
		return len(self.dataset.data_vars)

	def get_dataset(self, dset_idx: int) -> xa.Dataset:
		self.load_sector(dset_idx)
		return self.dataset

	def load_sector( self, sector: int, **kwargs ) -> bool:
		t0 = time.time()
		if (self.loaded_sector != sector) or (self.dataset is None):
			self.log.info(f" Loading sector {sector}, loaded_sector={self.loaded_sector}, #TICS={len(self._TICS)}, refresh={self.refresh}")
			if self.refresh: self.dataset = None
			else:
				self._load_cache_dataset(sector)
			if self.dataset is None:
				ymax = 0.0
				elems = []
				self._read_TICS(sector)
				for iT, TIC in enumerate(self._TICS):
					data_file = self.bls_file_path(sector,TIC)
					lc_file = self.lc_file_path(sector, TIC)
					dfbls = pd.read_csv( data_file, header=None, names=['Header', 'Data'] )
					dfbls = dfbls.set_index('Header').T
					period: float = np.float64(dfbls['per'].values[0])
					if os.path.exists(lc_file):
						try:
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
						except Exception as e:
							self.log.error(f"Error reading LC-{iT} ({TIC}) from sector-{sector} file={lc_file}: {e}")

				xarrays: Dict[str, xa.DataArray] = {}
				elems.sort(key=lambda x: x[0])
				self._TICS = [elem[2] for elem in elems]
				for elem in elems:
					edata = elem[1]
					xarrays[ edata['y'].name ] = edata['y']
					xarrays[ edata['t'].name ] = edata['t']
				self.dataset = xa.Dataset( xarrays, attrs=dict(ymax=ymax,TICS=self._TICS,sector=sector) )
				t1 = time.time()
				cache_file = self.cache_path(sector)
				self.log.info(f" Loaded sector {sector} files in {(t1-t0)/60:.3f} min, saving to {cache_file}")
				self.dataset.to_netcdf( cache_file, engine="netcdf4" )
			self.loaded_sector = sector
			return True
		return False

	def get_elem_slice(self, ielem: int, series_length: int = -1 ) -> Optional[Tuple[np.ndarray,float,float,str]]:
		TIC = self._TICS[ielem]
		dst: xa.DataArray = self.dataset[TIC+".time"]
		dsy: xa.DataArray = self.dataset[TIC+".y"]
		period = dsy.attrs["period"]
		snr = dsy.attrs["sn"]
		nanmask = np.isnan(dsy.values)
		cz: np.ndarray = np.stack([dst.values[~nanmask],dsy.values[~nanmask]],axis=0)
		if series_length == -1:
			series_length = min(dst.shape[0],self.max_series_length)
		if not self.in_range(period):
			self.log.info(f"   --- Dropping elem{ielem} ({TIC}): period={period} out of range={self.period_range}")
			return None
		elif (snr<self.snr_min) or (snr>self.snr_max):
			self.log.info(f"   --- Dropping elem{ielem} ({TIC}): snr={snr:.3f} out of range=[{self.snr_min:.3f},{self.snr_max:.3f}]")
			return None
		else:
			TE, TD = dst.values[series_length-1] - dst.values[0], dst.values[-1] - dst.values[0]
			if 2*period > TE:
				self.log.info(f"   --- Dropping elem{ielem} ({TIC}): 2*(period={period:.3f}) > TE={TE:.3f}, TD={TD:.3f}, maxP={self.period_range[1]:.3f}, series_length={series_length}")
				return None
			else:
				self.log.info(f"* ELEM-{ielem} ({TIC}): period={period:.2f} snr={snr:.2f} series_length={series_length}, data_length={dst.shape[0]}, TE={TE:.3f}, TD={TD:.3f}")
				i0: int = random.randint(0, dst.shape[0]-series_length)
				elem: np.ndarray = cz[:,i0:i0+series_length]
				return elem, period, snr, TIC

	def in_range(self, p: float) -> bool:
		if self.period_range is None: return True
		return (p >= self.period_range[0]) and (p <= self.period_range[1])

	def get_training_batch(self, batch_start: int) -> Optional[Dict[str,np.ndarray]]:
		elems, ielem, series_length = [], 0, -1
		periods, sns, tics  = [], [], []
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
		if len(elems) == 0: return None
		z = np.stack(elems,axis=0)
		t,y = z[:,0,:], z[:,1,:]
		train_data = dict( batch_end=ielem, slen=series_length, t=t, y =y, period = np.array(periods), sn = np.array(sns), sector=self.current_sector, TICS=np.array(tics) )
		self.log.info( f"get_training_batch({batch_start}), t{train_data['t'].shape}, y{train_data['y'].shape}, p{train_data['period'].shape}, TRangs={t[0][-1]-t[0][0]:.3f}")
		return train_data

	def get_training_element(self, sector: int, element_index: int) -> Dict[str,np.ndarray]:
		self.load_sector(sector)
		eslice = self.get_elem_slice(element_index)
		train_data = None
		if eslice is not None:
			elem, period, sn = eslice
			train_data = dict( t=elem[0], y=elem[1], period=period, sn=sn, sector=self.current_sector )
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

class MITElementLoader(ElementLoader):

	def __init__(self, cfg: DictConfig, **kwargs ):
		super().__init__(cfg)
		self.sector_range = cfg.sector_range
		self.loaded_file = -1
		self.filters = kwargs.get('filters',True)
		self.snr_min: float = cfg.get('snr_min',0.0 )
		self.snr_max: float = cfg.get('snr_max', 1.0e9 )
		print(f" MITElementLoader.snr: {self.snr_min:.3f} -> {self.snr_max:.3f}")
		self.max_series_length: int = cfg.get('max_series_length', 80000 )
		self.period_range: Tuple[float,float] = self.get_period_range()
		self._TICS: List[str]  = None
		self.preload = kwargs.get('preload',False)
		self.elems = []
		self.file_sort = None

	def set_tset(self, tset: TSet):
		ElementLoader.set_tset(self, tset)
		self.file_sort = self.get_file_sort(tset)
		return self

	def get_file_sort(self, tset: TSet):
		if   tset == TSet.Train:      return  list(range(self.ntfiles))
		elif tset == TSet.Validation: return  [self.ntfiles]
		elif tset == TSet.Update:     return  list(range(self.ntfiles+1))
		else: raise ValueError(f"Unknown tset: {tset}")

	def get_period_range(self) -> Tuple[float,float]:
		f0 = self.cfg.base_freq
		f1 = f0 + f0 * 2**self.cfg.noctaves
		return 1/f1, 1/f0

	@property
	def batch_size(self) -> int:
		return self.cfg.batch_size

	@property
	def TICS( self ) -> List[str]:
		self.load_data()
		return self._TICS

	def init_epoch(self, tset: TSet = TSet.Train):
		self.ifile = 0
		self.batch_offset = 0
		self.set_tset(tset)
		random.shuffle(self.file_sort)
		self._load_cache_dataset()

	@property
	def cache_path(self) -> str:
		os.makedirs(self.cfg.dataset_root, exist_ok=True)
		isector = self.cfg.sector_range[0] + self.file_sort[self.ifile]
		return f"{self.cfg.dataset_root}/sector-{isector}.nc"

	def _load_cache_dataset( self ):
		dspath: str = self.cache_path
		if os.path.exists(dspath):
			self.data = xa.open_dataset( dspath, engine="netcdf4" )
			self.get_sorted_TICS()
			self.log.info( f"Opened cache dataset from {dspath}, nvars = {len(self.data.data_vars)//2}")
		else:
			self.log.info( f"Cache file not found: {dspath}")

	def load_data( self ) -> bool:
		if (self.loaded_file != self.ifile) or (self.data is None):
			self._load_cache_dataset()
			self.loaded_file = self.ifile
			if self.preload:
				self.preload_elems()
			return True
		return False

	@property
	def file_size(self) -> int:
		self.load_data()
		return len(self._TICS)

	@property
	def nelements(self) -> int:
		return len(self._TICS)

	@property
	def nfiles(self) -> int:
		return self.cfg.sector_range[1] - self.cfg.sector_range[0]

	@property
	def ntfiles(self) -> int:
		return self.nfiles-1

	def get_element(self, elem_index: int) -> Optional[RDict]:
		self.load_data()
		if self.preload: return self.get_loaded_element(elem_index)
		else:            return self.get_raw_element(elem_index)

	def get_loaded_element(self, elem_index: int) -> Optional[RDict]:
		try:
			elem = self.elems[elem_index]
			return dict( **elem )
		except IndexError:
			self.log.error(f"get_loaded_element: elem_index={elem_index} outside elems[{len(self.elems)}], nelements={self.nelements}")
			return None

	def get_raw_element( self, elem_index: int ) -> Optional[RDict]:
		TIC = self._TICS[elem_index]
		dsy: xa.DataArray = self.data[TIC+".y"]
		period = dsy.attrs["period"]
		sn = dsy.attrs["sn"]
		if self.in_range(period) and (sn>self.snr_min) and (sn<self.snr_max):
			nanmask = np.isnan(dsy.values)
			dst: xa.DataArray = self.data[TIC + ".time"]
			y: np.ndarray = dsy.values[~nanmask]
			ss = y.std()
			if (ss > 0.0) and (ss < 1e10):
				train_data = dict( t=dst.values[~nanmask], y=y, period=period, sn=sn, sector=self.ifile, tic=TIC )
				return train_data
		return None

	def in_range(self, p: float) -> bool:
		if self.period_range is None: return True
		return (p >= self.period_range[0]) and (p <= self.period_range[1])

	def update_file(self):
		if self.batch_offset >= self.nelements-1:
			self.batch_offset = 0
			self.ifile += 1
			if (self.ifile >= self.ntfiles) or (self.tset == TSet.Validation):
				raise StopIteration
			self.load_data()

	def get_sorted_TICS(self) -> Optional[Dict[str,np.ndarray]]:
		elems = [ (self.data[TIC+".y"].size, TIC) for TIC in self.data.attrs['TICS'] ]
		elems.sort(key=lambda x: x[0])
		self._TICS = [elem[1] for elem in elems]

	def n_loaded_elems(self) -> int:
		return len(self.elems)

	def get_next_batch(self, update_file=True ) -> Optional[Dict[str,np.ndarray]]:
		ielem, periods, sns, tics, ts, ys, slens, b0  = 0, [], [], [], [], [], [], self.batch_offset
		if update_file: self.update_file()
		elif self.batch_offset >= self.nelements-1:
			self.batch_offset = 0
			return None
		for ielem in range( b0, self.nelements ):
			elem: RDict = self.get_raw_element(ielem)
			if elem is not None:
				ts.append(elem['t'])
				ys.append(elem['y'])
				periods.append(elem['period'])
				sns.append(elem['sn'])
				tics.append(elem['tic'])
				slens.append( elem['y'].size )
			if len(ts) >= self.cfg.batch_size:
				break
		if len(ts) == 0: return None
		self.batch_offset = ielem + 1
		slen = np.array(slens).min()
		yn = np.stack( [ y[:slen] for y in ys], axis=0 )
		tn = np.stack( [ t[:slen] for t in ts], axis=0 )
		return dict( t=tn, y=yn, period=np.array(periods), sn=np.array(sns), TICS=np.array(tics) )

	def preload_elems(self) -> Optional[Dict[str,np.ndarray]]:
		self.elems = []
		for ielem in range( 0, self.nelements ):
			elem: Optional[RDict] = self.get_raw_element(ielem)
			if elem is not None:
				self.elems.append( elem )


