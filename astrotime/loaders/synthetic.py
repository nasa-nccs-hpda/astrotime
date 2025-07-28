import time, os, math, numpy as np, xarray as xa, random
from astrotime.loaders.base import IterativeDataLoader, RDict, ElementLoader
from typing import List, Optional, Dict, Type, Union, Tuple, Any
import torch, shutil
from glob import glob
from omegaconf import DictConfig, OmegaConf
from astrotime.util.series import TSet

def merge( arrays: List[np.ndarray], slen: int ) -> np.ndarray:
	if len( arrays ) == 0: raise IndexError
	return np.stack( [ array[:slen] for array in arrays ], axis=0 )


class SyntheticElementLoader(ElementLoader):

	def __init__(self, cfg: DictConfig, **kwargs):
		super().__init__(cfg, **kwargs)
		self.batch_index = 0
		self.batch_size =self.cfg.batch_size
		self.file_sort = None
		self.current_batch = None
		self.use_batches = kwargs.get('use_batches',True)

	def set_tset(self, tset: TSet):
		ElementLoader.set_tset(self, tset)
		self.file_sort = self.get_file_sort(tset)
		return self

	@property
	def nfiles(self):
		return len(self.file_sort)

	def move_and_open( self, current_file ):
		dspath0 = f"{self.rootdir}/nc/{self.dset}-{current_file}.nc"
		dspath1 = f"{self.rootdir}/nc/{self.dset}-0-{current_file}.nc"
		try: shutil.move( dspath0, dspath1 )
		except FileNotFoundError: pass
		dataset = xa.open_dataset(dspath1, engine="netcdf4")
		return dataset, dspath0

	def get_file_sort(self, tset: TSet):
		if   tset == TSet.Train:      return  list(range(self.ntfiles))
		elif tset == TSet.Validation: return  [self.ntfiles]
		elif tset == TSet.Update:     return  list(range(self.ntfiles+1))
		else: raise ValueError(f"Unknown tset: {tset}")

	def set_file(self, file_idx: int):
		if (file_idx != self.ifile) or (self.data is None):
			self.ifile = file_idx
			self._load_cache_dataset()

	def init_epoch(self, tset: TSet = TSet.Train):
		self.ifile = 0
		self.batch_index = 0
		self.set_tset(tset)
		random.shuffle(self.file_sort)
		self._load_cache_dataset()

	def get_element(self, elem_index: int) -> Optional[RDict]:
		return self.get_batch_element( elem_index ) if self.use_batches else self.get_raw_element( elem_index )

	def get_raw_element(self, elem_index: int) -> Optional[RDict]:
		try:
			dsy: xa.DataArray = self.data[ f's{elem_index}' ]
			t: np.ndarray     = self.data[ f't{elem_index}' ].values
			y: np.ndarray = dsy.values
			return dict(t=t, y=y, p=dsy.attrs["period"], type=dsy.attrs["type"], o=dsy.attrs.get("octave"))
		except KeyError as ex:
			self.log.info(f"   Error getting elem-{elem_index} from dataset({self.dspath}): vars = {list(self.data.data_vars.keys())}")
			return None

	def add_octave_data(self, octave_data: List[Tuple[int,int,int]]):
		dataset, dspath, current_file, nupdates = None, None, -1, 0
		for (file_index, elem_index, octave) in octave_data:
			if current_file != file_index:
				if dataset is not None:
					print(f" * Writing {nupdates} octaves to file {current_file}: {dspath}")
					dataset.to_netcdf(dspath)
					nupdates = 0
				current_file = file_index
				dataset, dspath = self.move_and_open( current_file )
			dsy: xa.DataArray = dataset[f's{elem_index}']
			dsy.attrs["octave"] = octave
			nupdates += 1
		print(f" *** Writing {nupdates} octaves to file {current_file}: {dspath}")
		dataset.to_netcdf(dspath)

	def get_sort_ordering(self):
		sort_ordering = []
		for ielem in range(self.file_size):
			dsy: xa.DataArray = self.data[ f's{ielem}' ]
			sort_ordering.append( (ielem,dsy.size) )
		sort_ordering.sort(key=lambda x: x[1])
		return sort_ordering

	def get_batch_element(self, elem_index: int) -> Optional[RDict]:
		try:
			batch_idx = elem_index // self.batch_size
			if batch_idx != self.current_batch:
				self.current_batch = self.get_batch(batch_idx)
				self.batch_index = batch_idx
			ib, b = elem_index % self.batch_size, self.current_batch
			return dict(t=b['t'][ib], y=b['y'][ib], p=b['period'][ib])
		except IndexError:
			return None

	def get_next_batch( self ) -> Optional[Dict[str,Any]]:
		batch_start = self.batch_index*self.batch_size
		if batch_start >= self.file_size:
			self.ifile += 1
			self.batch_index = 0
			if self.ifile  >=  len(self.file_sort):
				raise StopIteration
			self._load_cache_dataset()
		batch: Optional[Dict[str,Any]] = self.get_batch(self.batch_index)
		self.batch_index += 1
		return batch

	def get_batch( self, batch_index: int ) -> Optional[Dict[str,Any]]:
		if self.data is not None:
			batch_start = batch_index * self.batch_size
			batch_end = min(batch_start + self.batch_size, self.file_size)
			t,y,p,o,stype,result,tlen0,tlen1 = [],[],[],[],[],{},1000000,0
			for ielem in range(batch_start, batch_end):
				elem = self.get_raw_element(ielem)
				if elem is not None:
					t.append(elem['t'])
					y.append(elem['y'])
					p.append(elem['p'])
					o.append(elem['o'])
					stype.append(elem['type'])
					if t[-1].size < tlen0: tlen0 = t[-1].size
					if t[-1].size > tlen1: tlen1 = t[-1].size
			if tlen1 > 0:
				result['t'] = merge(t,tlen0)
				result['y'] = merge(y,tlen0)
				result['period'] = np.array(p)
				result['octave'] = np.array(o)
				result['stype'] = np.array(stype)
				result['offset'] = batch_start
				result['file'] = self.file_index
				self.log.debug(f"get_batch(F{self.file_index}.B{batch_index}): y{result['y'].shape}, t{result['t'].shape}, len-diff={tlen1-tlen0}, pmax={result['period'].max():.3f}, trng0={result['t'][0][-1]-result['t'][0][0]:.3f}")
				return result
		return None

	@property
	def dspath(self) -> str:
		self.file_index = self.file_sort[self.ifile]
		return f"{self.rootdir}/nc/{self.dset}-{self.file_index}.nc"

	def _load_cache_dataset( self ):
		if os.path.exists(self.dspath):
			try:
				self.data = xa.open_dataset( self.dspath, engine="netcdf4" )
				self.log.info( f"Opened cache dataset from {self.dspath}, nvars = {len(self.data.data_vars)}")
				if self.tset == TSet.Update:
					print(f"Computing octaves from dataset-{self.ifile}: {self.dspath} with {len(self.data.data_vars)} vars")
			except KeyError as ex:
				print(f"Error reading file: {self.dspath}: {ex}")
		else:
			print( f"Cache file not found: {self.dspath}")

class SyntheticLoader:

	def __init__(self, cfg: DictConfig, tset: TSet, **kwargs ):
		super().__init__()
		self.cfg = cfg
		self.tset = tset
		self.files = None
		self.nfiles = None
		self.sector_index = 0
		self.period_range: Tuple[float,float] = None
		self.sector_shuffle = None
		self.series_length = cfg.series_length
		self.loaded_sector = None
		self.sector_batch_offset = None
		self.dataset: Optional[xa.Dataset] = None
		self.train_data: Dict[str,np.ndarray] = {}
		self.tset: TSet = None
		self._nbatches = -1

	def get_period_range(self) -> Tuple[float,float]:
		f0 = self.cfg.base_freq
		f1 = f0 + f0 * 2**self.cfg.noctaves
		return 1/f1, 1/f0


	def in_range(self, p: float) -> bool:
		if self.period_range is None: return True
		return (p >= self.period_range[0]) and (p <= self.period_range[1])

	@property
	def current_sector(self):
		return self.sector_shuffle[self.sector_index]

	def initialize(self):
		self.files = glob(f"{self.cfg.source}_*.nc", root_dir=self.cfg.dataset_root)
		self.nfiles = len(self.files)
		self.sector_shuffle = list(range(self.nfiles))
		self.period_range = self.get_period_range()
		self.init_epoch()

	def init_epoch(self):
		self.sector_batch_offset = 0
		self.sector_index = 0
		self._nbatches = -1
		random.shuffle(self.sector_shuffle)

	def get_next_batch( self ) -> Optional[Dict[str,Any]]:
		ibatch = self.sector_batch_offset//self.cfg.batch_size
		if (self._nbatches > 0) and ( ibatch>= self._nbatches-1):
			self.sector_index = self.sector_index + 1
			if self.sector_index == self.nfiles:
				raise StopIteration
			self.sector_batch_offset = 0

		if self.current_sector >= 0:
			self.load_sector(self.current_sector)
			if self.dataset is not None:
				batch_start = self.sector_batch_offset
				batch_end   = batch_start+self.cfg.batch_size
				result: RDict = { k: self.train_data[k][batch_start:batch_end] for k in ['t','y','period','stype'] }
				result['offset'] = batch_start
				result['sector'] = self.current_sector
				self.sector_batch_offset = batch_end
				self.log.debug(f"get_next_batch: y{result['y'].shape}, t{result['t'].shape}")
				return result
		return None

	def get_batch( self, sector_index: int, batch_index: int ) -> Optional[Dict[str,Any]]:
		self.load_sector(sector_index)
		batch_start = self.cfg.batch_size*batch_index
		batch_end   = batch_start+self.cfg.batch_size
		result: Dict[str,Any] = { k: self.train_data[k][batch_start:batch_end] for k in ['t','y','period','stype'] }
		result['offset'] = batch_start
		result['sector'] = self.current_sector
		return result

	def get_element( self, sector_index: int, element_index: int, **kwargs ) -> Optional[Dict[str,Any]]:
		self.load_sector(sector_index, **kwargs)
		element_data: Dict[str,Any] = { k: self.train_data[k][element_index] for k in ['t','y','period','stype'] }
		element_data['offset'] = element_index
		element_data['sector'] = self.current_sector
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
				try:
					self.dataset = xa.open_dataset( dspath, engine="netcdf4" )
					self.log.info( f"Opened cache dataset from {dspath} in in {time.time()-t0:.3f} sec, nvars = {len(self.dataset.data_vars)}")
				except KeyError as ex:
					self.log.error(f"Error reading file: {dspath}: {ex}")
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
			if self.dataset is not None:
				self.loaded_sector = sector
				if kwargs.get('training',True):
					self.update_training_data(**kwargs)
			return True
		return False

	def get_single_element(self, sector, elem_index, **kwargs ):
		self.load_sector(sector,training=False)
		svids = [ vid[1:] for vid in self.dataset.data_vars.keys() if vid[0]=='s']
		dsy: xa.DataArray = self.dataset['s' + svids[elem_index]]
		dst: xa.DataArray = self.dataset['t' + svids[elem_index]]
		return dict( t=dst.values, y=dsy.values, p=dsy.attrs["period"], type=dsy.attrs["type"], sector=sector, elem=elem_index )

	def get_elem_slice(self, svid: str, **kwargs ) -> Optional[Tuple[np.ndarray,float,str]]:
		try:
			dsy: xa.DataArray = self.dataset['s'+svid]
			dst: xa.DataArray = self.dataset['t'+svid]
			period = dsy.attrs["period"]
			stype = dsy.attrs["type"]
			ct, cy = dst.values, dsy.values
			cz: np.ndarray = np.stack([ct,cy],axis=0)
			if kwargs.get('filtered',True):
				if cz.shape[1] < self.series_length:
					return None
				else:
					TD = ct[-1] - ct[0]
					TE = ct[self.series_length] - ct[0]
					if period > TE:
						print(f"Dropping elem-{svid}: period={period:.3f} > TE={TE:.3f}, TD={TD:.3f}, maxP={self.period_range[1]:.3f}")
						return None
					else:
						if 2*period > TE:
							peak_idx: int = np.argmin(cy)
							TP = ct[peak_idx] - ct[0]
							i0 = 0 if (TP > period) else min( max( peak_idx - 10, 0 ), ct.shape[0] - self.series_length )
						else:
							i0: int = random.randint(0, ct.shape[0] - self.series_length)
						elem: np.ndarray = cz[:, i0:i0 + self.series_length]
						return elem, period, stype
			else:
				return cz, period, stype
		except KeyError as err:
			print(f"KeyError for elem-{svid}: {err} <-> dset-vars={list(self.dataset.data_vars.keys())}")
			return None

	def get_batch_element(self, bz: np.ndarray) -> np.ndarray:
		center = bz.shape[1] // 2
		bdata = bz[:,center-self.series_length//2:center+self.series_length//2]
		return bdata

	def update_training_data(self, **kwargs):
		periods, stypes, elems  = [], [], []
		svids = [ vid[1:] for vid in self.dataset.data_vars.keys() if vid[0]=='s']
		for svid in svids:
			signal = self.get_elem_slice(svid,**kwargs)
			if signal is not None:
				eslice, period, stype = signal
				if self.in_range(period):
					elems.append(eslice)
					periods.append(period)
					stypes.append(stype)
				else:
					print( f" -----> Period out of range: {period:.3f} <-> prng=({self.period_range[0]:.3f}, {self.period_range[1]:.3f}) f0,nO=({self.cfg.base_freq:.3f},{self.cfg.noctaves:.3f})" )
		z = np.stack(elems,axis=0)
		self.train_data['t'] = z[:,0,:]
		self.train_data['y'] = z[:,1,:]
		self.train_data['period'] = np.array(periods)
		self.train_data['stype'] = np.array(stypes)
		self._nbatches = math.ceil( self.train_data['t'].shape[0] / self.cfg.batch_size )
		trng: np.ndarray = z[:,0,-1] - z[:,0,0]
		self.log.info(f"Load sector-{self.loaded_sector}, size={z.shape[0]}, T-range: ({trng.min():.3f}->{trng.max():.3f}), mean={trng.mean():.3f}")

	def refresh(self):
		self.dataset = None

class SyntheticOctavesLoader(SyntheticLoader):

	def __init__(self, cfg: DictConfig, **kwargs ):
		super().__init__(cfg, **kwargs)
		self.nfreq: int = cfg.series_length
		self.base_freq: float = cfg.base_freq
		self.noctaves: int = cfg.noctaves


