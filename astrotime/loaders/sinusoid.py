import time, numpy as np, xarray as xa
from astrotime.loaders.base import DataLoader
from typing import List, Optional, Dict, Type, Union, Tuple, Any
from astrotime.util.logging import exception_handled
from astrotime.loaders.base import ElementLoader, RDict
from astrotime.util.series import TSet
from glob import glob
from omegaconf import DictConfig, OmegaConf
import logging, random, os

def merge( arrays: List[np.ndarray], slen: int ) -> np.ndarray:
	if len( arrays ) == 0: raise IndexError
	return np.stack( [ array[:slen] for array in arrays ], axis=0 )

class SinusoidLoader(DataLoader):

	def __init__(self, data_dir: str):
		super().__init__()
		self.data_dir = data_dir

	def get_dataset( self, dset_idx: int ) -> Dict:
		data = np.load( f"{self.data_dir}/sinusoids_{dset_idx}.npz", allow_pickle=True)
		return dict( y=data['sinusoids'], x=data['times'], target=data['periods'] )

class SinusoidElementLoader(ElementLoader):

	def __init__(self, cfg: DictConfig, tset: TSet, **kwargs):
		super().__init__(cfg, **kwargs)
		self.batch_index = 0
		self.tset = tset
		self.batch_size =self.cfg.batch_size
		self.current_batch = None
		self.elem_sort = None
		self.file_sort = list(range(self.ntfiles)) if (tset == TSet.Train) else [self.ntfiles]
		self.use_batches = kwargs.get('use_batches',True)
		self._files = None
		self._load_cache_dataset()

	@property
	def file_paths( self ) -> List[str]:
		if self._files is None:
			self._files = glob( self.cfg.dataset_files, root_dir=self.rootdir )
			self._files.sort()
		return self._files

	def set_file(self, file_idx: int):
		if (file_idx != self.ifile) or (self.data is None):
			self.ifile = file_idx
			self._load_cache_dataset()

	def init_epoch(self):
		random.shuffle(self.file_sort)
		self.ifile = 0
		self.batch_index = 0
		self._load_cache_dataset()

	@property
	def nelem(self):
		return self.file_size

	def get_element(self, elem_index: int) -> Optional[RDict]:
		return self.get_batch_element( elem_index ) if self.use_batches else self.get_raw_element( elem_index )

	def get_raw_element(self, elem_index: int) -> Optional[RDict]:
		try:
			eidx = self.elem_sort[elem_index][0]
			dsy: xa.DataArray = self.data[ f'y' ]
			dst: xa.DataArray = self.data[ f'time' ]
			print( f" get_raw_element: dsy{list(dsy.shape)} dst{list(dst.shape)}")
			y: np.ndarray = dsy.values
			return dict(t=dst.values, y=y/y.mean(), p=dsy.attrs["period"], type=dsy.attrs["type"])
		except KeyError as ex:
			print(f"\n    Error getting elem-{elem_index} from dataset({self.dspath}): vars = {list(self.data.data_vars.keys())}\n")
			raise ex

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
			return dict(t=b['t'][ib], y=b['y'][ib], p=b['period'][ib], type=b['stype'][ib])
		except IndexError:
			return None

	def check_epoch_end(self):
		end_index = self.ntfiles if (self.tset == TSet.Train) else 1
		if self.ifile >= end_index:
			raise StopIteration

	def get_next_batch( self ) -> Optional[Dict[str,Any]]:
		batch_start = self.batch_index*self.batch_size
		if batch_start >= self.file_size:
			self.ifile += 1
			self.batch_index = 0
			self.check_epoch_end()
			self._load_cache_dataset()
		batch: Optional[Dict[str,Any]] = self.get_batch(self.batch_index)
		if batch is not None:
			self.batch_index += 1
		return batch

	def get_batch( self, batch_index: int ) -> Optional[Dict[str,Any]]:
		if self.data is not None:
			batch_start = batch_index * self.batch_size
			batch_end = min(batch_start + self.batch_size, self.file_size)
			t,y,p,stype,result,tlen0,tlen1 = [],[],[],[],{},1000000,0
			for ielem in range(batch_start, batch_end):
				elem = self.get_raw_element(ielem)
				if elem is not None:
					t.append(elem['t'])
					y.append(elem['y'])
					p.append(elem['p'])
					stype.append(elem['type'])
					if t[-1].size < tlen0: tlen0 = t[-1].size
					if t[-1].size > tlen1: tlen1 = t[-1].size
			result['t'] = merge(t,tlen0)
			result['y'] = merge(y,tlen0)
			result['period'] = np.array(p)
			result['stype'] = np.array(stype)
			result['offset'] = batch_start
			result['file'] = self.ifile
			self.log.debug(f"get_batch(F{self.ifile}.B{batch_index}): y{result['y'].shape}, t{result['t'].shape}, len-diff={tlen1-tlen0}, pmax={result['period'].max():.3f}, trng0={result['t'][0][-1]-result['t'][0][0]:.3f}")
			return result
		return None

	@property
	def dspath(self) -> str:
		return f"{self.rootdir}/{self.file_paths[self.file_sort[self.ifile]]}"

	def _load_cache_dataset( self ):
		if os.path.exists(self.dspath):
			try:
				self.data = xa.open_dataset( self.dspath, engine="netcdf4" )
				self.elem_sort = self.get_sort_ordering()
				self.log.info( f"Opened cache dataset from {self.dspath}, nvars = {len(self.data.data_vars)}")
			except KeyError as ex:
				print(f"Error reading file: {self.dspath}: {ex}")
		else:
			print( f"Cache file not found: {self.dspath}")

class ncSinusoidLoader(DataLoader):

	def __init__(self, cfg: DictConfig ):
		super().__init__()
		self._files: List[str] = None
		self.cfg = cfg
		self._nelements = -1
		self.current_file = 0
		self.dataset: xa.Dataset = None
		self.batches_per_file = self.cfg.file_size // self.cfg.batch_size
		self.log = logging.getLogger()

	@property
	def dset_idx(self) -> int:
		return self.current_file

	@property
	def ndsets(self) -> int:
		return self.nfiles


	@property
	def file_paths( self ) -> List[str]:
		if self._files is None:
			self._files = glob( self.cfg.dataset_files, root_dir=self.cfg.dataset_root )
			self._files.sort()
		return self._files

	@property
	def nbatches_total(self) -> int:
		return int( self.nfiles * self.batches_per_file * self.cfg.dset_reduction )

	def nbatches(self, tset: TSet) -> int:
		nbval: int = int(self.nbatches_total * self.cfg.validation_fraction)
		if   tset == TSet.Validation:  return nbval
		elif tset == TSet.Train:       return self.nbatches_total - nbval
		else: raise Exception( f"Invalid TSet: {tset}")

	@property
	def batch_size(self) -> int:
		return self.cfg.batch_size

	@property
	def nfiles(self) -> int:
		return  len(self.file_paths)

	def file_path( self, file_index: int ) -> Optional[str]:
		try:
			return f"{self.cfg.dataset_root}/{self.file_paths[file_index]}"
		except IndexError:
			return None

	def get_element(self, dset_idx: int, element_index) -> Optional[Dict[str,Union[np.ndarray,float]]]:
		self.load_file(dset_idx)
		print( f" get_element: {list(self.dataset.data_vars.keys())}")
		y: np.ndarray = self.dataset['y'].isel(elem=element_index).values
		t: np.ndarray = self.dataset['t'].isel(elem=element_index).values
		p: float = float(self.dataset['p'][element_index])
		return  dict( y=y, t=t, p=p )

	def get_dataset_element(self, dset_idx: int, element_index, **kwargs) -> xa.Dataset:
		self.load_file(dset_idx)
		print(f" get_dataset_element: {list(self.dataset.data_vars.keys())}")
		y: np.ndarray = self.dataset['y'].isel(elem=element_index).values
		t: np.ndarray = self.dataset['t'].isel(elem=element_index).values
		p: float = float(self.dataset['p'][element_index])
		return xa.Dataset( dict( y=y, t=t ), attrs=dict( p=p ) )

	@exception_handled
	def load_file( self, file_index: int ) -> bool:
		if (self.current_file != file_index) or (self.dataset is None):
			file_path = self.file_path( file_index )
			if file_path is not None:
				t0 = time.time()
				self.dataset: xa.Dataset = xa.open_dataset( file_path, engine="netcdf4")
				slen: int = self.dataset['slen'].values.min()
				y: xa.DataArray = self.dataset['y'].isel(time=slice(0,slen))
				t: xa.DataArray = self.dataset['t'].isel(time=slice(0,slen))
				p: xa.DataArray = self.dataset['p']
				f: xa.DataArray = self.dataset['f']
				self.current_file = file_index
				self._nelements = self.dataset.sizes['elem']
				self.dataset = xa.Dataset( dict( y=y, t=t, p=p, f=f ) )
				self.log.info(f"Loaded {self._nelements} sinusoids in {time.time()-t0:.3f} sec from file: {file_path}, freq range = [{f.values.min():.3f}, {f.values.max():.3f}]")
				return True
		return False

	@exception_handled
	def get_batch( self, tset: TSet, batch_index: int ) -> xa.Dataset:
		t0 = time.time()
		if tset == TSet.Validation:
			batch_index = batch_index + self.nbatches(TSet.Train)
		file_index = batch_index // self.batches_per_file
		self.load_file(file_index)
		bstart = (batch_index % self.batches_per_file) * self.cfg.batch_size
		result = self.dataset.isel( elem=slice(bstart,bstart+self.cfg.batch_size))
		self.log.info( f" ----> BATCH-{file_index}.{batch_index}: bstart={bstart}, batch_size={self.cfg.batch_size}, batches_per_file={self.batches_per_file}, y{result['y'].shape} t{result['t'].shape} p{result['p'].shape}, dt={time.time()-t0:.4f} sec")
		return result

	def get_dataset(self, dset_idx: int) -> xa.Dataset:
		self.load_file(dset_idx)
		return self.dataset