import time, numpy as np, xarray as xa
from astrotime.loaders.base import DataLoader
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.util.logging import exception_handled
from astrotime.util.series import TSet
from glob import glob
from omegaconf import DictConfig, OmegaConf
import logging

class SinusoidLoader(DataLoader):

	def __init__(self, data_dir: str):
		super().__init__()
		self.data_dir = data_dir

	def get_dataset( self, dset_idx: int ) -> Dict:
		data = np.load( f"{self.data_dir}/sinusoids_{dset_idx}.npz", allow_pickle=True)
		return dict( y=data['sinusoids'], x=data['times'], target=data['periods'] )

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
		period: float = float(self.dataset['p'][element_index])
		return  dict( y=y, t=t, period=period )

	def get_dataset_element(self, dset_idx: int, element_index, **kwargs) -> xa.Dataset:
		self.load_file(dset_idx)
		print(f" get_dataset_element: {list(self.dataset.data_vars.keys())}")
		y: np.ndarray = self.dataset['y'].isel(elem=element_index).values
		t: np.ndarray = self.dataset['t'].isel(elem=element_index).values
		period: float = float(self.dataset['p'][element_index])
		return xa.Dataset( dict( y=y, t=t ), attrs=dict( period=period ) )

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