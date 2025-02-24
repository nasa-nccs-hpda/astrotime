import time, numpy as np, xarray as xa
from astrotime.loaders.base import DataLoader
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.util.logging import lgm, exception_handled
from glob import glob
from omegaconf import DictConfig, OmegaConf
from astrotime.encoders.base import Encoder
from torch import Tensor, device
import torch

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
		self.files: List[str] = None
		self.cfg = cfg
		self._nelements = -1
		self.current_file = 0
		self.dataset: xa.Dataset = None
		self.batches_per_file = self.cfg.file_size // self.cfg.batch_size

	@property
	def file_paths( self ) -> List[str]:
		if self.files is None:
			self.files = glob( self.cfg.dataset_files, root_dir=self.cfg.dataset_root )
			print( f"file_paths: dataset_root={self.cfg.dataset_root}, dataset_files={self.cfg.dataset_files}, nfiles={len(self.files)}")
			self.files.sort()
		return self.files

	@property
	def nbatches(self) -> int:
		return self.nfiles * self.batches_per_file

	@property
	def batch_size(self) -> int:
		return self.cfg.batch_size

	@property
	def nfiles(self) -> int:
		return len(self.file_paths)

	def file_path( self, file_index: int ) -> Optional[str]:
		try:
			return f"{self.cfg.dataset_root}/{self.file_paths[file_index]}"
		except IndexError:
			return None

	@classmethod
	def get_elem(cls, dset: xa.Dataset, elem: int) -> xa.DataArray:
		slen: int = int(dset['slen'][elem])
		y: np.ndarray = dset['y'].isel(elem=elem).values[:slen]
		t: np.ndarray = dset['t'].isel(elem=elem).values[:slen]
		p: float = float(dset['p'][elem])
		return xa.DataArray( y, dict( t=t ), ['t'], attrs=dict( period=p ) )

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
				lgm().log(f"Loaded {self._nelements} sinusoids in {time.time()-t0:.3f} sec from file: {file_path}, freq range = [{f.values.min():.3f}, {f.values.max():.3f}]")
				return True
		return False

	@exception_handled
	def get_data_element( self, batch_index: int, element_index: int ) -> xa.DataArray:
		self.load_file(batch_index)
		return self.get_elem( self.dataset, element_index )

	@exception_handled
	def get_batch( self, batch_index: int ) -> xa.Dataset:
		t0 = time.time()
		file_index = batch_index // self.batches_per_file
		self.load_file(file_index)
		bstart = (batch_index % self.batches_per_file) * self.cfg.batch_size
		result = self.dataset.isel( elem=slice(bstart,bstart+self.cfg.batch_size))
		lgm().log( f" ----> BATCH-{file_index}.{batch_index}: bstart={bstart}, batch_size={self.cfg.batch_size}, batches_per_file={self.batches_per_file}, y{result['y'].shape} t{result['t'].shape} p{result['p'].shape}, dt={time.time()-t0:.4f} sec")
		return self.dataset

	def get_dataset(self, dset_idx: int) -> xa.Dataset:
		self.load_file(dset_idx)
		return self.dataset