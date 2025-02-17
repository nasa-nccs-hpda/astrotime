import time, numpy as np, xarray as xa
from astrotime.loaders.base import DataLoader
from typing import List, Optional, Dict, Type
from astrotime.util.logging import lgm, exception_handled
from glob import glob
from astrotime.transforms.filters import TrainingFilter

class SinusoidLoader(DataLoader):

	def __init__(self, data_dir: str):
		super().__init__()
		self.data_dir = data_dir

	def get_dataset( self, dset_idx: int ) -> Dict:
		data = np.load( f"{self.data_dir}/sinusoids_{dset_idx}.npz", allow_pickle=True)
		return dict( y=data['sinusoids'], x=data['times'], target=data['periods'] )


class ncSinusoidLoader:

	def __init__(self, dataset_root, dataset_files, file_size, batch_size):
		self.files: List[str] = None
		self.dataset_root = dataset_root
		self.dataset_files = dataset_files
		self._nelements = -1
		self.current_file = 0
		self.dataset: xa.Dataset = None
		self.file_size = file_size
		self.filters: List[TrainingFilter] = []
		self.batch_size = batch_size
		self.batches_per_file = self.file_size // self.batch_size

	def add_filters(self, filters: List[TrainingFilter] ):
		self.filters.extend( filters )

	@property
	def file_paths( self ) -> List[str]:
		if self.files is None:
			self.files = glob( self.dataset_files, root_dir=self.dataset_root )
			self.files.sort()
		return self.files

	@property
	def nelements(self) -> int:
		if self._nelements == -1:
			self.load_file( self.current_file )
		return self._nelements

	@property
	def nbatches(self) -> int:
		return self.nfiles * self.batches_per_file

	@property
	def nfiles(self) -> int:
		return len(self.file_paths)

	def file_path( self, file_index: int ) -> Optional[str]:
		try:
			return f"{self.dataset_root}/{self.file_paths[file_index]}"
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
				self.dataset = self.apply_filters( xa.Dataset( dict(y=y,t=t,p=p,f=f) ) )
				lgm().log(f"Loaded {self._nelements} sinusoids in {time.time()-t0:.3f} sec from file: {file_path}, freq range = [{f.values.min():.3f}, {f.values.max():.3f}]")
				return True
		return False

	def apply_filters( self, dataset: xa.Dataset ) -> xa.Dataset:
		for f in self.filters:
			dataset = f.apply( dataset )
		return dataset

	@exception_handled
	def get_data_element( self, batch_index: int, element_index: int ) -> xa.DataArray:
		self.load_file(batch_index)
		return self.get_elem( self.dataset, element_index )

	@exception_handled
	def get_batch( self, batch_index: int ) -> xa.Dataset:
		file_index = batch_index // self.batches_per_file
		self.load_file(file_index)
		bstart = (batch_index % self.batches_per_file) * self.batch_size
		result = self.dataset.isel( elem=slice(bstart,bstart+self.batch_size) )
		return result