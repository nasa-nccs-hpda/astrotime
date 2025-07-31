from .base import AstrotimeDataset, RDict
import time, numpy as np, xarray as xa
from typing import List, Optional, Dict, Type, Union, Tuple, Any
from astrotime.util.logging import exception_handled
from astrotime.util.series import TSet
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import logging, random, os, math

def merge( arrays: List[np.ndarray], slen: int ) -> np.ndarray:
	if len( arrays ) == 0: raise IndexError
	return np.stack( [ array[:slen] for array in arrays ], axis=0 )

class SinusoidDataset(AstrotimeDataset):

	def __init__(self, cfg: DictConfig ):
		super(SinusoidDataset).__init__(cfg)
		self.data: xa.Dataset = None

	def _load_next_file( self ):
		if os.path.exists(self.dspath):
			try:
				self.data = xa.open_dataset( self.dspath, engine="netcdf4" )
				self.log.info( f"Opened cache dataset from {self.dspath}, nvars = {len(self.data.data_vars)}")
			except KeyError as ex:
				print(f"Error reading file: {self.dspath}: {ex}")
		else:
			print( f"Cache file not found: {self.dspath}")

	@property
	def dspath(self) -> str:
		return f"{self.rootdir}/{self.file_paths[self.file_sort[self.ifile]]}"

	def nelem_in_file(self) -> int:
		return self.data['y'].shape[0]

	def get_next_element(self) -> Optional[RDict]:
		self.update_element()
		try:
			y: np.ndarray = self.data[ 'y' ].values[self.ielement]
			t: np.ndarray = self.data[ 't' ].values[self.ielement]
			p = self.data['p'].values[self.ielement]
			nan_mask = np.isnan(y)
			y = y[~nan_mask]
			t = t[~nan_mask]
			return dict( t=t, y=y, p=p )
		except KeyError as ex:
			print(f"\n    Error getting elem-{self.ielement} from dataset({self.dspath}): vars = {list(self.data.data_vars.keys())}\n")
			raise ex

class  SinusoidDataLoader(DataLoader):

	def __init__(self, cfg: DictConfig, **kwargs):
		DataLoader.__init__( self,
			dataset= SinusoidDataset(cfg),
			collate_fn= self.build_batch,
			**kwargs)

	@classmethod
	def build_batch(cls, batch: List[RDict]) -> RDict:
		t, y, p, stype, result, slen = [], [], [], [], {}, 1e10
		for elem in batch:
			if elem is not None:
				t.append(elem['t'])
				y.append(elem['y'])
				p.append(elem['p'])
				slen = min(elem['y'].size, slen)
		result['t'] = merge(t, slen)
		result['y'] = merge(y, slen)
		result['period'] = np.array(p)
		return result

