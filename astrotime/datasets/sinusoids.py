from .base import AstrotimeDataset, RDict, TRDict
import time, torch, numpy as np, xarray as xa
from typing import List, Optional, Dict, Type, Union, Tuple, Any
from astrotime.util.logging import exception_handled
from astrotime.util.series import TSet
from torch.utils.data import DataLoader
from torch import Tensor, device
from omegaconf import DictConfig, OmegaConf
import logging, random, os, math

def merge( arrays: List[np.ndarray], slen: int ) -> np.ndarray:
	if len( arrays ) == 0: raise IndexError
	return np.stack( [ array[:slen] for array in arrays ], axis=0 )

class SinusoidDataset(AstrotimeDataset):

	def __init__(self, cfg: DictConfig, tset: TSet ):
		AstrotimeDataset.__init__(self, cfg, tset)
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
		try:
			return f"{self.rootdir}/{self._files[self.file_sort[self.ifile]]}"
		except IndexError as err:
			print(f"IndexError getting dspath: ifile={self.ifile}, nfiles={self.nfiles}")
			raise err


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

	def __init__(self, cfg: DictConfig, tset: TSet ):
		DataLoader.__init__( self,
			dataset= SinusoidDataset(cfg, tset),
			collate_fn= self.build_batch,
			batch_size= cfg.batch_size,
			num_workers= cfg.num_workers,
			pin_memory= cfg.pin_memory,
			persistent_workers= cfg.persistent_workers,
			prefetch_factor=4 )

	@classmethod
	def build_batch(cls, batch: List[RDict]) -> TRDict:
		tl, yl, pl, slen = [], [], [], 1e10
		for elem in batch:
			if elem is not None:
				tl.append(elem['t'])
				yl.append(elem['y'])
				pl.append(elem['p'])
				slen = min(elem['y'].size, slen)
		t: np.ndarray = merge(tl, slen)
		y: np.ndarray = merge(yl, slen)
		p: Tensor = torch.tensor(pl, dtype=torch.float32)
		z: Tensor = cls.to_tensor(t, y)
		return dict( input=z, target=1/p )

	@classmethod
	def to_tensor(cls, x: np.ndarray, y: np.ndarray) -> Tensor:
		Y: Tensor = torch.FloatTensor(y)
		X: Tensor = torch.FloatTensor(x)
		return torch.stack((X, Y), dim=1)

