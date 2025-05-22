import random, time, torch, numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from astrotime.encoders.base import Encoder
from torch import Tensor, device
from omegaconf import DictConfig, OmegaConf
from .embedding import EmbeddingLayer
from astrotime.util.math import tnorm
TRDict = Dict[str,Union[List[str],int,torch.Tensor]]
RDict = Dict[str,Union[List[str],int,np.ndarray]]

def bpop( batch: RDict, key: str ) -> np.ndarray:
	z: np.ndarray = batch.pop(key)
	return z if (z.ndim > 1) else z[None,:]

class IterativeEncoder(Encoder):

	def __init__(self, cfg: DictConfig, device: device ):
		super(IterativeEncoder, self).__init__( cfg, device )
		self.chan_first = True

	@property
	def nfeatures(self) -> int:
		return self.cfg.nfeatures

	def encode_batch(self, x0: np.ndarray, y0: np.ndarray ) -> Tuple[Tensor,Tensor]:
		with (self.device):
			x,y = self.apply_filters(x0,y0, dim=1)
			Y: Tensor = torch.FloatTensor(y).to(self.device)
			X: Tensor = torch.FloatTensor(x).to(self.device)
			Y = tnorm(Y, dim=1)
			if Y.ndim == 2: Y = torch.unsqueeze( Y, dim=2)
		#	self.log.debug( f" ** ENCODED BATCH: x{list(x0.shape)} y{list(y0.shape)} -> T{list(X.shape)} Y{list(Y.shape)}")
			if self.chan_first: Y = Y.transpose(1,2)
			return X, Y


class ValueEncoder(Encoder):

	def __init__(self, cfg: DictConfig, device: device ):
		super(ValueEncoder, self).__init__( cfg, device )
		self.chan_first = True

	@property
	def nfeatures(self) -> int:
		return self.cfg.nfeatures

	def encode_dset(self, dset: Dict[str,np.ndarray]) -> Tuple[Tensor,Tensor]:
		with (self.device):
			y1, x1 = [], []
			for idx, (y,x) in enumerate(zip(dset['y'],dset['x'])):
				x,y = self.apply_filters(x,y,dim=0)
				y1.append( torch.FloatTensor( y ).to(self.device) )
				x1.append( torch.FloatTensor( x ).to(self.device) )
			Y, X = torch.stack(y1, dim=0), torch.stack(x1, dim=0)
			Y = tnorm( Y, dim=1 )
			if Y.ndim == 2: Y = torch.unsqueeze(Y, dim=2)
			if self.chan_first: Y = Y.transpose(1, 2)
			return X, Y

	def encode_batch(self, x0: np.ndarray, y0: np.ndarray ) -> Tuple[Tensor,Tensor]:
		with (self.device):
			if x0.ndim == 1: x0 = x0[None,:]
			if y0.ndim == 1: y0 = y0[None,:]
			x,y = self.apply_filters(x0,y0, dim=1)
			i0: int = random.randint(0,  x.shape[1]-self.input_series_length )
			Y: Tensor = torch.FloatTensor(y[:,i0:i0 + self.input_series_length]).to(self.device)
			X: Tensor = torch.FloatTensor(x[:,i0:i0 + self.input_series_length]).to(self.device)
			Y = tnorm(Y, dim=1)
			if Y.ndim == 2: Y = torch.unsqueeze( Y, dim=2)
			self.log.debug( f" ** ENCODED BATCH: x{list(x0.shape)} y{list(y0.shape)} -> T{list(X.shape)} Y{list(Y.shape)}")
			if self.chan_first: Y = Y.transpose(1,2)
			return X, Y

	# def encode_periodic_batch(self, batch:  RDict) -> Optional[TRDict]:
	# 	with (self.device):
	# 		p: np.ndarray = batch.pop('period')
	# 		t: np.ndarray = bpop(batch,'t')
	# 		y: np.ndarray = bpop(batch,'y')
	# 		t,y = self.apply_filters(t,y, dim=1)
	# 		TL: np.ndarray = t[:,-1] - t[:,0]
	# 		TS: np.ndarray = t[:,self.input_series_length] - t[:,0]
	#
	#
	# 		i0: int = random.randint(0,  x.shape[1]-self.input_series_length )
	# 		Y: Tensor = torch.FloatTensor(y[:,i0:i0 + self.input_series_length]).to(self.device)
	# 		X: Tensor = torch.FloatTensor(x[:,i0:i0 + self.input_series_length]).to(self.device)
	# 		Y = tnorm(Y, dim=1)
	# 		if Y.ndim == 2: Y = torch.unsqueeze( Y, dim=2)
	# 		self.log.debug( f" ** ENCODED BATCH: x{list(x0.shape)} y{list(y0.shape)} -> T{list(X.shape)} Y{list(Y.shape)}")
	# 		if self.chan_first: Y = Y.transpose(1,2)
	# 		return X, Y

class ValueEmbeddingLayer(EmbeddingLayer):

	def __init__(self, cfg, device: device):
		EmbeddingLayer.__init__(self,cfg,device)

	def embed(self, ts: torch.Tensor, ys: torch.Tensor) -> Tensor:
		# print(f"     MODEL INPUT: ys{list(ys.shape)}: ({ys.min().item():.2f}, {ys.max().item():.2f}, {ys.mean().item():.2f}, {ys.std().item():.2f}) ")
		return ys

	@property
	def nfeatures(self) -> int:
		return 1

	@property
	def output_series_length(self):
		return self.cfg.series_length