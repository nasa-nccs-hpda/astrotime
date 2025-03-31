import random, time, torch, numpy as np
from typing import Any, Dict, List, Optional, Tuple
from astrotime.encoders.base import Encoder
from torch import Tensor, device
from omegaconf import DictConfig, OmegaConf
from torch.nn.functional import normalize
from astrotime.util.math import tmean, tstd, tmag, npnorm, shp
import logging

class Expansion(Encoder):

	def __init__(self, cfg: DictConfig, device: device ):
		super(Expansion, self).__init__( cfg, device )
		self.chan_first = True
		self.stride = self.input_series_length // self.output_series_length
		self._xstride: float = None
		self._trange: float = None

	@property
	def output_series_length(self):
		return self.cfg.nstrides

	def init_xstride(self, x: np.ndarray ):
		if self._xstride is None:
			self._trange = (x[:,-1] - x[:,0]).mean()
			self._xstride =  self._trange / self.output_series_length

	def encode_batch(self, xb: np.ndarray, yb: np.ndarray ) -> Tuple[Tensor,Tensor]:
		with (self.device):
			x,y = self.apply_filters(xb,yb, dim=1)
			x0: int = random.randint(0,  x.shape[1]-self.input_series_length )
			y: np.ndarray =  npnorm( y[:,x0:x0 + self.input_series_length], dim=0)
			x: np.ndarray =  x[:,x0:x0 + self.input_series_length]
			self.init_xstride(x)
			xy = np.concat( [x,y], axis=1 )
			#print(f"encode_batch input: x{shp(x)} y{shp(y)} xy{shp(xy)} xstride={self._xstride:.4f} output_series_length={self.output_series_length} trange={self._trange:.4f}")
			Z = np.apply_along_axis( self._apply_expansion, axis=1, arr=xy )
			Y = torch.FloatTensor( Z[:,:,1:] ).to(self.device)
			X = torch.FloatTensor( Z[:,:,0]  ).to(self.device)
			#print(f"apply_along_axis result: X{shp(X)} Y{shp(Y)} ")
			Y = normalize(Y,p=1,dim=2)
			if self.chan_first: Y = Y.transpose(1,2)
			self.log.debug( f" * ENCODED BATCH: x{list(xb.shape)} y{list(yb.shape)} -> X{list(X.shape)} Y{list(Y.shape)}")
			return X, Y

	def _apply_expansion(self, xy: np.ndarray ) -> np.ndarray:
		#print( f"_apply_expansion input: xy{shp(xy)}")
		s = xy.shape[0]//2
		x,y = xy[:s], xy[s:]
		X,Y = self.get_expansion_coeff( x, y )
		Z = np.concatenate( [ X[:,None], Y.reshape(X.shape[0],-1) ], axis=1 )
		#print(f"_apply_expansion output: X{shp(X)} Y{shp(Y)} Z{shp(Z)}")
		return Z

	def get_expansion_coeff(self, x: np.ndarray, y: np.ndarray ) -> Tuple[np.ndarray,np.ndarray]:
		raise NotImplementedError("Expansion.get_expansion_coeff() not implemented")