import random, time, torch, numpy as np
from typing import Any, Dict, List, Optional, Tuple
from astrotime.encoders.base import Encoder
from torch import Tensor, device
from omegaconf import DictConfig, OmegaConf
from astrotime.util.math import tmean, tstd, tmag, npnorm, shp
import logging

class Expansion(Encoder):

	def __init__(self, cfg: DictConfig, device: device ):
		super(Expansion, self).__init__( cfg, device )
		self.chan_first = True
		self.nstrides: float = self.cfg.nstrides
		self.stride = self.cfg.series_length // self.nstrides
		self._xstride: float = None
		self._trange: float = None

	def encode_dset(self, dset: Dict[str,np.ndarray]) -> Tuple[Tensor,Tensor]:
		with (self.device):
			y1, x1 = [], []
			for idx, (y,x) in enumerate(zip(dset['y'],dset['x'])):
				nanmask = ~np.isnan(y)
				x, y = x[nanmask], y[nanmask]
				x,y = self.apply_filters(x,y,dim=0)
				x0: int = random.randint(0, y.shape[0] - self.cfg.series_length)
				ys: np.ndarray =  npnorm( y[x0:x0 + self.cfg.series_length], dim=0)
				xs: np.ndarray =  x[x0:x0 + self.cfg.series_length]
				xc,yc = self.get_expansion_coeff( xs,ys )
				x1.append( xc ); y1.append( yc )
			X = torch.FloatTensor( np.concatenate( x1, axis=0 ) ).to(self.device)
			Y = torch.FloatTensor( np.concatenate( y1, axis=0 ) ).to(self.device)
			return X, Y

	def init_xstride(self, x: np.ndarray ):
		if self._xstride is None:
			self._trange = (x[:,-1] - x[:,0]).mean()
			self._xstride =  self._trange / self.nstrides

	def encode_batch(self, xb: np.ndarray, yb: np.ndarray ) -> Tuple[Tensor,Tensor]:
		with (self.device):
			x,y = self.apply_filters(xb,yb, dim=1)
			x0: int = random.randint(0,  x.shape[1]-self.series_length )
			y: np.ndarray =  npnorm( y[:,x0:x0 + self.series_length], dim=0)
			x: np.ndarray =  x[:,x0:x0 + self.series_length]
			self.init_xstride(x)
			xy = np.concat( [x,y], axis=1 )
			print(f"encode_batch input: x{shp(x)} y{shp(y)} xy{shp(xy)} xstride={self._xstride:.4f} nstrides={self.nstrides} trange={self._trange:.4f}")
			Z = np.apply_along_axis( self._apply_expansion, axis=1, arr=xy )
			Y = torch.FloatTensor( Z[:,1:] ).to(self.device)
			X = torch.FloatTensor( Z[:,0]  ).to(self.device)
			print(f"apply_along_axis result: X{shp(X)} Y{shp(Y)} ")
			if self.chan_first: Y = Y.transpose(1,2)
			self.log.info( f" * ENCODED BATCH: x{list(xb.shape)} y{list(yb.shape)} -> X{list(X.shape)} Y{list(Y.shape)}")
			return X,Y

	def _apply_expansion(self, xy: np.ndarray ) -> Tuple[np.ndarray,np.ndarray]:
		print( f"_apply_expansion input: xy{shp(xy)}")
		s = xy.shape[0]//2
		x,y = xy[:s], xy[s:]
		X,Y = self.get_expansion_coeff( x, y )
		Z = np.concatenate( [ X[:,None], Y.reshape(X.shape[0],-1) ], axis=1 )
		print(f"_apply_expansion output: X{shp(X)} Y{shp(Y)} Z{shp(Z)}")
		return Z

	def get_expansion_coeff(self, x: np.ndarray, y: np.ndarray ) -> Tuple[np.ndarray,np.ndarray]:
		raise NotImplementedError("Expansion.get_expansion_coeff() not implemented")