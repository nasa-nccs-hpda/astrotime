import random, time, torch, numpy as np
from typing import Any, Dict, List, Optional, Tuple
from astrotime.encoders.base import Encoder
from torch import Tensor, device
from omegaconf import DictConfig, OmegaConf
from astrotime.util.math import tmean, tstd, tmag, npnorm, shp
import logging
log = logging.getLogger("astrotime")


class PolyCoeffEncoder(Encoder):

	def __init__(self, cfg: DictConfig, device: device ):
		super(PolyCoeffEncoder, self).__init__( cfg, device )
		self.chan_first = True

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
				result = self.get_coeff( np.concatenate( [xs,ys], axis=0 ) )
				x1.append( result[0] ); y1.append( result[1] )
			X = torch.FloatTensor( np.concatenate( x1, axis=0 ) ).to(self.device)
			Y = torch.FloatTensor( np.concatenate( y1, axis=0 ) ).to(self.device)
			return X, Y

	def encode_batch(self, xb: np.ndarray, yb: np.ndarray ) -> Tuple[Tensor,Tensor]:
		with (self.device):
			x,y = self.apply_filters(xb,yb, dim=1)
			x0: int = random.randint(0,  x.shape[1]-self.series_length )
			y: np.ndarray =  npnorm( y[:,x0:x0 + self.series_length], dim=0)
			x: np.ndarray =  x[:,x0:x0 + self.series_length]
			result = np.apply_along_axis( self.get_coeff, axis=-1, arr=np.concatenate( [x,y], axis=0 ) )
			X = torch.FloatTensor( result[0] ).to(self.device)
			Y = torch.FloatTensor( result[1] ).to(self.device)
			if self.chan_first: Y = Y.transpose(1,2)
			log.info( f" ENCODED BATCH: x{list(xb.shape)} y{list(yb.shape)} -> X{list(X.shape)} Y{list(Y.shape)}")
			return X, Y

	def get_coeff(self, xy: np.ndarray ) -> np.ndarray:
		pass