import random, time, torch, numpy as np
from typing import Any, Dict, List, Optional, Tuple
from astrotime.encoders.base import Encoder
from torch import Tensor, device
from omegaconf import DictConfig, OmegaConf
from .embedding import EmbeddingLayer
from astrotime.util.math import tmean, tstd, tmag, tnorm, shp


class ValueEncoder(Encoder):

	def __init__(self, cfg: DictConfig, device: device ):
		super(ValueEncoder, self).__init__( cfg, device )
		self.chan_first = True

	def encode_dset(self, dset: Dict[str,np.ndarray]) -> Tuple[Tensor,Tensor]:
		with (self.device):
			y1, x1 = [], []
			for idx, (y,x) in enumerate(zip(dset['y'],dset['x'])):
				nanmask = ~np.isnan(y)
				x, y = x[nanmask], y[nanmask]
				x,y = self.apply_filters(x,y,dim=0)
				i0: int = random.randint(0, y.shape[0] - self.cfg.series_length)
				ys: Tensor = torch.FloatTensor( y[i0:i0 + self.cfg.series_length] ).to(self.device)
				xs: Tensor = torch.FloatTensor( x[i0:i0 + self.cfg.series_length] ).to(self.device)
				y1.append( torch.unsqueeze( tnorm(ys, dim=0), dim=0) )
				x1.append( torch.unsqueeze( xs, dim=0) )
			Y, X = torch.concatenate(y1, dim=0), torch.concatenate(x1, dim=0)
			if Y.ndim == 2: Y = torch.unsqueeze(Y, dim=2)
			return X, Y

	def encode_batch(self, x: np.ndarray, y: np.ndarray ) -> Tuple[Tensor,Tensor]:
		with (self.device):
			x,y = self.apply_filters(x,y, dim=1)
			i0: int = random.randint(0,  x.shape[1]-self.series_length )
			Y: Tensor = torch.FloatTensor(y[:,i0:i0 + self.series_length]).to(self.device)
			X: Tensor = torch.FloatTensor(x[:,i0:i0 + self.series_length]).to(self.device)
			Y = tnorm(Y,dim=1)
			if Y.ndim == 2: Y = torch.unsqueeze(Y, dim=2)
			if self.chan_first: Y = Y.transpose(1,2)
			self.log.info( f" ENCODED BATCH: x{list(x.shape)} y{list(y.shape)} -> T{list(X.shape)} Y{list(Y.shape)}")
			return X, Y


class ValueEmbeddingLayer(EmbeddingLayer):

	def __init__(self, cfg, device: device):
		EmbeddingLayer.__init__(self,cfg,device)

	def embed(self, ts: torch.Tensor, ys: torch.Tensor ) -> Tensor:
		self.log.debug(f" ys{list(ys.shape)} ts{list(ts.shape)}")
		return ys