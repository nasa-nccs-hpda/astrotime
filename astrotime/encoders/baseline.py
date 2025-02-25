import random, time, torch, numpy as np
from typing import Any, Dict, List, Optional, Tuple
from astrotime.encoders.base import Encoder
from torch import Tensor, device
from omegaconf import DictConfig, OmegaConf
from astrotime.util.math import tmean, tstd, tmag, tnorm, shp
from astrotime.util.logging import lgm, exception_handled


class ValueEncoder(Encoder):

	def __init__(self, cfg: DictConfig, device: device ):
		super(ValueEncoder, self).__init__( cfg, device )
		self.chan_first = True

	def encode_dset(self, dset: Dict[str,np.ndarray]) -> Tuple[Tensor,Tensor]:
		with (self.device):
			y1, x1 = [], []
			for idx, (y,x) in enumerate(zip(dset['y'],dset['x'])):
				x,y = self.apply_filters(x,y,dim=0)
				x0: int = random.randint(0, self.cfg.max_series_length - self.cfg.series_length)
				ys: Tensor = torch.FloatTensor( y[x0:x0 + self.cfg.series_length] ).to(self.device)
				xs: Tensor = torch.FloatTensor( x[x0:x0 + self.cfg.series_length] ).to(self.device)
				y1.append( torch.unsqueeze( tnorm(ys, dim=0), dim=0) )
				x1.append( torch.unsqueeze( xs, dim=0) )
			Y, X = torch.concatenate(y1, dim=0), torch.concatenate(x1, dim=0)
			if Y.ndim == 2: Y = torch.unsqueeze(Y, dim=2)
			return X, Y

	def encode_batch(self, x: np.ndarray, y: np.ndarray ) -> Tuple[Tensor,Tensor]:
		with (self.device):
			x,y = self.apply_filters(x,y, dim=1)
			x0: int = random.randint(0, self.cfg.max_series_length - self.cfg.series_length )
			Y: Tensor = torch.FloatTensor(y[:,x0:x0 + self.series_length]).to(self.device)
			X: Tensor = torch.FloatTensor(x[:,x0:x0 + self.series_length]).to(self.device)
			Y = tnorm(Y,dim=1)
			if Y.ndim == 2: Y = torch.unsqueeze(Y, dim=2)
			if self.chan_first: Y = Y.transpose(1,2)
			lgm().log( f" ENCODED BATCH: x{list(x.shape)} y{list(y.shape)} -> T{list(X.shape)} Y{list(Y.shape)}")
			return X, Y


class ValueEmbeddingLayer(torch.nn.Module):

	def __init__(self, cfg, device: device):
		torch.nn.Module.__init__(self)
		self.requires_grad_(False)
		self.device = device
		self.cfg = cfg

	def forward(self, input: torch.Tensor ):
		ys: torch.Tensor = input[:, 1:, :]
		ts: torch.Tensor = input[:, 0, :]
		return ys