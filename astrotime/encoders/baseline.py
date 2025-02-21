import random, time, torch, numpy as np
from typing import Any, Dict, List, Optional, Tuple
from astrotime.encoders.base import Encoder
from torch import Tensor
from astrotime.util.math import tmean, tstd, tmag, tnorm, shp


class ValueEncoder(Encoder):

	def __init__(self, device, series_length: int, max_series_len: int ):
		super(ValueEncoder, self).__init__( device, series_length )
		self.max_series_len = max_series_len

	def encode_dset(self, dset: Dict[str,np.ndarray]) -> Tuple[Tensor,Tensor]:
		with (self.device):
			y1, x1 = [], []
			for idx, (y,x) in enumerate(zip(dset['y'],dset['x'])):
				x,y = self.apply_filters(x,y,dim=0)
				x0: int = random.randint(0, self.max_series_length - self.series_length)
				ys: Tensor = torch.from_numpy( y[x0:x0 + self.series_length] ).to(self.device)
				xs: Tensor = torch.from_numpy(x[x0:x0 + self.series_length] ).to(self.device)
				y1.append( torch.unsqueeze( tnorm(ys, dim=0), dim=0) )
				x1.append( torch.unsqueeze( xs, dim=0) )
			Y, X = torch.concatenate(y1, dim=0), torch.concatenate(x1, dim=0)
			if Y.ndim == 2: Y = torch.unsqueeze(Y, dim=2)
			return X, Y

	def encode_batch(self, x: np.ndarray, y: np.ndarray ) -> Tuple[Tensor,Tensor]:
		with (self.device):
			x,y = self.apply_filters(x,y, dim=1)
			x0: int = random.randint(0, self.max_series_length - self.series_length )
			Y: Tensor = torch.from_numpy(y[x0:x0 + self.series_length]).to(self.device)
			X: Tensor = torch.from_numpy(x[x0:x0 + self.series_length]).to(self.device)
			Y = tnorm(Y,dim=1)
			if Y.ndim == 2: Y = torch.unsqueeze(Y, dim=2)
			return X, Y