from torch import nn
import torch, math, numpy as np
from typing import List, Tuple, Mapping
from astrotime.util.math import tmean, tstd, tmag, npnorm, shp
from astrotime.util.tensor_ops import check_nan
from omegaconf import DictConfig

class HLoss(nn.Module):
	def __init__(self, cfg: DictConfig,**kwargs):
		super(HLoss, self).__init__()
		self.maxh = kwargs.get('maxh',cfg.maxh)
		self.h = None
		self.rh = None

class ExpU(nn.Module):

	def __init__(self, cfg: DictConfig ) -> None:
		super().__init__()
		self.f0: float = cfg.base_freq
		self.f1 = self.f0 * torch.pow(2, cfg.noctaves+1)
		self.relu = nn.ReLU()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		xs = x - self.relu( x-self.f1 )
		result = self.f0 * (torch.pow(2, xs) - 1)
		return result

class ExpLoss(nn.Module):
	def __init__(self, cfg: DictConfig):
		super().__init__()
		self.f0: float = cfg.base_freq

	def forward(self, product: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		# print(f"ExpLoss(f0={self.f0:.3f}): product{shp(product)} ({product.min().item():.3f} -> {product.max().item():.3f}), target{shp(target)} ({target.min().item():.3f} -> {target.max().item():.3f})")
		result = torch.abs(torch.log2((product + self.f0) / (target + self.f0))).mean()
		return result

class ElemExpLoss(nn.Module):
	def __init__(self, cfg: DictConfig):
		super().__init__()
		self.f0: float = cfg.base_freq

	def forward(self, product: float, target: float) -> float:
		result = abs(math.log2((product + self.f0) / (target + self.f0)))
		return result

class ElemExpHLoss(HLoss):
	def __init__(self, cfg: DictConfig):
		super().__init__(cfg)
		self.f0: float = cfg.base_freq

	def get_harmonic(self, y: float, t: float) -> Tuple[float,float]:
		h: float = float(round(y/t)) if (y > t) else 1.0/round(t/y)
		return (h, h) if ((round(1/h) <= self.maxh) and (h <= self.maxh) and (h>0)) else (1.0, h)

	def forward(self, product: float, target: float) -> float:
		self.h, self.rh = self.get_harmonic(product, target)
		result = abs(math.log2((product + self.f0) / (self.h*target + self.f0)))
		return result

class ExpHLoss(HLoss):
	def __init__(self, cfg: DictConfig,**kwargs):
		super().__init__(cfg,**kwargs)
		self.f0: float = cfg.base_freq
		self._harmonics = None

	def get_harmonic(self, y: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
		rh: torch.Tensor = torch.where(y > t, torch.round(y / t), 1 / torch.round(t / y)).detach().squeeze()
		valid: torch.Tensor = torch.logical_and(torch.round(1 / rh) <= self.maxh, rh <= self.maxh)
		valid: torch.Tensor = torch.logical_and(valid, rh > 0)
		h: torch.Tensor = torch.where(valid, rh, torch.ones_like(rh))
		try: self._harmonics = h if (self._harmonics is None) else torch.concat((self._harmonics, h.squeeze()))
		except RuntimeError: print(f"ExpHLoss.harmonic.concat: h={h}, harmonics={self._harmonics}")
		return h, rh

	def forward(self, product: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		self.h, self.rh = self.get_harmonic(product, target)
		result = torch.abs(torch.log2((product + self.f0) / (self.h*target + self.f0))).mean()
		return result

	def harmonics(self) -> np.ndarray:
		rv: torch.Tensor = self._harmonics
		self._harmonics = None
		return rv.cpu().numpy()