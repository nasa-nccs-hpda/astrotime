from torch import nn
import torch, math, numpy as np
from astrotime.util.math import shp
from typing import List, Tuple, Mapping
from astrotime.util.math import tmean, tstd, tmag, npnorm, shp
from astrotime.util.tensor_ops import check_nan
from astrotime.encoders.embedding import EmbeddingLayer
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

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		result = self.f0 * (torch.pow(2, x) - 1)
		# print(f"ExpU(xmax={self.xmax:.3f}): xm={x.max().item():.3f} xrm={xr.max().item():.3f} xsm={xs.max().item():.3f} rm={result.max().item():.3f}",flush=True)
		# print(f"ExpU: x{shp(x)} result{shp(result)}")
		return result

class BExpU(nn.Module):

	def __init__(self, cfg: DictConfig ) -> None:
		super().__init__()
		self.f0: float = cfg.base_freq
		self.xmax = cfg.noctaves+1
		self.relu = nn.ReLU()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		xr = self.relu( x-self.xmax )
		xs = x - xr
		result = self.f0 * (torch.pow(2, xs) - 1)
		# print(f"ExpU(xmax={self.xmax:.3f}): xm={x.max().item():.3f} xrm={xr.max().item():.3f} xsm={xs.max().item():.3f} rm={result.max().item():.3f}",flush=True)
		# print(f"ExpU: x{shp(x)} result{shp(result)}")
		return result

class ExpLoss(nn.Module):
	def __init__(self, cfg: DictConfig):
		super().__init__()
		self.f0: float = cfg.base_freq

	def forward(self, product: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		zf = self.f0 * 1.01
		pf: torch.Tensor = (product + zf)
		tf: torch.Tensor = (target  + zf)
		ptr: torch.Tensor = torch.log2(pf/tf)
		#print(f"ExpLoss(zf={zf:.6f}): y{shp(product)}({product.min().item():.6f} -> {product.max().item():.3f}), pf{shp(pf)}({pf.min().item():.6f} -> {pf.max().item():.3f}), tf{shp(tf)}({tf.min().item():.3f} -> {tf.max().item():.3f}), ptr{shp(ptr)}({ptr.min().item():.3f} -> {ptr.max().item():.3f})")
		result = torch.abs( ptr ).mean()
		return result

class OctaveRegressionLoss(nn.Module):
	def __init__(self, cfg: DictConfig, embedding: EmbeddingLayer):
		super().__init__()
		self.f0: float = cfg.base_freq
		self.embedding: EmbeddingLayer = embedding

	def forward(self, product: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		octaves: torch.Tensor = self.embedding.get_octave_data()
		base_f = self.f0 * torch.pow(2, octaves)
		starg = (target/base_f) - 1
		result = torch.abs( product-starg ).mean()
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