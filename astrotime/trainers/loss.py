from torch import nn
import torch, math, numpy as np
from omegaconf import DictConfig

class HLoss(nn.Module):
	def __init__(self, cfg: DictConfig):
		super(HLoss, self).__init__()
		self.maxh = cfg.maxh
		self._h: torch.Tensor = None
		print(f"HLoss: maxh={self.maxh}")

	def h(self) -> np.ndarray:
		return self._h.cpu().numpy()

class ExpU(nn.Module):

	def __init__(self, cfg: DictConfig) -> None:
		super().__init__()
		self.f0: float = cfg.base_freq

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		result = self.f0 * (torch.pow(2, x) - 1)
		return result

class ExpLoss(nn.Module):
	def __init__(self, cfg: DictConfig):
		super(ExpLoss, self).__init__()
		self.f0: float = cfg.base_freq

	def forward(self, product: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		result = torch.abs(torch.log2((product + self.f0) / (target + self.f0))).mean()
		return result

class ElemExpLoss(nn.Module):
	def __init__(self, cfg: DictConfig):
		super(ElemExpLoss, self).__init__()
		self.f0: float = cfg.base_freq

	def forward(self, product: float, target: float) -> float:
		result = abs(math.log2((product + self.f0) / (target + self.f0)))
		return result

class ElemExpHLoss(HLoss):
	def __init__(self, cfg: DictConfig):
		super(ElemExpHLoss, self).__init__(cfg)
		self.f0: float = cfg.base_freq

	def harmonic(self, y: float, t: float) -> float:
		h: float = round(y / t) if (y > t) else 1.0 / round(t / y)
		return h if ((round(1 / h) <= self.maxh) and (h <= self.maxh)) else 1.0

	def forward(self, product: float, target: float) -> float:
		self._h: float = self.harmonic(product, target)
		result = abs(math.log2((product + self.f0) / (self._h * target + self.f0)))
		return result

class ExpHLoss(HLoss):
	def __init__(self, cfg: DictConfig):
		super(ExpHLoss, self).__init__(cfg)
		self.f0: float = cfg.base_freq
		self._harmonics = None

	def harmonic(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
		h: torch.Tensor = torch.where(y > t, torch.round(y / t), 1 / torch.round(t / y)).detach().squeeze()
		valid: torch.Tensor = torch.logical_and(torch.round(1 / h) <= self.maxh, h <= self.maxh)
		valid: torch.Tensor = torch.logical_and(valid, h > 0)
		h: torch.Tensor = torch.where(valid, h, torch.ones_like(h))
		try: self._harmonics = h if (self._harmonics is None) else torch.concat((self._harmonics, h.squeeze()))
		except RuntimeError: print(f"ExpHLoss.harmonic.concat: h={h}, harmonics={self._harmonics}")
		return h

	def forward(self, product: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		self._h: torch.Tensor = self.harmonic(product, target)
		result = torch.abs(torch.log2((product + self.f0) / (self._h * target + self.f0))).mean()
		return result

	def harmonics(self) -> np.ndarray:
		rv: torch.Tensor = self._harmonics
		self._harmonics = None
		return rv.cpu().numpy()