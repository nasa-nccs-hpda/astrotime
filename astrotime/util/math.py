import numpy as np
from typing import List, Optional, Dict, Type, Any
import torch
from torch import Tensor

def is_power_of_two(n: int) -> bool:
	if n <= 0: return False
	return (n & (n - 1)) == 0

def logspace(start: float, stop: float, N: int) -> np.ndarray:
	return np.pow(10.0, np.linspace(np.log10(start), np.log10(stop), N))

def shp( x ) -> List[int]:
	return list(x.shape)

def tmean(x: Tensor) -> float:
	return torch.mean(x).item()

def tstd(x: Tensor) -> float:
	return torch.std(x).item()

def tmag(x: Tensor) -> float:
	return torch.max(x).item()

def tnorm(x: Tensor, dim: int) -> Tensor:
	m: Tensor = x.mean( dim=dim, keepdim=True)
	s: Tensor = torch.std( x, dim=dim, keepdim=True)
	return (x - m) / s

def npnorm(x: np.ndarray, dim: int) -> np.ndarray:
	m: np.ndarray = x.mean( axis=dim, keepdims=True)
	s: np.ndarray = x.std( axis=dim, keepdims=True)
	return (x - m) / s