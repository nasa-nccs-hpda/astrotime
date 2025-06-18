import numpy as np
from typing import List, Optional, Dict, Type, Any
import torch, xarray as xa
from torch import Tensor
Array = Tensor | np.ndarray

def is_power_of_two(n: int) -> bool:
	if n <= 0: return False
	return (n & (n - 1)) == 0

def logspace(start: float, stop: float, N: int) -> np.ndarray:
	return np.power(10.0, np.linspace(np.log10(start), np.log10(stop), N))

# def log2space__(start: float, stop: float, N: int) -> np.ndarray:
# 	return np.power(2.0, np.linspace(np.log2(start), np.log2(stop), N))

def l2space( f0: float, noctaves: int, nfo: int) -> np.ndarray:
	exp: np.ndarray = np.array(range(noctaves*nfo)) / nfo
	return f0 * np.power( 2.0, exp )

def shp( x ) -> List[int]:
	return list(x.shape)

def tmean(x: Tensor) -> float:
	return torch.mean(x).item()

def tstd(x: Tensor) -> float:
	return torch.std(x).item()

def tmag(x: Tensor) -> float:
	return torch.max(x).item()

def tnorm(x: Tensor, dim: int=0) -> Tensor:
	m: Tensor = x.mean( dim=dim, keepdim=True)
	s: Tensor = torch.std( x, dim=dim, keepdim=True)
	return (x - m) / (s + 1e-4)

def npnorm(x: np.ndarray, dim: int) -> np.ndarray:
	m: np.ndarray = x.mean( axis=dim, keepdims=True)
	s: np.ndarray = x.std( axis=dim, keepdims=True)
	return (x - m) / (s + 1e-4)

def nnan(x: Array) -> int:
	if type(x) is xa.DataArray: x = x.values
	return np.sum(np.isnan(x)) if (type(x) is np.ndarray) else torch.sum(torch.isnan(x)).item()

def hasNaN(x: Array) -> bool:
	if type(x) is xa.DataArray: x = x.values
	return np.isnan(x).any() if (type(x) is np.ndarray) else torch.isnan(x).any()

def nan_mask(x: np.ndarray, axis=0) -> np.ndarray:
	return ~np.any( np.isnan(x), axis=axis )