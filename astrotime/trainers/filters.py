from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal, Optional
from astrotime.util.logging import exception_handled, log_timing, shp
import time, numpy as np, xarray as xa, torch
from astrotime.util.math import Array
from torch import Tensor
import logging
log = logging.getLogger()

class TrainingFilter(object):

	def __init__(self, mparms: Dict[str, Any]=None, **custom_parms):
		super().__init__()
		self.parms = dict( **mparms, **custom_parms ) if (mparms is not None) else custom_parms

	def __setattr__(self, key: str, value: Any) -> None:
		if ('parms' in self.__dict__.keys()) and (key in self.parms.keys()):
			self.parms[key] = value
		else:
			super(TrainingFilter, self).__setattr__(key, value)

	def __getattr__(self, key: str) -> Any:
		if 'parms' in self.__dict__.keys() and (key in self.parms.keys()):
			return self.parms[key]
		return super(TrainingFilter, self).__getattribute__(key)

	def apply(self, x: Array, y: Array, dim: int) -> Tuple[Array,Array]:
		raise NotImplemented(f"Abstract method 'apply' of base class {type(self).__name__} not implemented")

class RandomDownsample(TrainingFilter):

	def __init__(self,  mparms: Dict[str, Any]=None, **custom_parms):
		super().__init__( mparms,**custom_parms)

	def _downsample(self, x: np.ndarray, y: np.ndarray, dim: int ) -> Tuple[np.ndarray, np.ndarray]:
		mask = np.random.rand(x.shape[dim]) > self.sparsity
		return np.compress(mask, x, dim), np.compress(mask, y, dim)

	def _t_downsample(self, x: Tensor, y: Tensor, dim: int) -> Tuple[Tensor, Tensor]:
		mask: np.ndarray = ( torch.rand(x.shape[dim]) < self.sparsity )
		if   dim == 0: return x[mask,...], y[mask,...]
		elif dim == 1: return x[:,mask,...], y[:,mask,...]
		elif dim == 2: return x[:,:,mask,...], y[:,:,mask,...]
		else: raise Exception( f"Unsupported dim: {dim}")

	@exception_handled
	def apply(self, x: Array, y: Array, dim: int) -> Tuple[Array,Array]:
		if self.sparsity == 0.0: return x,y
		if type(x) is Tensor: return self._t_downsample(x,y,dim)
		else:                 return self._downsample(x,y,dim)

class Norm(TrainingFilter):

	def __init__(self, mparms: Dict[str, Any], **custom_parms):
		super().__init__(mparms,**custom_parms)

	@exception_handled
	def apply(self, x: Array, y: Array, dim: int) -> Tuple[Array,Array]:
		if type(x) is Tensor:
			if   self.norm == "mean":  return x,  y/y.mean(dim=dim,keepdim=True)
			elif self.norm == "std":   return x,  (y-y.mean(dim=dim,keepdim=True))/y.std(dim=dim,keepdim=True)
			elif self.norm == "max":   return x,  y/y.max(dim=dim,keepdim=True)
		else:
			if   self.norm == "mean":  return x,  y/y.mean(axis=dim,keepdims=True)
			elif self.norm == "std":   return x,  (y-y.mean(axis=dim,keepdims=True))/y.std(axis=dim,keepdims=True)
			elif self.norm == "max":   return x,  y/y.max(axis=dim,keepdims=True)
		raise Exception( f"Unsupported normalization type: {self.ntype}")

# class GaussianNoise(TrainingFilter):
#
# 	def __init__(self,  mparms: Dict[str, Any], **custom_parms):
# 		super().__init__(mparms,**custom_parms)
#
# 	def _add_noise(self, x: np.ndarray, y: np.ndarray ) -> Tuple[np.ndarray, np.ndarray]:
# 		nsr = np.interp(self.noise, self.param_domain, self.logspace)
# 		spower = np.mean(y*y)
# 		std = np.sqrt(spower * nsr)
# 		log.info(f"Add noise: noise={self.noise:.3f}, nsr={nsr:.3f}, spower={spower:.3f}, std={std:.3f}")
# 		self.current_noise = np.random.normal(0.0, std, size=y.shape[1])
# 		y = y + self.current_noise
# 		return x, y
#
# 	@exception_handled
# 	def apply(self, x: np.ndarray, y: np.ndarray ) -> Tuple[np.ndarray, np.ndarray]:
# 		if self.noise == 0.0: return x,y
# 		return self._add_noise(x,y)
#
# class PeriodicGap(TrainingFilter):
#
# 	def __init__(self,  mparms: Dict[str, Any], **custom_parms):
# 		super().__init__(mparms,**custom_parms)
#
# 	def _mask_gaps(self,  x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
# 		tsize = x.shape[-1]
# 		xspace= np.linspace( 0.0, 1.0,tsize )
# 		gmask = np.full( tsize, True, dtype=bool )
# 		p: float = self.gap_period
# 		while p < 1.0:
# 			gd = self.gap_size*self.gap_period/2.0
# 			xp0 = max( (p-gd), 0.0 )
# 			xp1 = min( (p+gd), 1.0 )
# 			gmask[ (xspace >= xp0) & (xspace < xp1) ] = False
# 			p = p + self.gap_period
# 		return x[...,gmask], y[...,gmask]
#
# 	@exception_handled
# 	def apply(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
# 		if self.gap_size == 0.0: return (x,y)
# 		return self._mask_gaps(x,y)
#
# class Smooth(TrainingFilter):
#
# 	def __init__(self,  mparms: Dict[str, Any], **custom_parms):
# 		super().__init__(mparms,**custom_parms)
#
# 	@exception_handled
# 	def apply(self, batch: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
# 		if self.smoothing == 0.0: return batch
# 		y: np.ndarray = batch['y']
# 		sigma = self.smoothing * y.shape[1] * 0.02
# 		batch['y'] = self.smooth(y, sigma)
# 		return batch
#
# 	@classmethod
# 	def smooth(cls, x: np.ndarray, sigma: float) -> np.ndarray:
# 		return gaussian_filter1d(x, sigma=sigma, mode='wrap')
#
# class Envelope(TrainingFilter):
#
# 	def __init__(self,  mparms: Dict[str, Any], **custom_parms):
# 		super().__init__(mparms,**custom_parms)
#
# 	@exception_handled
# 	def apply(self, batch: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
# 		if self.envelope == 0.0: return batch
# 		if self.envelope > 1.0:
# 			tmax = (1 + (self.envelope-1)*(self.nperiods-1))*2*np.pi
# 			theta: np.ndarray = np.linspace(0.0, tmax, num=batch['y'].shape[-1] )
# 			env: np.ndarray = 0.5 - 0.5*np.cos(theta)
# 		else:
# 			theta: np.ndarray = np.linspace(0.0, 2*np.pi, num=batch['y'].shape[-1] )
# 			env: np.ndarray = 1 - 0.5*self.envelope*(1 + np.cos(theta))
# 		batch['y'] = batch['y'] * env
# 		return batch
#
#
#
#
#
#
