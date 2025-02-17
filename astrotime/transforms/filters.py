
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal, Optional
from astrotime.util.logging import lgm, exception_handled, log_timing, shp
from scipy.ndimage.filters import uniform_filter1d, gaussian_filter1d
import time, numpy as np, xarray as xa, math
import tensorflow as tf

class TrainingFilter(object):

	def __init__(self, device, mparms: Dict[str, Any], **custom_parms):
		super().__init__()
		self.device = device
		self.parms = dict( **mparms, **custom_parms )
		self.freq = None

	def __setattr__(self, key: str, value: Any) -> None:
		if ('parms' in self.__dict__.keys()) and (key in self.parms.keys()):
			self.parms[key] = value
		else:
			super(TrainingFilter, self).__setattr__(key, value)

	def __getattr__(self, key: str) -> Any:
		if 'parms' in self.__dict__.keys() and (key in self.parms.keys()):
			return self.parms[key]
		return super(TrainingFilter, self).__getattribute__(key)

	def apply(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		raise NotImplemented(f"Abstract method 'apply' of base class {type(self).__name__} not implemented")

class GaussianNoise(TrainingFilter):

	def __init__(self, device, mparms: Dict[str, Any], **custom_parms):
		super().__init__(device,mparms,**custom_parms)

	def _add_noise(self, batch: Dict[str,np.ndarray]):
		nsr = np.interp(self.noise, self.param_domain, self.logspace)
		spower = np.mean(batch['y'] * batch['y'])
		std = np.sqrt(spower * nsr)
		lgm().log(f"Add noise: noise={self.noise:.3f}, nsr={nsr:.3f}, spower={spower:.3f}, std={std:.3f}")
		self.current_noise = np.random.normal(0.0, std, size=batch['y'].shape[1])
		batch['y'] = batch['y'] + self.current_noise

	@exception_handled
	def apply(self, dset: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
		if self.noise == 0.0: return dset
		self._add_noise(dset)
		return dset

class RandomDownsample(TrainingFilter):

	def __init__(self, device, mparms: Dict[str, Any], **custom_parms):
		super().__init__(device,mparms,**custom_parms)

	def _downsample(self, x: np.ndarray, y: np.ndarray ) -> Tuple[np.ndarray, np.ndarray]:
		mask = np.random.rand(x.shape[-1]) > self.sparsity
		return x[...,mask], y[...,mask]

	@exception_handled
	def apply(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		if self.sparsity == 0.0: return x,y
		return self._downsample(x,y)

class PeriodicGap(TrainingFilter):

	def __init__(self, device, mparms: Dict[str, Any], **custom_parms):
		super().__init__(device,mparms,**custom_parms)

	def _mask_gaps(self,  x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		tsize = x.shape[-1]
		xspace= np.linspace( 0.0, 1.0,tsize )
		gmask = np.full( tsize, True, dtype=bool )
		p: float = self.gap_period
		while p < 1.0:
			gd = self.gap_size*self.gap_period/2.0
			xp0 = max( (p-gd), 0.0 )
			xp1 = min( (p+gd), 1.0 )
			gmask[ (xspace >= xp0) & (xspace < xp1) ] = False
			p = p + self.gap_period
		return x[...,gmask], y[...,gmask]

	@exception_handled
	def apply(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		if self.gap_size == 0.0: return (x,y)
		return self._mask_gaps(x,y)

class Smooth(TrainingFilter):

	def __init__(self, device, mparms: Dict[str, Any], **custom_parms):
		super().__init__(device,mparms,**custom_parms)

	@exception_handled
	def apply(self, batch: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
		if self.smoothing == 0.0: return batch
		y: np.ndarray = batch['y']
		sigma = self.smoothing * y.shape[1] * 0.02
		batch['y'] = self.smooth(y, sigma)
		return batch

	@classmethod
	def smooth(cls, x: np.ndarray, sigma: float) -> np.ndarray:
		return gaussian_filter1d(x, sigma=sigma, mode='wrap')

class Envelope(TrainingFilter):

	def __init__(self, device, mparms: Dict[str, Any], **custom_parms):
		super().__init__(device,mparms,**custom_parms)

	@exception_handled
	def apply(self, batch: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
		if self.envelope == 0.0: return batch
		if self.envelope > 1.0:
			tmax = (1 + (self.envelope-1)*(self.nperiods-1))*2*np.pi
			theta: np.ndarray = np.linspace(0.0, tmax, num=batch['y'].shape[-1] )
			env: np.ndarray = 0.5 - 0.5*np.cos(theta)
		else:
			theta: np.ndarray = np.linspace(0.0, 2*np.pi, num=batch['y'].shape[-1] )
			env: np.ndarray = 1 - 0.5*self.envelope*(1 + np.cos(theta))
		batch['y'] = batch['y'] * env
		return batch






