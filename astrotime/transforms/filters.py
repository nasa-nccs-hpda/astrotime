
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal, Optional
from astrotime.util.logging import lgm, exception_handled, log_timing, shp
from scipy.ndimage.filters import uniform_filter1d, gaussian_filter1d
import time, numpy as np, xarray as xa, math
import tensorflow as tf

class TrainingFilter(object):

	def __init__(self, mparms: Dict[str, Any], **custom_parms):
		super().__init__()
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

	def apply(self, x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
		raise NotImplemented(f"Abstract method 'apply' of base class {type(self).__name__} not implemented")

class GaussianNoise(TrainingFilter):

	def __init__(self,  mparms: Dict[str, Any], **custom_parms):
		super().__init__(mparms,**custom_parms)

	def _add_noise(self, x: tf.Tensor, y: tf.Tensor ) -> Tuple[tf.Tensor, tf.Tensor]:
		nsr = np.interp(self.noise, self.param_domain, self.logspace)
		spower = np.mean(y*y)
		std = np.sqrt(spower * nsr)
		lgm().log(f"Add noise: noise={self.noise:.3f}, nsr={nsr:.3f}, spower={spower:.3f}, std={std:.3f}")
		self.current_noise = np.random.normal(0.0, std, size=y.shape[1])
		y = y + self.current_noise
		return x, y

	@exception_handled
	def apply(self, x: tf.Tensor, y: tf.Tensor ) -> Tuple[tf.Tensor, tf.Tensor]:
		if self.noise == 0.0: return x,y
		return self._add_noise(x,y)

class RandomDownsample(TrainingFilter):

	def __init__(self,  mparms: Dict[str, Any], **custom_parms):
		super().__init__(mparms,**custom_parms)

	def _downsample(self, x: tf.Tensor, y: tf.Tensor ) -> Tuple[tf.Tensor, tf.Tensor]:
		mask: tf.Tensor = tf.math.less( tf.random.uniform(x.shape[1]), self.sparsity )
		return tf.boolean_mask(x,mask,1), tf.boolean_mask(y,mask,1)

	@exception_handled
	def apply(self, x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
		if self.sparsity == 0.0: return x,y
		return self._downsample(x,y)

class PeriodicGap(TrainingFilter):

	def __init__(self,  mparms: Dict[str, Any], **custom_parms):
		super().__init__(mparms,**custom_parms)

	def _mask_gaps(self,  x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
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
	def apply(self, x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
		if self.gap_size == 0.0: return (x,y)
		return self._mask_gaps(x,y)

class Smooth(TrainingFilter):

	def __init__(self,  mparms: Dict[str, Any], **custom_parms):
		super().__init__(mparms,**custom_parms)

	@exception_handled
	def apply(self, batch: Dict[str,tf.Tensor]) -> Dict[str,tf.Tensor]:
		if self.smoothing == 0.0: return batch
		y: tf.Tensor = batch['y']
		sigma = self.smoothing * y.shape[1] * 0.02
		batch['y'] = self.smooth(y, sigma)
		return batch

	@classmethod
	def smooth(cls, x: tf.Tensor, sigma: float) -> tf.Tensor:
		return gaussian_filter1d(x, sigma=sigma, mode='wrap')

class Envelope(TrainingFilter):

	def __init__(self,  mparms: Dict[str, Any], **custom_parms):
		super().__init__(mparms,**custom_parms)

	@exception_handled
	def apply(self, batch: Dict[str,tf.Tensor]) -> Dict[str,tf.Tensor]:
		if self.envelope == 0.0: return batch
		if self.envelope > 1.0:
			tmax = (1 + (self.envelope-1)*(self.nperiods-1))*2*np.pi
			theta: tf.Tensor = np.linspace(0.0, tmax, num=batch['y'].shape[-1] )
			env: tf.Tensor = 0.5 - 0.5*np.cos(theta)
		else:
			theta: tf.Tensor = np.linspace(0.0, 2*np.pi, num=batch['y'].shape[-1] )
			env: tf.Tensor = 1 - 0.5*self.envelope*(1 + np.cos(theta))
		batch['y'] = batch['y'] * env
		return batch






