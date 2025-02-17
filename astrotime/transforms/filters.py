
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal, Optional
from astrotime.util.logging import lgm, exception_handled, log_timing, shp
from scipy.ndimage.filters import uniform_filter1d, gaussian_filter1d
import time, numpy as np, xarray as xa, math

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

	def apply(self, batch: xa.Dataset) -> xa.Dataset:
		raise NotImplemented(f"Abstract method 'apply' of base class {type(self).__name__} not implemented")

class GaussianNoise(TrainingFilter):

	def __init__(self, device, mparms: Dict[str, Any], **custom_parms):
		super().__init__(device,mparms,**custom_parms)

	def _add_noise(self, batch: xa.Dataset):
		nsr = np.interp(self.noise, self.param_domain, self.logspace)
		spower = np.mean(batch['y'] * batch['y'])
		std = np.sqrt(spower * nsr)
		lgm().log(f"Add noise: noise={self.noise:.3f}, nsr={nsr:.3f}, spower={spower:.3f}, std={std:.3f}")
		self.current_noise = np.random.normal(0.0, std, size=batch['y'].shape[1])
		batch['y'] = batch['y'] + self.current_noise

	@exception_handled
	def apply(self, batch: xa.Dataset) -> xa.Dataset:
		if self.noise == 0.0: return batch
		self._add_noise(batch)
		return batch

class RandomDownsample(TrainingFilter):

	def __init__(self, device, mparms: Dict[str, Any], **custom_parms):
		super().__init__(device,mparms,**custom_parms)

	def _downsample(self, batch: xa.Dataset  ):
		mask = np.random.rand(batch['t'].shape[1]) > self.sparsity
		for dvar in ['t','y']:
			batch[dvar] = batch[dvar][:,mask]

	@exception_handled
	def apply(self, batch: xa.Dataset) -> xa.Dataset:
		if self.sparsity == 0.0: return batch
		self._downsample(batch)
		return batch

class PeriodicGap(TrainingFilter):

	def __init__(self, device, mparms: Dict[str, Any], **custom_parms):
		super().__init__(device,mparms,**custom_parms)

	def _mask_gaps(self, batch: xa.Dataset ):
		t0, tsize = time.time(), batch['t'].shape[1]
		xspace= np.linspace( 0.0, 1.0,tsize )
		gmask = np.full( tsize, True, dtype=bool )
		p: float = self.gap_period
		while p < 1.0:
			gd = self.gap_size*self.gap_period/2.0
			xp0 = max( (p-gd), 0.0 )
			xp1 = min( (p+gd), 1.0 )
			gmask[ (xspace >= xp0) & (xspace < xp1) ] = False
			p = p + self.gap_period
		for dvar in ['t','y']:
			batch[dvar] = batch[dvar][:,gmask]
		lgm().log( f" --> Computed periodic gaps ({self.gap_size},{self.gap_period}) in {time.time()-t0:.4f} sec")


	@exception_handled
	def apply(self, batch: xa.Dataset) -> xa.Dataset:
		if self.gap_size == 0.0: return batch
		self._mask_gaps(batch)
		return batch

class Smooth(TrainingFilter):

	def __init__(self, device, mparms: Dict[str, Any], **custom_parms):
		super().__init__(device,mparms,**custom_parms)

	@exception_handled
	def apply(self, batch: xa.Dataset):
		if self.smoothing == 0.0: return batch
		sigma = self.smoothing * batch.shape[1] * 0.02
		batch['y'] = self.smooth(batch['y'], sigma)
		return batch

	@classmethod
	def smooth(cls, x: xa.DataArray, sigma: float) ->xa.DataArray:
		data = gaussian_filter1d(x.values, sigma=sigma, mode='wrap')
		return x.copy(data=data)

class Envelope(TrainingFilter):

	def __init__(self, device, mparms: Dict[str, Any], **custom_parms):
		super().__init__(device,mparms,**custom_parms)

	@exception_handled
	def apply(self, batch: xa.Dataset) :
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






