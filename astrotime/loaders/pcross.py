import time, os, math, numpy as np, xarray as xa
from typing import List, Optional, Dict, Type, Union, Tuple
from omegaconf import DictConfig
import logging, random

class PlanetCrossingDataGenerator:

	def __init__(self, cfg: DictConfig ):
		super().__init__()
		self.log = logging.getLogger()
		self.cfg = cfg
		self.arange: Tuple[float,float] = cfg.arange
		self.hrange: Tuple[float, float] = cfg.hrange
		self.noise = cfg.noise
		self.q2 = math.sqrt(math.log(2))

	def get_element(  self, time: xa.DataArray, y: xa.DataArray ) -> xa.DataArray:
		period: float  = y.attrs['period']
		a: float =     random.uniform( self.arange[0], self.arange[1] )
		h: float =     random.uniform( self.hrange[0], self.hrange[1] )
		noise: np.ndarray = np.random.normal(0.0, self.noise, time.shape[0])
		tvals: np.ndarray = time.values
		dt = np.mod(tvals,period) - period/2
		pcross: np.ndarray =  (1 - h * np.exp(-(a*dt/period) ** 2)) + noise
		signal: xa.DataArray = y.copy( data=pcross )
		signal.attrs.update(width=a, mag=h, noise=self.noise)
		return signal

	def process_batch(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
		tvals, y, p = batch['t'], batch['y'], batch['p']
		a: float =     random.uniform( self.arange[0], self.arange[1] )
		h: float =     random.uniform( self.hrange[0], self.hrange[1] )
		period = p[:,None]
		noise: np.ndarray = np.random.normal(0.0, self.noise, tvals.shape)
		self.log.info( f"PlanetCrossingDataGenerator.process_batch: a={a}, h={h}, period{period.shape}, tvals{tvals.shape}, y{y.shape}")
		dt = np.mod(tvals,period) - period
		pcross: np.ndarray =  (1 - h * np.exp(-(a*dt/period) ** 2)) + noise
		return dict( t=tvals, y=pcross, p=p )

