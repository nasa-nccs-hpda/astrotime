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
		yheight  = np.median( y.values )
		# a: float =     random.uniform( self.arange[0], self.arange[1] )
		# h: float =     random.uniform( self.hrange[0], self.hrange[1] )
		noise: np.ndarray = np.random.normal(0.0, self.noise * yheight, time.shape[0])

		a = 30.0
		h = 0.3

		tvals: np.ndarray = time.values
		dt = np.mod(tvals,period) - period/2
		pcross: np.ndarray = yheight * (1 - h * np.exp(-(a*dt/period) ** 2)) + noise

		print( f"PlanetCrossingDataGenerator.get_element: a={a:.2f} h={h:.2f} period={period:.2f} dt{dt.shape}({dt.min():.3f},{dt.max():.3f})")
		print(f" **** ( time{time.shape} y{y.shape} ) -> pcross{pcross.shape}({pcross.min():.3f},{pcross.max():.3f})")

		signal: xa.DataArray = y.copy( data=pcross )
		signal.attrs.update(width=a, mag=h, noise=self.noise)
		return signal

