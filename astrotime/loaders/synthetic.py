import time, os, numpy as np, xarray as xa
from typing import List, Optional, Dict, Type, Union, Tuple
from omegaconf import DictConfig
import logging, random

class PlanetCrossingDataGenerator:

	def __init__(self, cfg: DictConfig ):
		super().__init__()
		self.cfg = cfg
		self.arange: Tuple[float,float] = cfg.arange
		self.hrange: Tuple[float, float] = cfg.hrange
		self.noise = cfg.noise

	def get_element(  self, time: xa.DataArray, y: xa.DataArray ) -> xa.DataArray:
		period: float  = y.attrs['period']
		yheight  = np.median( y.values )
		a: float =     random.uniform( self.arange[0], self.arange[1] ) * period
		h: float =     random.uniform( self.hrange[0], self.hrange[1] ) * yheight
		phase: float = random.uniform(0, period )
		time_values: np.ndarray = time.values
		taus: np.ndarray = np.arange( phase+time_values.min(), time_values.max(), period )
		crossing: np.ndarray = h * np.exp(-(a * (time_values[:,None] - taus[None,:])) ** 2)
		noise: np.ndarray = np.random.normal(0.0, self.noise*yheight, crossing.shape[0])
		crossings: np.ndarray =  yheight - crossing.sum(axis=1) + noise
		signal: xa.DataArray = y.copy( data= 2*self.hrange[1] - crossings )
		signal.attrs.update( width=a, mag=h, phase=phase )
		return signal

