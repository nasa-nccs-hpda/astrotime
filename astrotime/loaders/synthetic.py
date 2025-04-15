import time, os, numpy as np, xarray as xa
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

	def get_element(  self, time: xa.DataArray, y: xa.DataArray ) -> xa.DataArray:
		period: float  = y.attrs['period']
		yheight  = np.median( y.values )
		a: float =     random.uniform( self.arange[0], self.arange[1] )
		h: float =     random.uniform( self.hrange[0], self.hrange[1] ) * yheight
		phase: float = random.uniform(0, period )
		tvals: np.ndarray = time.values
		taus: np.ndarray = np.arange( phase+tvals.min(), tvals.max(), period )
		dt : np.ndarray = (tvals[:,None] - taus[None,:])/period
		crossing: np.ndarray = h * np.exp(-(a*dt) ** 2)
		noise: np.ndarray = np.random.normal(0.0, self.noise*yheight, crossing.shape[0])
		crossings: np.ndarray =  crossing.sum(axis=1)
		signal: xa.DataArray = y.copy( data= 2*self.hrange[1] - crossings  + noise )
		signal.attrs.update( width=a, mag=h, phase=phase )
		self.log.info(f"PlanetCrossingDataGenerator.get_element:\n ----> time{time.shape} y{y.shape} period={period:.2f} trange={tvals.max()-tvals.min():.2f} yheight={yheight:.2f} a={a:.2f} h={h:.2f} "
		              f"phase={phase:.2f} taus{taus.shape} dt{dt.shape}({dt.mean():.2f},{dt.std():.2f}) crossing{crossing.shape}  noise{noise.shape} crossings{crossings.shape} signal{signal.shape}")
		return signal

