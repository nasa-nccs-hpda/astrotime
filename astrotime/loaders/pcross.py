import time, os, math, numpy as np, xarray as xa
from typing import List, Optional, Dict, Type, Union, Tuple
from omegaconf import DictConfig
import logging, random

class PlanetCrossingDataGenerator:

	def __init__(self, cfg: DictConfig ):
		super().__init__()
		self.log = logging.getLogger()
		self.cfg = cfg
		self.arange: Tuple[float,float]  = cfg.get('arange',(0.01,0.1) )
		self.wrange: Tuple[float, float] = cfg.get('wrange',(0.01,0.1) )
		self.nrange: Tuple[float, float] = cfg.get('nrange',(0.01,0.1) )

	def signal(self, t: np.ndarray, p: float|np.ndarray, **kwargs ):
		a: float =  kwargs.get( 'amplitude', random.uniform( *self.arange ) )
		w: float =  kwargs.get( 'width', random.uniform( *self.wrange ) )
		n: float =  kwargs.get( 'noise', random.uniform( *self.nrange ) )
		noise: np.ndarray = np.random.normal(0.0, n, t.shape )
		dt = np.mod(t,p) - p/2
		s: np.ndarray = 1 - 3*a*np.exp(-(dt/(w*p)) ** 2)
		return s + noise

	def process_batch(self, batch: Dict[str, np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
		t, p = batch['t'], batch['p']
		batch['y'] = self.signal( t, p[:,None], **kwargs )
		return batch



