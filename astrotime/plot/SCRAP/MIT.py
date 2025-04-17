import logging, numpy as np
import xarray as xa
from plot.param import STIntParam, STFloatParam
from astrotime.loaders.MIT import MITLoader
from torch import nn, optim, Tensor, FloatTensor
from plot.base import SignalPlot, bounds
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.backend_bases import KeyEvent, MouseEvent, MouseButton
from astrotime.util.logging import exception_handled
from astrotime.encoders.embedding import EmbeddingLayer
from astrotime.util.math import npnorm
from typing import List, Optional, Dict, Type, Union, Tuple, Any, Set
from astrotime.util.math import tnorm
log = logging.getLogger()

def tolower(ls: Optional[List[str]]) -> List[str]:
	return [a.lower() for a in ls] if (ls is not None) else []

def znorm(ydata: np.ndarray) -> np.ndarray:
	y0,y1 = ydata.min(), ydata.max()
	return (ydata-y0)/(y1-y0)

def unorm(ydata: np.ndarray) -> np.ndarray:
	y0,y1 = ydata.min(), ydata.max()
	z = (y1-ydata)/(y1-y0)
	return 1 - 2*z

class PeriodMarkers:

	def __init__(self, name: str, ax: Axes, **kwargs):
		self.name = name
		self.log = logging.getLogger()
		self.ax: Axes = ax
		self.origin: float = None
		self.period: float = None
		self.markers: List[Line2D] = []
		self.yrange = kwargs.get('yrange', (-1,1) )
		self.npm: int = kwargs.get('npm', 25 )
		self.color: str = kwargs.get('color', 'green' )
		self.alpha: float = kwargs.get('alpha', 0.5 )
		self.linestyle: str = kwargs.get('linestyle', '-')

	def set_visible(self, b: bool):
		for m in self.markers:
			m.set_visible(b)

	def update(self, origin: float, period: float = None ):
		self.origin = origin
		if period is not None:
			self.period = period
		self.refresh()

	@property
	def fig(self):
		return self.ax.get_figure()

	def refresh( self ):
		for pid in range(0,self.npm):
			tval = self.origin + (pid-self.npm//2) * self.period
			if pid >= len(self.markers):  self.markers.append( self.ax.axvline( tval, self.yrange[0], self.yrange[1], color=self.color, linestyle=self.linestyle, alpha=self.alpha) )
			else:                         self.markers[pid].set_xdata([tval,tval])
		self.log.info( f"  PeriodMarkers({self.name}:{id(self):02X}).refresh( origin={self.origin:.2f}, period={self.period:.2f} ), #markers = {len(self.markers)} -- --- -- ")
