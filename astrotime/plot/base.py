import math, time, numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
from matplotlib.axes import Axes
from matplotlib.ticker import NullLocator
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from util.logging import exception_handled, log_timing
from .param import Number, Parameter, STParam, STFloatParam, STFloatValuesParam, Parameterized
import logging
log = logging.getLogger(__name__)

def bounds( y: np.ndarray ) -> Tuple[float,float]:
	ymin, ymax = y.min(), y.max()
	buff = 0.05 * (ymax - ymin)
	return ymin-buff, ymax+buff

class SignalPlot(Parameterized):

	def __init__(self, **kwargs):
		Parameterized.__init__(self)
		self.ax: Axes = None
		self.aux_axes = []
		self.annotation: Annotation = None

	@exception_handled
	def initialize(self, ax: Axes, aux_axes: List[Axes] = None):
		self.ax = ax
		self.annotation: Annotation = self.ax.annotate("", (0.75, 0.95), xycoords='axes fraction')
		if aux_axes is None:
			self.aux_axes = aux_axes
		with plt.ioff():
			self._setup()

	def display_text(self, message: str ):
		self.annotation.set_text( message )

	def update(self, val):
		raise NotImplementedError( "Abstract method" )

	def _setup(self):
		raise NotImplementedError( "Abstract method" )


class SignalPlotFigure(object):

	def __init__(self, plots: List[SignalPlot], **kwargs):
		plt.rc('xtick', labelsize=8)
		self.plots = plots
		self.nplots = len(plots)
		self.sparms: Dict[str,STParam] = {}
		self.callbacks = []
		with plt.ioff():
			self._setup( **kwargs )

	def count_sparms(self):
		sparms = set()
		for plot in self.plots:
			for sn in plot.sparms.keys():
				sparms.add(sn)
		return len(sparms)

	def link_plot(self, plot ):
		for sn, sp in plot.sparms.items():
			if sn in self.sparms:   plot.share_param(self.sparms[sn])
			else:                   self.sparms[sn] = sp
		self.callbacks.append(plot.update)

	@exception_handled
	def _setup(self, **kwargs):
		self.fig, self.axs = plt.subplots(self.nplots, 1, figsize=kwargs.get('figsize', (15, 9)))
		self.nparms = self.count_sparms()
		adjust_factor = max( self.nparms, 6 )
		plt.subplots_adjust( bottom=0.03*(adjust_factor+1) )
		axes = [self.axs] if self.nplots == 1 else self.axs
		self.aux_ax = plt.axes( (0.7,  0.0,  0.25,  0.03*adjust_factor ) )
		self.aux_ax.xaxis.set_major_locator(NullLocator())
		self.aux_ax.yaxis.set_major_locator(NullLocator())
		for plot, ax in zip(self.plots, axes):
			plot.initialize(ax,[self.aux_ax])
			self.link_plot(plot)
		self.callbacks.append(self.update)
		for aid, sp in enumerate(self.sparms.values()):
			sax = plt.axes((0.1, 0.03*aid, 0.55, 0.03))
			sp.widget(sax, self.callbacks)
		log.info(f"SignalPlotFigure._setup complete")

	@exception_handled
	def update(self, val: Any = None, **kwargs):
		self.fig.canvas.draw_idle()

	@exception_handled
	def show(self):
		plt.show()
		log.info(f"SignalPlotFigure.show complete")
