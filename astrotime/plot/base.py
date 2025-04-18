import math, time, numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from typing import Any, Dict, List, Tuple, Type, Optional, Callable
from astrotime.util.logging import exception_handled, log_timing
from .param import Number, Parameter, STParam, STFloatParam, STFloatValuesParam, Parameterized
import logging
log = logging.getLogger()

def bounds( y: np.ndarray ) -> Tuple[float,float]:
	ymin, ymax = y.min(), y.max()
	buff = 0.05 * (ymax - ymin)
	return ymin-buff, ymax+buff

class SignalPlot(Parameterized):
	_instances = []

	def __init__(self, **kwargs):
		Parameterized.__init__(self)
		self.ax: Axes = None
		self.log = logging.getLogger()
		self.annotation: Annotation = None
		SignalPlot._instances.append( self)

	@property
	def fig(self):
		return self.ax.get_figure()

	@exception_handled
	def initialize(self, ax: Axes):
		self.ax = ax
		self.annotation: Annotation = self.ax.annotate("", (0.75, 0.95), xycoords='axes fraction')
		self._setup()
		self.fig.canvas.mpl_connect('button_press_event', self.button_press)
		self.fig.canvas.mpl_connect('button_release_event', self.button_release)
		self.fig.canvas.mpl_connect('key_press_event', self.key_press)
		self.fig.canvas.mpl_connect('key_release_event', self.key_release)
		self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

	def key_press(self, event: KeyEvent) -> Any:
		pass

	def key_release(self, event: KeyEvent) -> Any:
		pass

	def button_press(self, event: MouseEvent) -> Any:
		pass

	def button_release(self, event: MouseEvent) -> Any:
		pass

	def on_motion(self, event: MouseEvent) -> Any:
		pass

	def display_text(self, message: str ):
		self.annotation.set_text( message )

	def update(self, val):
		raise NotImplementedError( "Abstract method" )

	@classmethod
	def process_event(cls, **event_data ):
		for splot in cls._instances:
			splot.process_ext_event(**event_data)

	def process_ext_event(self, **event_data):
		pass

	def _setup(self):
		raise NotImplementedError( "Abstract method" )


class SignalPlotFigure(object):

	def __init__(self, plots: List[SignalPlot], **kwargs):
		plt.rc('xtick', labelsize=8)
		self.log = logging.getLogger()
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

	def key_press(self, event: KeyEvent) -> Any:
		ke: KeyEvent = event
		for sn, sp in self.sparms.items():
			sp.process_key_press(ke.key)

	def key_release(self, event: KeyEvent) -> Any:
		pass

	def button_press(self, event: MouseEvent) -> Any:
		pass
		#self.log.info(f" ------- SignalPlotFigure: button-press event: {event} ---------" )
		# for plot in self.plots:
		# 	plot.process_button_press( event )
		# self.log.info(f" ------- SignalPlotFigure: button-press complete ---------")

	def button_release(self, event: MouseEvent) -> Any:
		pass

	def on_motion(self, event: MouseEvent) -> Any:
		pass

	@exception_handled
	def _setup(self, **kwargs):
		self.fig, self.axs = plt.subplots(self.nplots, 1, figsize=kwargs.get('figsize', (15, 9)))
		self.fig.canvas.mpl_connect('button_press_event', self.button_press)
		self.fig.canvas.mpl_connect('button_release_event', self.button_release)
		self.fig.canvas.mpl_connect('key_press_event', self.key_press)
		self.fig.canvas.mpl_connect('key_release_event', self.key_release)
		self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
		self.nparms = self.count_sparms()
		adjust_factor = max( self.nparms, 6 )
		plt.subplots_adjust( bottom=0.03*(adjust_factor+1) )
		axes = [self.axs] if self.nplots == 1 else self.axs
		for plot, ax in zip(self.plots, axes):
			plot.initialize(ax)
			self.link_plot(plot)
		self.callbacks.append(self.update)
		for aid, sp in enumerate(self.sparms.values()):
			root_size = (0.1, 0.03*aid, 0.55, 0.03)
			sax = plt.axes(root_size)
			sp.widget(sax, self.callbacks )
		log.info(f"SignalPlotFigure._setup complete")

	@exception_handled
	def update(self, val: Any = None, **kwargs):
		self.log.info(f" ------- SignalPlotFigure: widget-generated update({val}) ---------")
		self.fig.canvas.draw_idle()

	@exception_handled
	def show(self):
		plt.show()
		log.info(f"SignalPlotFigure.show complete")
#
# class SignalPlotFigure1(object):
#
# 	def __init__(self, plots: List[SignalPlot], **kwargs):
# 		plt.rc('xtick', labelsize=8)
# 		self.log = logging.getLogger()
# 		self.plots = plots
# 		self.nplots = len(plots)
# 		self.sparms: Dict[str,STParam] = {}
# 		self.callbacks = []
# 		with plt.ioff():
# 			self._setup( **kwargs )
#
# 	def count_sparms(self):
# 		sparms = set()
# 		for plot in self.plots:
# 			for sn in plot.sparms.keys():
# 				sparms.add(sn)
# 		return len(sparms)
#
# 	def link_plot(self, plot ):
# 		for sn, sp in plot.sparms.items():
# 			if sn in self.sparms:   plot.share_param(self.sparms[sn])
# 			else:                   self.sparms[sn] = sp
# 		self.callbacks.append(plot.update)
#
# 	def key_press(self, event: KeyEvent) -> Any:
# 		pass
#
# 	def key_release(self, event: KeyEvent) -> Any:
# 		pass
#
# 	def button_press(self, event: MouseEvent) -> Any:
# 		pass
#
# 	def button_release(self, event: MouseEvent) -> Any:
# 		pass
#
# 	def on_motion(self, event: MouseEvent) -> Any:
# 		pass
#
# 	@exception_handled
# 	def _setup(self, **kwargs):
# 		self.fig, self.axs = plt.subplots(self.nplots, 1, figsize=kwargs.get('figsize', (15, 9)))
# 		self.fig.canvas.mpl_connect('button_press_event', self.button_press)
# 		self.fig.canvas.mpl_connect('button_release_event', self.button_release)
# 		self.fig.canvas.mpl_connect('key_press_event', self.key_press)
# 		self.fig.canvas.mpl_connect('key_release_event', self.key_release)
# 		self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
# 		self.nparms = self.count_sparms()
# 		adjust_factor = max( self.nparms, 6 )
# 		plt.subplots_adjust( bottom=0.03*(adjust_factor+1) )
# 		axes = [self.axs] if self.nplots == 1 else self.axs
# 		for plot, ax in zip(self.plots, axes):
# 			plot.initialize(ax)
# 			self.link_plot(plot)
# 		self.callbacks.append(self.update)
# 		for aid, sp in enumerate(self.sparms.values()):
# 			root_size = (0.1, 0.03*aid, 0.55, 0.03)
# 			sax = plt.axes(root_size)
# 			aux_axes = self.get_aux_axes(list(root_size), sp.aux_sizes )
# 			sp.widget(sax, self.callbacks, aux_axes)
# 		log.info(f"SignalPlotFigure._setup complete")
#
# 	def get_aux_axes(self, root_size: List[float], aux_sizes: List[float]) -> List[ Axes]:
# 		aux_axes: List[ Axes] = []
# 		p0 = root_size[0] + root_size[2]
# 		for asize in aux_sizes:
# 			adims = ( p0, root_size[1], asize, root_size[3] )
# 			aux_axes.append( plt.axes( adims ) )
# 			p0 = p0 + asize
# 		return aux_axes
#
#
# 	@exception_handled
# 	def update(self, val: Any = None, **kwargs):
# 		self.log.info(f" ------- SignalPlotFigure: widget-generated update({val}) ---------")
# 		self.fig.canvas.draw_idle()
#
# 	@exception_handled
# 	def show(self):
# 		plt.show()
#		log.info(f"SignalPlotFigure.show complete")