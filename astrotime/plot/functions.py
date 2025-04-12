import logging, numpy as np
from matplotlib import ticker
from .base import SignalPlot, bounds
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent, MouseButton
from astrotime.util.logging import exception_handled
from typing import List, Optional, Dict, Type, Union, Tuple, Any, Set, Callable

class FunctionPlot(SignalPlot):

	def __init__(self, name: str, function: Callable, **kwargs):
		SignalPlot.__init__(self, **kwargs)
		self.name = name
		self.function = function
		self.xs = None
		self.npts = kwargs.get('npts', 1000)
		self.plot: Line2D = None
		self.domain = kwargs.get('domain', (-1,1) )

	@exception_handled
	def button_press(self, event: MouseEvent) -> Any:
		if event.button == MouseButton.RIGHT:
			if ("shift" in event.modifiers) and (event.inaxes == self.ax):
				self.update()

	@exception_handled
	def key_press(self, event: KeyEvent) -> Any:
		if event.key in ['ctrl+f','alt+ƒ']:
			self.update()

	@exception_handled
	def button_release(self, event: MouseEvent) -> Any:
		pass

	@exception_handled
	def on_motion(self, event: MouseEvent) -> Any:
		pass

	@exception_handled
	def _setup(self ):
		self.xs = np.linspace( self.domain[0], self.domain[1], self.npts)
		ys = self.function(self.xs)
		self.plot = self.ax.plot(self.xs, ys, label='y', color='blue', marker=".", linewidth=1, markersize=2, alpha=0.5)[0]
		self.ax.title.set_text(f"{self.name}")
		self.ax.title.set_fontsize(8)
		self.ax.title.set_fontweight('bold')
		self.ax.set_xlim(self.xs.min(),self.xs.max())
		self.ax.set_ylim(ys.min(),ys.max())

	@exception_handled
	def get_element_data(self) -> Tuple[np.ndarray,np.ndarray]:
		ys = self.function(self.xs)
		return self.xs, ys

	@exception_handled
	def update(self, val=0, **kwargs ):
		xdata, ydata = self.get_element_data()
		self.plot.set_ydata(ydata)
		self.plot.set_xdata(xdata)
		title = f"{self.name}"
		self.ax.title.set_text(title)
		self.ax.set_xlim(xdata.min(),xdata.max())
		self.ax.set_ylim(ydata.min(),ydata.max())
		self.ax.figure.canvas.draw_idle()

	@exception_handled
	def show(self):
		plt.show()

