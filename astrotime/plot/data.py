import math, time, numpy as np
from .param import Number, STParam, STIntParam, STFloatParam, Parameterized
from .base import SignalPlot, bounds
from matplotlib.lines import Line2D
import xarray as xa
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from astrotime.util.logging import lgm, exception_handled, log_timing

class SignalTransformPlot(SignalPlot):

	def __init__(self, name: str, x: np.ndarray, y: np.ndarray, target: np.ndarray, annotations: List[str] = None, **kwargs):
		SignalPlot.__init__(self, **kwargs)
		self.name = name
		self.x = x
		self.y = y
		self.target = target
		self.annotations: List[str] = [a.lower() for a in annotations] if (annotations is not None) else []
		self.ofac = kwargs.get('upsample_factor',1)
		self.lines: Dict[str,Line2D] = {}
		self.add_param( STIntParam('element', (0,self.y.shape[0])  ) )
		self.transax = None

	@exception_handled
	def _setup(self):
		ydata: np.ndarray = self.y[self.element] if self.y.ndim == 3 else self.y[self.element]
		xdata: np.ndarray = self.x if self.x.ndim == 1 else self.x[self.element]
		self.lines['y0'], = self.ax.plot(xdata, ydata[:,0], label='y0', color='blue', marker="o", linewidth=1, markersize=3 )
		self.lines['y1'], = self.ax.plot(xdata, ydata[:,1], label='y1', color='green', marker=".", linewidth=1, markersize=1)
		self.lines['target'] = self.ax.axvline(x=1.0/self.target[self.element], color='r', linestyle='-')
		self.ax.title.set_text(self.name)
		self.ax.title.set_fontsize(8)
		self.ax.title.set_fontweight('bold')
		self.ax.set_xlim(xdata[0],xdata[-1])

	@exception_handled
	def update_peak_interp(self, xp: np.ndarray, yp: np.ndarray):
		lgm().log(f"\n ** update_peak_interp: xp{list(xp.shape)} ({xp.mean():.3f}), yp{list(yp.shape)} ({yp.mean():.3f}) " )
		if self.peak_plot is not None:
			try: self.peak_plot.remove()
			except: pass
		self.peak_plot, = self.ax.plot(    xp,  yp, label=self.transform.name, color='green', marker=".", linewidth=1, markersize=2, alpha=0.5 )

	@exception_handled
	def update(self, val):
		ydata: np.ndarray = self.y[self.element] if self.y.ndim == 3 else self.y[self.element]
		self.lines['y0'].set_ydata(ydata[:,0])
		self.lines['y1'].set_ydata(ydata[:,1])
		self.lines['target'].remove()
		self.lines['target'] = self.ax.axvline(x=1.0/self.target[self.element], color='r', linestyle='-')
		self.ax.set_ylim(*bounds(ydata))
		if self.x.ndim == 2:
			xdata: np.ndarray = self.x[self.element]
			self.plot.set_xdata(xdata)
			self.ax.set_xlim(*bounds(xdata))

	#	self.update_markers()
	#	self.update_annotations(self.x, pdata)

	# def update_annotations(self, xvals: np.ndarray, yvals: np.ndarray):
	# 	if ('l1' in self.annotations) or ('l2'  in self.annotations):
	# 		t0 = time.time()
	# 		xp, yp, mindx = get_peak( xvals, yvals, upsample=self.ofac )
	# 		if self.ofac > 1: self.update_peak_interp( xp, yp )
	# 		error = abs(xp[mindx] - self.transform.signal.freq)
	# 		self.display_text( f"Error: {error:.5f}" )
	# 		lgm().log( f" ** UPDATE ANNOTATIONS in time={time.time()-t0:.4f} sec")
