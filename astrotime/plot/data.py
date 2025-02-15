import math, time, numpy as np
from .param import Number, STParam, STIntParam, STFloatParam, Parameterized
from .base import SignalPlot, bounds
from matplotlib.lines import Line2D
from astrotime.util.series import get_peak
from matplotlib.collections import PathCollection
import xarray as xa
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from astrotime.util.logging import lgm, exception_handled, log_timing


class SignalDataPlot(SignalPlot):

	def __init__(self, signal: InteractiveSignal, **kwargs):
		SignalPlot.__init__(self, **kwargs)
		self.signal: InteractiveSignal = signal
		self.plot: PathCollection | Line2D = None
		self.interp_factor = kwargs.get('interp_factor',1)
		self.inherit_params(signal)

	@exception_handled
	def _setup(self):
		x,y = self.signal.xydata()
		lgm().log( f"SignalDataPlot._setup: x{x.shape}, y{y.shape}")
		self.plot = self.ax.scatter(x, y, s=2, color='blue', marker=".")
		self.ax.title.set_text(self.signal.name)
		self.ax.title.set_fontsize(8)
		self.ax.title.set_fontweight('bold')
		self.display_text( f"Freq: {self.signal.freq:.3f}" )

		# self.ax.xaxis.set_major_locator(NullLocator())

	@exception_handled
	def update(self, val):
		t0 = time.time()
		x,y = self.signal.xydata()
		self.plot.set_offsets(np.c_[x, y])
		bnds = bounds(y)
		self.ax.set_ylim(*bnds)
		self.display_text(f"Freq: {self.signal.freq:.3f}")
		lgm().log(f" ** UPDATE {self.plot.__class__.__name__} with (x{x.shape},y{y.shape}), ybnds={bnds}, parms={[str(v) for v in self.signal.sparms.values()]} in time={time.time()-t0:.4f} sec")

class SignalTransformPlot(SignalPlot):

	def __init__(self, transform: SignalTransform, annotations: List[str] = None, **kwargs):
		SignalPlot.__init__(self, **kwargs)
		self.transform: SignalTransform = transform
		self.annotations: List[str] = [a.lower() for a in annotations] if (annotations is not None) else []
		self.ofac = kwargs.get('upsample_factor',1)
		self.plot: Line2D = None
		self.peak_plot: Line2D = None
		self.marker_lines = []
		self.inherit_params(self.transform)
		self.transax = None

	@exception_handled
	def _setup(self):
		sig_transform: xa.DataArray = self.transform.process_signal()
		if sig_transform is not None:
			t, f = sig_transform.coords['t'].values, sig_transform.coords['f'].values
			nt, tindex0 = t.size, t.size // 2
			lgm().log(f"SignalTransformPlot._setup: sig_transform{list(sig_transform.dims)}=> {list(sig_transform.shape)}, nt={nt}, tindex0={tindex0}")
			spectrum = sig_transform.isel(t=tindex0).values
			self.add_param(STIntParam('time_index', (0, nt), value=tindex0))
			self.plot, = self.ax.plot(f, spectrum, label=self.transform.name, color='blue', marker="o", linewidth=1, markersize=3 )
			self.peak_plot = None
			self.ax.title.set_text(self.transform.name)
			self.ax.title.set_fontsize(8)
			self.ax.title.set_fontweight('bold')
			self.ax.set_xlim(f[0], f[-1])

	def update_markers(self):
		lgm().log(f" ** update_markers:")
		for art in list(self.ax.lines):
			if art != self.plot:
				art.remove()
		for (marker, color, linestyle) in self.transform.get_markers():
			lgm().log(f"  -----> {marker} {color}")
			self.marker_lines.append( self.ax.axvline(x=marker, color=color, linestyle=linestyle) )

	@exception_handled
	def update_peak_interp(self, xp: np.ndarray, yp: np.ndarray):
		lgm().log(f"\n ** update_peak_interp: xp{list(xp.shape)} ({xp.mean():.3f}), yp{list(yp.shape)} ({yp.mean():.3f}) " )
		if self.peak_plot is not None:
			try: self.peak_plot.remove()
			except: pass
		self.peak_plot, = self.ax.plot(    xp,  yp, label=self.transform.name, color='green', marker=".", linewidth=1, markersize=2, alpha=0.5 )

	@exception_handled
	def update(self, val):
		t0 = time.time()
		sig_transform: xa.DataArray = self.transform.process_signal()
		if sig_transform is not None:
			spectrum: np.ndarray = sig_transform.isel(t=self.time_index).values
			f: np.ndarray = sig_transform.coords['f'].values
			lgm().log( f"SignalTransformPlot.update: f{list(f.shape)}, spectrum{list(spectrum.shape)}, process_signal time = {time.time()-t0:.4f} sec")
			self.plot.set_ydata(spectrum)
			self.plot.set_xdata(f)
			self.ax.set_ylim(*bounds(spectrum))
			self.ax.set_xlim(f[0], f[-1])
			self.update_markers()
			self.update_annotations(f, spectrum)

	def update_annotations(self, xvals: np.ndarray, yvals: np.ndarray):
		if ('l1' in self.annotations) or ('l2'  in self.annotations):
			t0 = time.time()
			xp, yp, mindx = get_peak( xvals, yvals, upsample=self.ofac )
			if self.ofac > 1: self.update_peak_interp( xp, yp )
			error = abs(xp[mindx] - self.transform.signal.freq)
			self.display_text( f"Error: {error:.5f}" )
			lgm().log( f" ** UPDATE ANNOTATIONS in time={time.time()-t0:.4f} sec")
