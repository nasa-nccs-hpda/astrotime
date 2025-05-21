import logging, numpy as np
import xarray as xa
from .param import STIntParam
from astrotime.loaders.base import IterativeDataLoader
from .base import SignalPlot, bounds
from matplotlib.lines import Line2D
from astrotime.util.logging import exception_handled
from typing import List, Optional, Dict, Type, Union, Tuple
log = logging.getLogger()

def tolower(ls: Optional[List[str]]) -> List[str]:
	return [a.lower() for a in ls] if (ls is not None) else []

class SignalDataPlot(SignalPlot):

	def __init__(self, name: str, x: np.ndarray, y: np.ndarray, target: np.ndarray = None,  **kwargs):
		SignalPlot.__init__(self, **kwargs)
		self.name = name
		self.x = x
		self.y = y
		self.target = target
		self.annotations: List[str] = tolower( kwargs.get('annotations',None) )
		self.colors = ['blue', 'green'] + [ 'yellow' ] * 16
		self.ofac = kwargs.get('upsample_factor',1)
		self.lines: Dict[str,Line2D] = {}
		self.add_param( STIntParam('element', (0,self.y.shape[0])  ) )
		self.transax = None

	@exception_handled
	def _setup(self):
		ydata: np.ndarray = self.y[self.element]
		xdata: np.ndarray = self.x if self.x.ndim == 1 else self.x[self.element]
		if ydata.ndim == 1:
			self.lines['y'], = self.ax.plot(xdata, ydata, label='y', color='blue', marker="o", linewidth=1, markersize=3)
		else:
			for ic in range(0,ydata.ndim):
				self.lines[f'y{ic}'], = self.ax.plot(xdata, ydata[:,ic], label=f'y{ic}', color=self.colors[ic], marker=".", linewidth=1, markersize=1)
		if self.target is not None:
			self.lines['target'] = self.ax.axvline(x=1.0/self.target[self.element], color='r', linestyle='-')
		self.ax.title.set_text(self.name)
		self.ax.title.set_fontsize(8)
		self.ax.title.set_fontweight('bold')
		self.ax.set_xlim(xdata[0],xdata[-1])

	@exception_handled
	def update_peak_interp(self, xp: np.ndarray, yp: np.ndarray):
		log.info(f"\n ** update_peak_interp: xp{list(xp.shape)} ({xp.mean():.3f}), yp{list(yp.shape)} ({yp.mean():.3f}) " )
		if self.peak_plot is not None:
			try: self.peak_plot.remove()
			except: pass
		self.peak_plot, = self.ax.plot(    xp,  yp, label=self.transform.name, color='green', marker=".", linewidth=1, markersize=2, alpha=0.5 )

	@exception_handled
	def update(self, val):
		ydata: np.ndarray = self.y[self.element]
		if ydata.ndim == 1:
			self.lines['y'].set_ydata(ydata)
		else:
			for ic in range(0,ydata.ndim):
				self.lines[f'y{ic}'].set_ydata(ydata[:,ic])
		if self.target is not None:
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
	# 		log.info( f" ** UPDATE ANNOTATIONS in time={time.time()-t0:.4f} sec")


class SignalDatasetPlot(SignalPlot):

	def __init__(self, name: str, data_loader: IterativeDataLoader, **kwargs):
		SignalPlot.__init__(self, **kwargs)
		self.name = name
		self.data_loader: IterativeDataLoader = data_loader
		self.dset_idx = data_loader.dset_idx
		self.annotations: List[str] = tolower( kwargs.get('annotations',None) )
		self.colors = ['blue', 'green'] + [ 'yellow' ] * 16
		self.ofac = kwargs.get('upsample_factor',1)
		self.lines: Dict[str,Line2D] = {}
		self.add_param( STIntParam('element', (0,data_loader.nelements)  ) )
		self.transax = None

	def set_dset_index(self, dset_index: int ):
		self.dset_idx = dset_index

	@exception_handled
	def _setup(self):
		xdata, ydata, target = self.get_element_data()
		self.lines['y'], = self.ax.plot(xdata, ydata, label='y', color='blue', marker=".", markersize=1)
	#	self.lines['target'] = self.ax.axvline(x=1.0/target, color='r', linestyle='-')
		self.ax.title.set_text(self.name)
		self.ax.title.set_fontsize(8)
		self.ax.title.set_fontweight('bold')
		self.ax.set_xlim(xdata[0],xdata[-1])

	def get_element_data(self) -> Tuple[np.ndarray,np.ndarray,float]:
		element: Dict[str,np.ndarray|float] = self.data_loader.get_element(self.dset_idx,self.element)
		print( element.keys())
		ydata: np.ndarray = element['y']
		xdata: np.ndarray = element['t']
		target: float = element['p'] if ('p' in element) else element['period']
		return xdata, ydata, target

	@exception_handled
	def update_peak_interp(self, xp: np.ndarray, yp: np.ndarray):
		log.info(f"\n ** update_peak_interp: xp{list(xp.shape)} ({xp.mean():.3f}), yp{list(yp.shape)} ({yp.mean():.3f}) " )
		if self.peak_plot is not None:
			try: self.peak_plot.remove()
			except: pass
		self.peak_plot, = self.ax.plot(    xp,  yp, label=self.transform.name, color='green', marker=".", linewidth=1, markersize=2, alpha=0.5 )

	@exception_handled
	def update(self, val):
		xdata, ydata, target = self.get_element_data()
		self.lines['y'].set_ydata(ydata)
	#	self.lines['target'].remove()
	#	self.lines['target'] = self.ax.axvline(x=1.0/target, color='r', linestyle='-')
		self.ax.set_ylim(*bounds(ydata))
		self.plot.set_xdata(xdata)
		self.ax.set_xlim(*bounds(xdata))
