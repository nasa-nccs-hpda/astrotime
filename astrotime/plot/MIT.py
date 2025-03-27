import logging, numpy as np
import xarray as xa
from .param import STIntParam
from astrotime.loaders.MIT import MITLoader
from .base import SignalPlot, bounds
from matplotlib.lines import Line2D
from astrotime.util.logging import exception_handled
from typing import List, Optional, Dict, Type, Union, Tuple
log = logging.getLogger("astrotime")

def tolower(ls: Optional[List[str]]) -> List[str]:
	return [a.lower() for a in ls] if (ls is not None) else []


class MITDatasetPlot(SignalPlot):

	def __init__(self, name: str, data_loader: MITLoader, sector: int, **kwargs):
		SignalPlot.__init__(self, **kwargs)
		self.name = name
		self.sector: int = sector
		self.data_loader: MITLoader = data_loader
		self.TICS: List[str] = data_loader.TICS(sector)
		self.annotations: List[str] = tolower( kwargs.get('annotations',None) )
		self.colors = ['blue', 'green'] + [ 'yellow' ] * 16
		self.ofac = kwargs.get('upsample_factor',1)
		self.plot: Line2D = None
		self.add_param( STIntParam('element', (0,len(self.TICS))  ) )
		self.transax = None

	def set_dset_index(self, dset_index: int ):
		self.dset_idx = dset_index

	@exception_handled
	def _setup(self):
		xdata, ydata, target = self.get_element_data()
		self.plot: Line2D = self.ax.plot(xdata, ydata, label='y', color='blue', marker=".", linewidth=1, markersize=2, alpha=0.8)[0]
		self.ax.title.set_text(self.name)
		self.ax.title.set_fontsize(8)
		self.ax.title.set_fontweight('bold')
		self.ax.set_xlim(xdata[0],xdata[-1])

	def get_element_data(self) -> Tuple[np.ndarray,np.ndarray,float]:
		element: xa.Dataset = self.data_loader.get_dataset_element(self.sector,self.TICS[self.element])
		t, y = element.data_vars['time'], element.data_vars['y']
		ydata: np.ndarray = y.values
		xdata: np.ndarray = t.values
		target: float = y.attrs['period']
		return xdata, ydata, target

	# @exception_handled
	# def update_peak_interp(self, xp: np.ndarray, yp: np.ndarray):
	# 	log.info(f"\n ** update_peak_interp: xp{list(xp.shape)} ({xp.mean():.3f}), yp{list(yp.shape)} ({yp.mean():.3f}) " )
	# 	if self.peak_plot is not None:
	# 		try: self.peak_plot.remove()
	# 		except: pass
	# 	self.peak_plot, = self.ax.plot(    xp,  yp, label=self.transform.name, color='green', marker=".", linewidth=1, markersize=2, alpha=0.5 )

	@exception_handled
	def update(self, val):
		xdata, ydata, target = self.get_element_data()
		self.plot.set_ydata(ydata)
	#	self.lines['target'].remove()
	#	self.lines['target'] = self.ax.axvline(x=1.0/target, color='r', linestyle='-')
		self.ax.set_ylim(*bounds(ydata))
		self.plot.set_xdata(xdata)
		self.ax.set_xlim(xdata[0],xdata[-1])
		self.log.info( f"Plot update: xlim={self.ax.get_xlim()} ({xdata[0]:.3f},{xdata[-1]:.3f}), xdata.shape={self.plot.get_xdata().shape} " )
