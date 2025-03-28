import logging, numpy as np
import xarray as xa
from .param import STIntParam
from astrotime.loaders.MIT import MITLoader
from torch import nn, optim, Tensor, FloatTensor
from .base import SignalPlot, bounds
from matplotlib.lines import Line2D
from astrotime.util.logging import exception_handled
from astrotime.encoders.embedding import EmbeddingLayer
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.util.math import tnorm
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

	def set_sector(self, sector: int ):
		self.sector = sector

	@exception_handled
	def _setup(self):
		xdata, ydata, target = self.get_element_data()
		self.plot: Line2D = self.ax.plot(xdata.squeeze(), ydata.squeeze(), label='y', color='blue', marker=".", linewidth=1, markersize=2, alpha=0.5)[0]
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
		self.plot.set_xdata(xdata)
		self.ax.set_xlim(xdata[0],xdata[-1])
		self.ax.set_ylim(ydata.min(),ydata.max())
		self.log.info( f"Plot update: xlim={self.ax.get_xlim()} ({xdata[0]:.3f},{xdata[-1]:.3f}), xdata.shape={self.plot.get_xdata().shape} " )


class MITTransformPlot(SignalPlot):

	def __init__(self, name: str, data_loader: MITLoader, transforms: Dict[str,EmbeddingLayer], embedding_space: np.ndarray, sector: int, **kwargs):
		SignalPlot.__init__(self, **kwargs)
		self.name = name
		self.sector: int = sector
		self.transforms: Dict[str,EmbeddingLayer] = transforms
		self.data_loader: MITLoader = data_loader
		self.embedding_space: np.ndarray = embedding_space
		self.TICS: List[str] = data_loader.TICS(sector)
		self.annotations: List[str] = tolower( kwargs.get('annotations',None) )
		self.colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'sienna', 'yellow']
		self.ofac = kwargs.get('upsample_factor',1)
		self.plots: Dict[str,Line2D] = {}
		self.target_marker: Line2D = None
		self.add_param( STIntParam('element', (0,len(self.TICS))  ) )
		self.transax = None

	def set_sector(self, sector: int ):
		self.sector = sector

	@exception_handled
	def _setup(self):
		series_data: xa.Dataset = self.data_loader.get_dataset_element(self.sector, self.TICS[self.element])
		target: float = series_data.data_vars['y'].attrs['period']
		for iplot, (tname, transform) in enumerate(self.transforms.items()):
			tdata: np.ndarray = self.apply_transform(transform,series_data)
			self.plots[tname] = self.ax.plot(self.embedding_space, tdata.squeeze(), label=tdata, color=self.colors[iplot], marker=".", linewidth=1, markersize=2, alpha=0.5)[0]
		self.target_marker: Line2D = self.ax.axvline(x=1.0/target, color='grey', linestyle='-')
		self.ax.title.set_text(self.name)
		self.ax.title.set_fontsize(8)
		self.ax.title.set_fontweight('bold')
		self.ax.set_xlim( self.embedding_space[0], self.embedding_space[-1] )
		self.ax.set_xscale('log')

	@exception_handled
	def apply_transform( self, transform: EmbeddingLayer, series_data: xa.Dataset, feature: int = -1 ) -> np.ndarray:
		ts_tensors: Dict[str,Tensor] =  { k: FloatTensor(series_data.data_vars[k].values).to(transform.device) for k in ['time','y'] }
		transformed: Tensor = transform.embed( ts_tensors['time'][None,:], tnorm(ts_tensors['y'][None,:],dim=1) )
		embedding = transformed[:,feature] if (feature >= 0) else (transformed*transformed).mean(dim=1).sqrt()
		ydata: np.ndarray = embedding.to('cpu').numpy()
		return ydata

	@exception_handled
	def update(self, val):
		series_data: xa.Dataset = self.data_loader.get_dataset_element(self.sector, self.TICS[self.element])
		target: float = series_data.data_vars['y'].attrs['period']
		ymin, ymax = 10000, -10000
		for iplot, (tname, transform) in enumerate(self.transforms.items()):
			tdata: np.ndarray = self.apply_transform(transform,series_data)
			self.plots[tname].set_ydata(tdata)
			tmin, tmax= tdata.min(), tdata.max()
			if tmin < ymin: ymin = tmin
			if tmax > ymax: ymax = tmax
		self.ax.set_ylim(ymin,ymax)
		self.target_marker.set_xdata([target,target])
		self.target_marker.set_ydata([ymin,ymax])

