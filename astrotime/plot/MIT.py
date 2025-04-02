import logging, numpy as np
import xarray as xa
from .param import STIntParam
from astrotime.loaders.MIT import MITLoader
from torch import nn, optim, Tensor, FloatTensor
from .base import SignalPlot, bounds
from matplotlib.lines import Line2D
from matplotlib.backend_bases import KeyEvent, MouseEvent, MouseButton
from astrotime.util.logging import exception_handled
from astrotime.encoders.embedding import EmbeddingLayer, GPUEmbeddingLayer
from typing import List, Optional, Dict, Type, Union, Tuple, Any
from astrotime.util.math import tnorm, npnorm
log = logging.getLogger("astrotime")

def tolower(ls: Optional[List[str]]) -> List[str]:
	return [a.lower() for a in ls] if (ls is not None) else []

def znorm(ydata: np.ndarray) -> np.ndarray:
	y0,y1 = ydata.min(), ydata.max()
	return (ydata-y0)/(y1-y0)

def unorm(ydata: np.ndarray) -> np.ndarray:
	y0,y1 = ydata.min(), ydata.max()
	z = (y1-ydata)/(y1-y0)
	return 1 - 2*z

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
		self.period_markers= []
		self.markers_origin: float = 0.0
		self.target_period: float = 1.0
		self.transax = None

	def process_external_event(self, event: Dict[str,Any] ) -> None:
		# dict(type='period_grid', id='ww_analysis', center=event.xdata, period=event.ydata, color='yellow')
		self.log.info(f"MITDatasetPlot.process_external_event: event={event}")

	@exception_handled
	def button_press(self, event: MouseEvent) -> Any:
		if ("shift" in event.modifiers) and (event.button == MouseButton.RIGHT):
			self.markers_origin = event.xdata
			self.update_period_markers()
			self.fig.canvas.draw_idle()

	@exception_handled
	def button_release(self, event: MouseEvent) -> Any:
		pass

	@exception_handled
	def on_motion(self, event: MouseEvent) -> Any:
		pass

	def set_sector(self, sector: int ):
		self.sector = sector

	def update_period_markers(self, npm: int = 7 ):
		for pid in range(0,npm):
			tval = self.markers_origin + (pid-npm//2) * self.target_period
			if pid >= len(self.period_markers):  self.period_markers.append( self.ax.axvline( tval, -1, 1, color='green', linestyle='-', alpha=0.5) )
			else:                                self.period_markers[pid].set_xdata([tval,tval])

	@exception_handled
	def _setup(self):
		xs, ys, target = self.get_element_data()
		self.target_period = target
		self.plot: Line2D = self.ax.plot(xs, ys, label='y', color='blue', marker=".", linewidth=1, markersize=2, alpha=0.5)[0]
		self.ax.title.set_text(self.name)
		self.ax.title.set_fontsize(8)
		self.ax.title.set_fontweight('bold')
		self.ax.set_xlim(xs[0],xs[-1])
		self.ax.set_ylim(-1,1)
		self.markers_origin: float = xs[np.argmax(np.abs(ys))]
		self.update_period_markers()

	def get_element_data(self) -> Tuple[np.ndarray,np.ndarray,float]:
		element: xa.Dataset = self.data_loader.get_dataset_element(self.sector,self.TICS[self.element])
		t, y = element.data_vars['time'], element.data_vars['y']
		ydata: np.ndarray = y.values
		xdata: np.ndarray = t.values
		target: float = y.attrs['period']
		return xdata, znorm(ydata.squeeze()), target

	@exception_handled
	def update(self, val):
		xdata, ydata, target = self.get_element_data()
		self.plot.set_ydata(ydata)
		self.plot.set_xdata(xdata)
		self.ax.set_xlim(xdata[0],xdata[-1])
		self.log.info( f"Plot update: xlim={self.ax.get_xlim()} ({xdata[0]:.3f},{xdata[-1]:.3f}), xdata.shape={self.plot.get_xdata().shape} " )
		self.markers_origin: float = xdata[np.argmax(np.abs(ydata))]
		self.target_period = target
		self.update_period_markers()


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
		period: float = series_data.data_vars['y'].attrs['period']
		freq = 1.0 / period
		for iplot, (tname, transform) in enumerate(self.transforms.items()):
			tdata: np.ndarray = self.apply_transform(transform,series_data)
			self.plots[tname] = self.ax.plot(self.embedding_space, tdata.squeeze(), label=tname, color=self.colors[iplot], marker=".", linewidth=1, markersize=2, alpha=0.5)[0]
		self.target_marker: Line2D = self.ax.axvline( freq, 0.0, 1.0, color='grey', linestyle='-')
		self.ax.title.set_text(self.name)
		self.ax.title.set_fontsize(8)
		self.ax.title.set_fontweight('bold')
		self.ax.set_xlim( self.embedding_space[0], self.embedding_space[-1] )
		self.ax.set_ylim( 0.0, 1.0 )
		self.ax.set_xscale('log')
		self.ax.legend(loc="upper left")

	@exception_handled
	def button_press(self, event: MouseEvent) -> Any:
		if ("shift" in event.modifiers) and (event.button == MouseButton.RIGHT):
			event_data = dict(type='period_grid', id='ww_analysis', center=event.xdata, period=event.ydata, color='yellow')
			for listener in self.self.listeners:
				listener(event_data)

	@exception_handled
	def apply_transform( self, transform: EmbeddingLayer, series_data: xa.Dataset) -> np.ndarray:
		if type(EmbeddingLayer) == GPUEmbeddingLayer:
			ts_tensors: Dict[str,Tensor] =  { k: FloatTensor(series_data.data_vars[k].values).to(transform.device) for k in ['time','y'] }
			transformed: Tensor = transform.embed( ts_tensors['time'][None,:], tnorm(ts_tensors['y'][None,:],dim=1) )
			embedding: np.ndarray = transform.magnitude( transformed ).to('cpu').numpy()
		else:
			signals: Dict[str,np.ndarray] =  { k: series_data.data_vars[k].values for k in ['time','y'] }
			transformed: Tensor = transform.embed( signals['time'][None,:], npnorm(signals['y'][None,:],dim=1) )
			embedding: np.ndarray = transform.magnitude( transformed )
		return znorm(embedding)

	@exception_handled
	def update(self, val):
		series_data: xa.Dataset = self.data_loader.get_dataset_element(self.sector, self.TICS[self.element])
		period: float = series_data.data_vars['y'].attrs['period']
		for iplot, (tname, transform) in enumerate(self.transforms.items()):
			tdata: np.ndarray = self.apply_transform(transform,series_data)
			self.plots[tname].set_ydata(tdata)
		freq = 1.0/period
		self.target_marker.set_xdata([freq,freq])

