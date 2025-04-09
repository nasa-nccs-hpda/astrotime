import logging, numpy as np
import xarray as xa
from .param import STIntParam, STFloatParam
from astrotime.loaders.MIT import MITLoader
from torch import nn, optim, Tensor, FloatTensor
from .base import SignalPlot, bounds
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.backend_bases import KeyEvent, MouseEvent, MouseButton
from astrotime.util.logging import exception_handled
from astrotime.encoders.embedding import EmbeddingLayer
from astrotime.util.math import npnorm
from typing import List, Optional, Dict, Type, Union, Tuple, Any, Set
from astrotime.util.math import tnorm
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

class PeriodMarkers:

	def __init__(self, name: str, ax: Axes, **kwargs):
		self.name = name
		self.log = logging.getLogger("astrotime")
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
		self.log.info( f"\n PeriodMarkers({self.name}:{id(self):02X}).refresh( origin={self.origin:.2f}, period={self.period:.2f} ), #markers = {len(self.markers)} -- --- -- \n")

class MITDatasetPlot(SignalPlot):

	def __init__(self, name: str, data_loader: MITLoader, sector: int, **kwargs):
		SignalPlot.__init__(self, **kwargs)
		self.name = name
		self.sector: int = sector
		self.data_loader: MITLoader = data_loader
		self.TICS: List[str] = data_loader.TICS(sector)
		self.annotations: List[str] = tolower( kwargs.get('annotations',None) )
		self.ofac = kwargs.get('upsample_factor',1)
		self.plot: Line2D = None
		self.add_param( STIntParam('element', (0,len(self.TICS))  ) )
		self.period_markers: Dict[str,PeriodMarkers] = {}
		self.target_period = None
		self.ext_pm_ids: Set[str] = set()
		self.transax = None
		self.origin = None

	@exception_handled
	def update_period_markers(self, **marker_data ) -> str:
		pm_name=  marker_data['id']
		pm = self.period_markers.setdefault( pm_name, PeriodMarkers( pm_name, self.ax, color=marker_data['color'] ) )
		for pm in self.period_markers.values():
			period = marker_data.get('period',None) if (pm.name == pm_name) else None
			pm.update( self.origin, period )
		return pm_name

	def set_markers_visible(self, b: bool):
		for pm in self.period_markers.values():
			pm.set_visible(b)

	@exception_handled
	def process_external_event(self, event_data: Dict[str,Any] ) -> None:
		pass
	#	if event_data['type'] == 'period_grid':
	#		self.ext_pm_ids.add(  self.update_period_markers(**event_data)  )


	@exception_handled
	def button_press(self, event: MouseEvent) -> Any:
		if event.button == MouseButton.RIGHT:
			if "shift" in event.modifiers:
				if event.inaxes == self.ax:
					self.origin = event.xdata
					self.update_period_markers(id="dataset", period=self.target_period, axes=event.inaxes, color="green" )
				else:
					event_data: Dict = self._shared_params.get(id(event.inaxes))
					if event_data is not None:
						self.update_period_markers(id=event_data['id'], period=event_data['period'], axes=event_data['axes'], color="darkviolet" )

	#		if "ctrl" in event.modifiers:
	#			self.update_period_markers(id=list(self.ext_pm_ids)[0], origin=event.xdata, period=self.target_period)

	@exception_handled
	def button_release(self, event: MouseEvent) -> Any:
		pass

	@exception_handled
	def on_motion(self, event: MouseEvent) -> Any:
		pass

	def set_sector(self, sector: int ):
		self.sector = sector

	@exception_handled
	def _setup(self):
		xs, ys, target = self.get_element_data()
		self.target_period = target
		self.plot: Line2D = self.ax.plot(xs, ys, label='y', color='blue', marker=".", linewidth=1, markersize=2, alpha=0.5)[0]
		self.ax.title.set_text( f"{self.name}: TP={target:.3f} (F={1/target:.3f})")
		self.ax.title.set_fontsize(8)
		self.ax.title.set_fontweight('bold')
		self.ax.set_xlim(xs[0],xs[-1])
		self.ax.set_ylim( ys.min(), ys.max() )
		self.origin = xs[np.argmax(np.abs(ys))]
		self.update_period_markers(  id="dataset", origin=self.origin, period=self.target_period, axes=self.ax, color="green" )

	@exception_handled
	def get_element_data(self) -> Tuple[np.ndarray,np.ndarray,float]:
		element: xa.Dataset = self.data_loader.get_dataset_element(self.sector,self.TICS[self.element])
		t, y = element.data_vars['time'], element.data_vars['y']
		ydata: np.ndarray = y.values.squeeze()
		xdata: np.ndarray = t.values
		target: float = y.attrs['period']
		nan_mask = np.isnan(ydata)
		return xdata[~nan_mask], npnorm(ydata[~nan_mask],dim=0), target

	@exception_handled
	def update(self, val):
		xdata, ydata, target = self.get_element_data()
		self.plot.set_ydata(ydata)
		self.plot.set_xdata(xdata)
		self.target_period = target
		self.ax.set_xlim(xdata[0],xdata[-1])
		self.ax.set_ylim( ydata.min(), ydata.max() )
		self.origin = xdata[np.argmax(np.abs(ydata))]
		self.log.info( f" ---- DatasetPlot-> update({self.element}:{self.TICS[self.element]}):  xlim=({xdata[0]:.3f},{xdata[-1]:.3f}), ylim=({ydata[0]:.3f},{ydata[-1]:.3f}), xdata.shape={self.plot.get_xdata().shape} origin={self.origin} ---" )
		self.update_period_markers( id="dataset", period=self.target_period, axes=self.ax, color="green" )
		self.ax.title.set_text(f"{self.name}: TP={target:.3f} (F={1/target:.3f})")
		self.ax.figure.canvas.draw_idle()

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
		self.colors = ['darkviolet', 'darkorange', 'saddlebrown', 'darkturquoise', 'magenta' ]
		self.ofac = kwargs.get('upsample_factor',1)
		self.plots: Dict[str,Line2D] = {}
		self.target_marker: Line2D = None
		self.selection_marker: Line2D = None
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
		self.target_marker: Line2D = self.ax.axvline( freq, 0.0, 1.0, color='green', linestyle='-')
		self.selection_marker: Line2D = self.ax.axvline( 0, 0.0, 1.0, color=self.colors[0], linestyle='-', linewidth=2)
		self.ax.title.set_text(self.name)
		self.ax.title.set_fontsize(8)
		self.ax.title.set_fontweight('bold')
		self.ax.set_xlim( self.embedding_space[0], self.embedding_space[-1] )
		self.ax.set_ylim( 0.0, 1.0 )
		self.ax.set_xscale('log')
		self.ax.legend(loc="upper left")

	@exception_handled
	def button_press(self, event: MouseEvent) -> Any:
		if ("shift" in event.modifiers) and (event.button == MouseButton.RIGHT) and(event.inaxes == self.ax):
			freq, period = event.xdata, 1/event.xdata
			self._shared_params[id(self.ax)] = dict(id="transform", period=period, axes=self.ax)
			self.log.info(f"---- MITTransformPlot.button_press: selected freq={freq:.2f}, period={period:.2f} --- ")
			self.ax.title.set_text(f"{self.name}: TP={period:.3f} (F={freq:.3f})")
			self.selection_marker.set_xdata([freq, freq])

	@exception_handled
	def apply_transform( self, transform: EmbeddingLayer, series_data: xa.Dataset ) -> np.ndarray:
		slen = transform.cfg.series_length
		ts_tensors: Dict[str,Tensor] =  { k: FloatTensor(series_data.data_vars[k].values[:slen]).to(transform.device) for k in ['time','y'] }
		transformed: Tensor = transform.embed( ts_tensors['time'][None,:], tnorm(ts_tensors['y'][None,:],dim=1) )
		embedding: np.ndarray = transform.magnitude( transformed ).to('cpu').numpy()
		return znorm(embedding)

	@exception_handled
	def update(self, val):
		series_data: xa.Dataset = self.data_loader.get_dataset_element(self.sector, self.TICS[self.element])
		target_period: float = series_data.data_vars['y'].attrs['period']
		transform_peak = None
		for iplot, (tname, transform) in enumerate(self.transforms.items()):
			tdata: np.ndarray = self.apply_transform(transform,series_data)
			self.log.info(f"---- MITTransformPlot({iplot}) {tname}[{self.element})] update: tdata{tdata.shape}, mean={tdata.mean():.2f} --- ")
			self.plots[tname].set_ydata(tdata)
			if iplot == 0:
				freq_data = transform.embedding_space.cpu().numpy()
				transform_peak =  freq_data[ np.argmax(tdata) ]
		target_freq = 1.0/target_period
		self.target_marker.set_xdata([target_freq,target_freq])
		transform_period = 1.0/transform_peak
		self.selection_marker.set_xdata([transform_peak,transform_peak])
		self._shared_params[id(self.ax)] = dict(id="transform", period=transform_period, axes=self.ax)
		self.ax.title.set_text(f"{self.name}: TP={transform_period:.3f} (F={transform_peak:.3f})")
		self.ax.figure.canvas.draw_idle()

