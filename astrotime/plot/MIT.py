import logging, numpy as np
import xarray as xa
from .param import STIntParam
from matplotlib import ticker
from torch import nn, optim, Tensor, FloatTensor
from .base import SignalPlot, bounds
from astrotime.loaders.MIT import MITLoader
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.backend_bases import KeyEvent, MouseEvent, MouseButton
from astrotime.loaders.base import IterativeDataLoader
from astrotime.util.logging import exception_handled
from astrotime.encoders.embedding import Transform, EmbeddingLayer
from typing import List, Optional, Dict, Type, Union, Tuple, Any, Set
from astrotime.util.math import tnorm
log = logging.getLogger()

def tolower(ls: Optional[List[str]]) -> List[str]:
	return [a.lower() for a in ls] if (ls is not None) else []

def l2norm(ydata: np.ndarray) -> np.ndarray:
	m,s = ydata.mean(), ydata.std()
	return (ydata-m)/s

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
		self.log = logging.getLogger()
		self.ax: Axes = ax
		self.origin: float = None
		self.period: float = None
		self.markers: List[Line2D] = []
		self.yrange = kwargs.get('yrange', (-1,1) )
		self.npm: int = kwargs.get('npm', 7 )
		self.color: str = kwargs.get('color', 'green' )
		self.alpha: float = kwargs.get('alpha', 0.5 )
		self.linestyle: str = kwargs.get('linestyle', '-')

	def update(self, origin: float, period: float = None ):
		self.origin = origin
		if period is not None:
			self.period = period
		self.refresh()

	@property
	def fig(self):
		return self.ax.get_figure()

	def refresh(self):
		self.log.info( f" PeriodMarkers({self.name}:{id(self):02X}).refresh( origin={self.origin:.2f}, period={self.period:.2f} ) -- --- -- ")
		for pid in range(0,self.npm):
			tval = self.origin + (pid-self.npm//2) * self.period
			if pid >= len(self.markers):  self.markers.append( self.ax.axvline( tval, self.yrange[0], self.yrange[1], color=self.color, linestyle=self.linestyle, alpha=self.alpha) )
			else:                         self.markers[pid].set_xdata([tval,tval])

class MITDatasetPlot(SignalPlot):

	def __init__(self, name: str, data_loader: IterativeDataLoader, sector: int, **kwargs):
		SignalPlot.__init__(self, **kwargs)
		self.name = name
		self.sector: int = sector
		self.data_loader: MITLoader = data_loader
		self.refresh = kwargs.get('refresh', False)
		self.TICS: List[str] = data_loader.TICS(sector)
		self.annotations: List[str] = tolower( kwargs.get('annotations',None) )
		self.colors = ['blue', 'green'] + [ 'yellow' ] * 16
		self.ofac = kwargs.get('upsample_factor',1)
		self.plot: Line2D = None
		self.add_param( STIntParam('element', (0,len(self.TICS))))
		self.period_markers: Dict[str,PeriodMarkers] = {}
		self.ext_pm_ids: Set[str] = set()
		self.transax = None
		self.origin = None
		self.period = None
		self.fold_period: float = None

	@exception_handled
	def update_period_marker(self) -> str:
		pm_name= str(id(self.ax))
		pm = self.period_markers.setdefault( pm_name, PeriodMarkers( pm_name, self.ax ) )
		pm.update( self.origin, self.period )
		self.log.info( f" ---- DatasetPlot-> update_period_marker origin={self.origin:.3f} period={self.period:.3f} ---")
		self.update_pm_origins()
		return pm_name

	@exception_handled
	def update_pm_origins(self) :
		for pm in self.period_markers.values():
			pm.update( self.origin )

	@exception_handled
	def button_press(self, event: MouseEvent) -> Any:
		if event.button == MouseButton.RIGHT:
			if ("shift" in event.modifiers) and (event.inaxes == self.ax):
				self.origin = event.xdata
				self.update_pm_origins()

	@exception_handled
	def key_press(self, event: KeyEvent) -> Any:
		if event.key in ['ctrl+f','alt+ƒ']:
			if self.fold_period is None:    self.fold_period = self.period if (event.key == 'ctrl+f') else self.get_ext_period()
			else :                          self.fold_period = None
			self.log.info(f"                 MITDatasetPlot-> key_press({event.key}), fold period = {self.fold_period} ")
			self.update(period=self.fold_period)
		elif event.key in ['ctrl+t']:
			self.data_loader.update_test_mode()
			args = dict(title="Synthetic Sinusoids") if (self.data_loader.test_mode_index == 1) else {}
			self.update(**args)

	@exception_handled
	def process_ext_event(self, **event_data):
		if event_data['id'] == 'period-update':
			pm_name = event_data['ax']
			if pm_name != str(id(self.ax)):
				period = event_data['period']
				pm = self.period_markers.setdefault(pm_name, PeriodMarkers(pm_name, self.ax, color=event_data['color']))
				pm.update( self.origin, period )

	@exception_handled
	def get_ext_period(self) -> float:
		for pm in self.period_markers.values():
			if str(id(self.ax)) != pm.name:
				return pm.period

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
		xs, ys, self.period = self.get_element_data()
		self.origin = xs[np.argmax(np.abs(ys))]
		self.plot: Line2D = self.ax.plot(xs, ys, label='y', color='blue', marker=".", linewidth=1, markersize=2, alpha=0.5)[0]
		self.ax.title.set_text(f"{self.name}: TP={self.period:.3f} (F={1/self.period:.3f})")
		self.ax.title.set_fontsize(8)
		self.ax.title.set_fontweight('bold')
		self.ax.set_xlim(xs[0],xs[-1])
		self.update_period_marker()
		self.ax.set_ylim(ys.min(),ys.max())

	@exception_handled
	def get_element_data(self) -> Tuple[np.ndarray,np.ndarray,float]:
		element: xa.Dataset = self.data_loader.get_dataset_element(self.sector,self.TICS[self.element], refresh=self.refresh )
		t, y = element.data_vars['time'], element.data_vars['y']
		ydata: np.ndarray = y.values
		xdata: np.ndarray = t.values
		target: float = y.attrs['period']
		if self.fold_period is not None:
			xdata = xdata - np.floor(xdata/self.fold_period)*self.fold_period
		return xdata, znorm(ydata.squeeze()), target

	@exception_handled
	def update(self, val=0, **kwargs ):
		xdata, ydata, self.period = self.get_element_data()
		self.origin = xdata[np.argmax(np.abs(ydata))]
		self.plot.set_ydata(ydata)
		self.plot.set_xdata(xdata)
		self.plot.set_linewidth( 1 if (self.fold_period is None) else 0)
		fold_period = kwargs.get('period')
		active_period = self.period if (fold_period is None) else fold_period
		title = f"{self.name}: TP={active_period:.3f} (F={1/active_period:.3f})"
		self.ax.title.set_text( kwargs.get('title',title) )
		self.update_period_marker()
		self.ax.set_xlim(xdata.min(),xdata.max())
		try:  self.ax.set_ylim(ydata.min(),ydata.max())
		except: self.log.info( f" ------------------ Error in y bounds: {ydata.min()} -> {ydata.max()}" )
		self.log.info( f" ---- MITDatasetPlot-> update({self.element}:{self.TICS[self.element]}): xlim=({xdata.min():.3f},{xdata.max():.3f}), ylim=({ydata.min():.3f},{ydata.max():.3f}), xdata.shape={self.plot.get_xdata().shape} origin={self.origin} ---" )
		self.ax.figure.canvas.draw_idle()


class MITTransformPlot(SignalPlot):

	def __init__(self, name: str, data_loader: IterativeDataLoader, transform: Transform, sector: int, **kwargs):
		SignalPlot.__init__(self, **kwargs)
		self.name = name
		self.sector: int = sector
		self.transform: Transform = transform
		self.data_loader: IterativeDataLoader = data_loader
		self.TICS: List[str] = data_loader.TICS(sector)
		self.annotations: List[str] = tolower( kwargs.get('annotations',None) )
		self.colors = [ 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'darkviolet', 'darkorange', 'saddlebrown', 'darkturquoise' ]
		self.ofac = kwargs.get('upsample_factor',1)
		self.normtype = kwargs.get('norm', 'z')
		self.plots: List[Line2D] = []
		self.target_marker: Line2D = None
		self.selection_marker: Line2D = None
		self.add_param( STIntParam('element', (0,len(self.TICS))  ) )
		self.transax = None
		self.nlines = -1

	@property
	def tname(self):
		return self.transform.name

	def norm(self, embedding: np.ndarray):
		if self.normtype == "z":  return znorm(embedding)
		if self.normtype == "u":  return unorm(embedding)
		if self.normtype == "l2": return l2norm(embedding)
		raise Exception(f"MITTransformPlot.norm: unknown normtype={self.normtype}")

	def set_sector(self, sector: int ):
		self.sector = sector

	@exception_handled
	def _setup(self):
		series_data: xa.Dataset = self.data_loader.get_dataset_element(self.sector, self.TICS[self.element])
		period: float = series_data.data_vars['y'].attrs['period']
		freq = 1.0 / period
		tdata: np.ndarray = self.apply_transform(series_data)
		x = self.transform.xdata.squeeze()
		y = tdata[None,:] if (tdata.ndim == 1) else tdata
		self.nlines = y.shape[0]
		for ip in range(self.nlines):
			self.plots.append( self.ax.plot(x, y[ip], label=f"{self.tname}-{ip}", color=self.colors[ip], marker=".", linewidth=1, markersize=2, alpha=0.5)[0] )
		self.ax.set_xlim( self.transform.xdata.min(), self.transform.xdata.max() )
		self.ax.set_ylim( tdata.min(), tdata.max() )
		self.target_marker: Line2D = self.ax.axvline( freq, 0.0, 1.0, color='grey', linestyle='-', linewidth=3, alpha=0.5)
		self.selection_marker: Line2D = self.ax.axvline( 0, 0.0, 1.0, color='black', linestyle='-', linewidth=1, alpha=1.0)
		self.ax.title.set_text(f"{self.name}: TPeriod={period:.3f} (Freq={freq:.3f})")
		self.ax.title.set_fontsize(8)
		self.ax.title.set_fontweight('bold')
		self.ax.set_xscale('log')
		self.ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
		self.ax.xaxis.set_major_locator(ticker.LogLocator(base=2, numticks=8))
		self.ax.legend(loc="upper right", fontsize=8)

	@exception_handled
	def button_press(self, event: MouseEvent) -> Any:
		if event.inaxes == self.ax and (event.button == MouseButton.RIGHT):
			self.log.info(f"           *** ---- MITTransformPlot.button_press: selected freq={event.xdata:.2f} mods={event.modifiers} --- ")
			if "shift" in event.modifiers:
				freq, period = event.xdata, 1/event.xdata
				self.log.info(f"           *** ---- MITTransformPlot.button_press: selected freq={freq:.2f}, period={period:.2f} --- ")
				self.ax.title.set_text(f"{self.name}: TP={period:.3f} (F={freq:.3f})")
				self.selection_marker.set_xdata([freq, freq])
				self.process_event( id="period-update", period=period, ax=str(id(self.ax)), color=self.colors[0] )
			elif "ctrl" in event.modifiers:
				for t in self.transforms.values():
					t.process_event( id="crtl-mouse-press", x=event.xdata, y=event.ydata, ax=event.inaxes )
				self.update()

	def key_press(self, event: KeyEvent) -> Any:
		if event.key.startswith( 'ctrl+'):
			for t in self.transforms.values():
				t.process_event( id="KeyEvent", key=event.key, ax=event.inaxes )
			self.update()

	@exception_handled
	def apply_transform( self, series_data: xa.Dataset ) -> np.ndarray:
		slen = self.transform.cfg.series_length
		ts_tensors: Dict[str,Tensor] =  { k: FloatTensor(series_data.data_vars[k].values[:slen]).to(self.transform.device) for k in ['time','y'] }
		x,y = ts_tensors['time'].squeeze(), tnorm(ts_tensors['y'].squeeze())
		transformed: Tensor = self.transform.embed( x, y )
		embedding: np.ndarray = self.transform.magnitude( transformed )
		self.log.info( f"MITTransformPlot.apply_transform: x{list(x.shape)}, y{list(y.shape)} -> transformed{list(transformed.shape)}  embedding{list(embedding.shape)} ---> x min={embedding.min():.3f}, max={embedding.max():.3f}, mean={embedding.mean():.3f} ---")
		return self.norm(embedding)

	def update_selection_marker(self, freq ) -> float:
		period = 1/freq
		self.selection_marker.set_xdata([freq, freq])
		self.process_event(id="period-update", period=period, ax=str(id(self.ax)), color=self.colors[0])
		return period

	@exception_handled
	def update(self, val=0):
		series_data: xa.Dataset = self.data_loader.get_dataset_element(self.sector, self.TICS[self.element])
		target_period: float = series_data.data_vars['y'].attrs['period']
		tdata: np.ndarray = self.apply_transform(series_data)
		for ip in range(self.nlines):
			self.plots[ip].set_ydata(tdata[ip])
			self.plots[ip].set_xdata(self.transform.xdata)
		self.ax.set_xlim( self.transform.xdata.min(), self.transform.xdata.max() )
		self.ax.set_ylim( tdata.min(), tdata.max() )
		self.log.info(f"---- MITTransformPlot {self.tname}[{self.element})] update: tdata{tdata.shape}, x range=({self.transform.xdata.min():.3f}->{self.transform.xdata.max():.3f}) --- ")
		target_freq = self.transform.get_target_freq( target_period )
		self.target_marker.set_xdata([target_freq,target_freq])
		transform_peak_freq = self.transform.xdata[np.argmax(tdata)]
		transform_period = self.update_selection_marker(transform_peak_freq)
		self.ax.title.set_text(f"{self.name}: TP={transform_period:.3f} (F={transform_peak_freq:.3f})")
		self.ax.figure.canvas.draw_idle()

