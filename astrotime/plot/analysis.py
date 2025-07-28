import logging, os, csv, pickle, numpy as np
from .param import STIntParam, STFloatParam
from matplotlib import ticker
from torch import nn, optim, Tensor, FloatTensor
from astrotime.util.series import TSet
from .base import SignalPlot, bounds
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.backend_bases import KeyEvent, MouseEvent, MouseButton
from astrotime.loaders.base import IterativeDataLoader, ElementLoader, RDict
from astrotime.util.logging import exception_handled
from astrotime.encoders.embedding import Transform
from astrotime.trainers.model_evaluator import ModelEvaluator
from typing import List, Optional, Dict, Type, Union, Tuple, Any, Set
from astrotime.util.math import tnorm
log = logging.getLogger()

def sH(h: float|np.ndarray) -> str:
	if type(h) is np.ndarray:
		h = h.item() if h.ndim == 0 else h[0]
	if abs(h) > 1:
		return str(round(h))
	else:
		sh = round(1/h)
		return f"1/{sh}" if sh > 1 else str(sh)

def sL(l: float|np.ndarray) -> str:
	if type(l) is np.ndarray:
		l = l.item() if l.ndim == 0 else l[0]
	return f"{l:.4f}"

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
		self.npm: int = kwargs.get('npm', 40 )
		self.color: str = kwargs.get('color', 'red' )
		self.alpha: float = kwargs.get('alpha', 0.35 )
		self.linestyle: str = kwargs.get('linestyle', '-')
		self.linewidth: int = kwargs.get('linewidth', 1)

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
			tval = self.origin + (pid-self.npm//2)*self.period
			if pid >= len(self.markers):  self.markers.append( self.ax.axvline( tval, self.yrange[0], self.yrange[1], color=self.color, linestyle=self.linestyle, alpha=self.alpha, linewidth=self.linewidth) )
			else:                         self.markers[pid].set_xdata([tval,tval])

class RawDatasetPlot(SignalPlot):

	def __init__(self, name: str, data_loader: ElementLoader, **kwargs):
		SignalPlot.__init__(self, **kwargs)
		self.name = name
		self.version = name.split(':')[0]
		self.data_loader: ElementLoader = data_loader.set_tset( TSet.Train )
		self.annotations: List[str] = tolower( kwargs.get('annotations',None) )
		self.ofac = kwargs.get('upsample_factor',1)
		self.plot: Line2D = None
		self.add_param( STIntParam('element', (0,self.data_loader.nelem), key_press_mode=1 ) )
		self.add_param( STIntParam('file', (0, self.data_loader.nfiles), key_press_mode=2 ) )
		self.period_markers: Dict[str,PeriodMarkers] = {}
		self.ext_pm_ids: Set[str] = set()
		self.transax = None
		self.origin = None
		self.period = None
		self.fold_period = None
		self.transforms = {}
		self.snr = 0.0

	@exception_handled
	def update_period_marker(self) -> str:
		pm_name= str(id(self.ax))
		pm = self.period_markers.setdefault( pm_name, PeriodMarkers( pm_name, self.ax, color="black", alpha=0.8 ) )
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
	def process_ext_event(self, **event_data):
		if event_data['id'] == 'period-update':
			pm_name = event_data['ax']
			if pm_name != str(id(self.ax)):
				period = event_data['period']
				pm = self.period_markers.setdefault(pm_name, PeriodMarkers(pm_name, self.ax, color=event_data['color'], linewidth=2 ) )
				pm.update( self.origin, period )
				title = f"{self.name},{self.file},{self.element}): TP={self.period:.3f} (TF={1/self.period:.3f}), MP={period:.3f} (MF={1/period:.3f}) snr={self.snr:.3f}"
				self.ax.title.set_text(title)

	@exception_handled
	def get_ext_period(self) -> float:
		for pm in self.period_markers.values():
			if str(id(self.ax)) != pm.name:
				return pm.period
		return np.nan

	@exception_handled
	def button_release(self, event: MouseEvent) -> Any:
		pass

	@exception_handled
	def key_press(self, event: KeyEvent) -> Any:
		if event.key in ['ctrl+f','alt+ƒ']:
			if self.fold_period is None:    self.fold_period = self.period if (event.key == 'ctrl+f') else self.get_ext_period()
			else :                          self.fold_period = None
			self.log.info(f"                 DatasetPlot-> key_press({event.key}), fold period = {self.fold_period} ")
			self.update(period=self.fold_period)

	@exception_handled
	def on_motion(self, event: MouseEvent) -> Any:
		pass

	@exception_handled
	def _setup(self):
		xs, ys, self.period, self.snr, stype = self.get_element_data()
		if ys is not None:
			self.origin = xs[np.argmax(np.abs(ys))]
			self.plot: Line2D = self.ax.plot(xs, ys, label='y', color='blue', marker=".", linewidth=1, markersize=2, alpha=0.5)[0]
			self.ax.title.set_text(f"{self.name}({stype},{self.file},{self.element}): TP={self.period:.3f} (F={1/self.period:.3f})")
			self.ax.title.set_fontsize(8)
			self.ax.title.set_fontweight('bold')
			self.ax.set_xlim(xs[0],xs[-1])
			self.update_period_marker()
			self.ax.set_ylim(ys.min(),ys.max())
		else:
			self.ax.title.set_text("Plot Error: See log file for details")

	def get_element_data(self) -> Tuple[np.ndarray,np.ndarray,float,float,str]:
		self.data_loader.set_file(self.file)
		element: Dict[str,Union[np.ndarray,float]] = self.data_loader.get_element(self.element)
		if element is not None:
			self.log.info(f" * DatasetPlot-> get_element_data({self.element}): keys={list(element.keys())}")
			ydata: np.ndarray = element['y']
			xdata: np.ndarray = element['t']
			stype = element.get('type','LC')
			target: float = element['p'] if ('p' in element) else element['period']
			snr: float = element.get('sn',0.0)
			if self.fold_period is not None:
				xdata = xdata - np.floor(xdata/self.fold_period)*self.fold_period
			return xdata, ydata.squeeze(), target, snr, stype
		else:
			return None,None,None,None,None

	@exception_handled
	def update(self, val=0, **kwargs ):
		xdata, ydata, self.period, self.snr, stype = self.get_element_data()
		if ydata is not None:
			self.log.debug(f" ---------> get_element_data: xdata{xdata.shape}, ydata{ydata.shape}, period={self.period:.3f}, snr={self.snr:.3f}, stype={stype}")
			self.origin = xdata[np.argmax(np.abs(ydata))]
			self.plot.set_ydata(ydata)
			self.plot.set_xdata(xdata)
			fold_period = kwargs.get('period')
			active_period = self.period if (fold_period is None) else fold_period
			title = f"{self.name}({stype},{self.file},{self.element}): TP={active_period:.3f}"
			self.ax.title.set_text( kwargs.get('title',title) )
			self.update_period_marker()
			self.ax.set_xlim(xdata.min(),xdata.max())
			self.ax.set_ylim(ydata.min(), ydata.max())
			try:  self.ax.set_ylim(ydata.min(),ydata.max())
			except: self.log.info( f" ------------------ Error in y bounds: {ydata.min()} -> {ydata.max()}" )
			self.log.info( f" ---- ----> E-{self.element}: xlim=({xdata.min():.3f},{xdata.max():.3f}), ylim=({ydata.min():.3f},{ydata.max():.3f}), xdata.shape={self.plot.get_xdata().shape} origin={self.origin} ---" )
			self.ax.figure.canvas.draw_idle()
		else:
			self.ax.title.set_text("Plot Error: See log file for details")

class DatasetPlot(SignalPlot):

	def __init__(self, name: str, data_loader: IterativeDataLoader, sector:int=0, **kwargs):
		SignalPlot.__init__(self, **kwargs)
		self.name = name
		self.version = name.split(':')[0]
		self.sector: int = sector
		self.data_loader: IterativeDataLoader = data_loader
		self.refresh = kwargs.get('refresh', False)
		self.annotations: List[str] = tolower( kwargs.get('annotations',None) )
		self.ofac = kwargs.get('upsample_factor',1)
		self.plot: Line2D = None
		self.add_param( STIntParam('element', (0,self.data_loader.nelements)  ) )
		self.period_markers: Dict[str,PeriodMarkers] = {}
		self.ext_pm_ids: Set[str] = set()
		self.transax = None
		self.origin = None
		self.period = None
		self.model_period: float = 0.0
		self.registered_elements: Dict[Tuple[int,int],Tuple[float,float]] = self.load_registered_elements()

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
			self.log.info(f"                 DatasetPlot-> key_press({event.key}), fold period = {self.fold_period} ")
			self.update(period=self.fold_period)
		elif event.key in ['ctrl+t']:
			self.data_loader.update_test_mode()
			args = dict(title=" Sinusoids") if (self.data_loader.test_mode_index == 1) else {}
			self.update(**args)
		elif event.key in ['ctrl+s']:
			self.register_element()

	@exception_handled
	def process_ext_event(self, **event_data):
		if event_data['id'] == 'period-update':
			pm_name = event_data['ax']
			if pm_name != str(id(self.ax)):
				period = event_data['period']
				pm = self.period_markers.setdefault(pm_name, PeriodMarkers(pm_name, self.ax, color=event_data['color']))
				pm.update( self.origin, period )
				self.model_period = period
				title = f"{self.name}({self.sector},{self.element}): TP={self.period:.3f} (TF={1/self.period:.3f}), MP={period:.3f} (MF={1/period:.3f})"
				self.ax.title.set_text(title)

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
		xs, ys, self.period, snr, stype = self.get_element_data()
		self.origin = xs[np.argmax(np.abs(ys))]
		self.plot: Line2D = self.ax.plot(xs, ys, label='y', color='blue', marker=".", linewidth=1, markersize=2, alpha=0.5)[0]
		self.ax.title.set_text(f"{self.name}({stype},{self.sector},{self.element}): TP={self.period:.3f} (F={1/self.period:.3f})")
		self.ax.title.set_fontsize(8)
		self.ax.title.set_fontweight('bold')
		self.ax.set_xlim(xs[0],xs[-1])
		self.update_period_marker()
		self.ax.set_ylim(ys.min(),ys.max())

	def register_element(self):
		element: Dict[str,Union[np.ndarray,float]] = self.data_loader.get_element(self.sector,self.element ) # , refresh=self.refresh )
		f: float = 1/element['p']
		snr: float = element['sn']
		self.registered_elements[ (self.sector,self.element) ] = (f,snr)
		self.save_registered_elements()

	@property
	def refile(self):
		cdir = f"{self.data_loader.cfg.cache_path}/registered_elements"
		os.makedirs(cdir,exist_ok=True)
		return f"{cdir}/{self.version}.pkl"

	def save_registered_elements(self):
		with open( self.refile, 'wb') as file:
			pickle.dump(self.registered_elements, file)
			self.log.info( f" ---- DatasetPlot-> save_registered_elements: {self.refile} saved" )

	def load_registered_elements(self) -> Dict[Tuple[int,int],Tuple[float,float]]:
		if os.path.exists(self.refile):
			with open(self.refile, 'rb') as file:
				rv = pickle.load( file )
				self.log.info( f" ---- DatasetPlot-> loaded {len(rv)} registered elements from {self.refile}")
				return rv
		return {}

	@exception_handled
	def get_element_data(self) -> Tuple[np.ndarray,np.ndarray,float,float,str]:
		self.data_loader.set_params( { pn: pv.value_selected() for pn, pv in self._sparms.items()} )
		element: Optional[RDict] = self.data_loader.get_element(self.sector,self.element) # , refresh=self.refresh )
		ydata: np.ndarray = element['y']
		xdata: np.ndarray = element['t']
		stype = element.get('type','LC')
		target: float = element['p'] if ('p' in element) else element['period']
		snr: float = element.get('sn',0.0)
		return xdata, znorm(ydata.squeeze()), target, snr, stype

	@exception_handled
	def update(self, val=0, **kwargs ):
		xdata, ydata, self.period, snr, stype = self.get_element_data()
		self.origin = xdata[np.argmax(np.abs(ydata))]
		self.plot.set_ydata(ydata)
		self.plot.set_xdata(xdata)
		self.plot.set_linewidth( 1 if (self.fold_period is None) else 0)
		fold_period = kwargs.get('period')
		active_period = self.period if (fold_period is None) else fold_period
		title = f"{self.name}({stype},{self.sector},{self.element}): TP={active_period:.3f} (TF={1 / active_period:.3f}), MP={self.model_period:.3f} (MF={1/self.model_period:.3f})"
		self.ax.title.set_text( kwargs.get('title',title) )
		self.update_period_marker()
		self.ax.set_xlim(xdata.min(),xdata.max())
		try:  self.ax.set_ylim(ydata.min(),ydata.max())
		except: self.log.info( f" ------------------ Error in y bounds: {ydata.min()} -> {ydata.max()}" )
		self.log.info( f" ---- DatasetPlot-> update({self.element}: xlim=({xdata.min():.3f},{xdata.max():.3f}), ylim=({ydata.min():.3f},{ydata.max():.3f}), xdata.shape={self.plot.get_xdata().shape} origin={self.origin} ---" )
		self.ax.figure.canvas.draw_idle()


class TransformPlot(SignalPlot):

	def __init__(self, name: str, data_loader: ElementLoader, transform: Transform, **kwargs):
		SignalPlot.__init__(self, **kwargs)
		self.name = name
		self.transform: Transform = transform
		self.data_loader: ElementLoader = data_loader
		self.annotations: List[str] = tolower( kwargs.get('annotations',None) )
		self.colors = [ 'black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'darkviolet', 'darkorange', 'saddlebrown', 'darkturquoise' ]
		self.ofac = kwargs.get('upsample_factor',1)
		self.plots: List[Line2D] = []
		self.target_marker: Line2D = None
		self.selection_marker: Line2D = None
		self.add_param( STIntParam('element', (0,self.data_loader.nelements)  ) )
		self.transax = None
		self.nlines = -1
		self.transforms = {}

	@property
	def tname(self):
		return self.transform.name

	def set_sector(self, sector: int ):
		self.sector = sector

	@exception_handled
	def _setup(self):
		series_data: Dict[str,Union[np.ndarray,float]] = self.data_loader.get_element(self.element)
		period: float = series_data['p']
		freq = 1.0 / period
		tdata: np.ndarray = self.apply_transform(series_data).squeeze()
		x = self.transform.xdata.squeeze()
		y = tdata[None,:] if (tdata.ndim == 1) else tdata
		self.nlines = y.shape[0]
		print( f"PLOT: x{list(x.shape)} y{list(y.shape)} ")
		for ip in range(self.nlines):
			alpha = 1.0 - ip/self.nlines
			lw = 2 if (ip == 0) else 1
			self.plots.append( self.ax.plot(x, y[ip], label=f"{self.tname}-{ip}", color=self.colors[ip], marker=".", linewidth=lw, markersize=lw, alpha=alpha)[0] )
		self.ax.set_xlim( x.min(), x.max() )
		self.ax.set_ylim( y.min(), y.max() )
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
			self.log.info(f"           *** ---- TransformPlot.button_press: selected freq={event.xdata:.2f} mods={event.modifiers} --- ")
			if "shift" in event.modifiers:
				freq, period = event.xdata, 1/event.xdata
				self.log.info(f"           *** ---- TransformPlot.button_press: selected freq={freq:.2f}, period={period:.2f} --- ")
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
	def apply_transform( self, series_data: Dict[str,Union[np.ndarray,float]] ) -> np.ndarray:
		ts_tensors: Dict[str,Tensor] =  { k: FloatTensor(series_data[k]).to(self.transform.device) for k in ['t','y'] }
		x,y = ts_tensors['t'].squeeze(), tnorm(ts_tensors['y'].squeeze())
		transformed: Tensor = self.transform.embed( x, y )
		embedding: np.ndarray = self.transform.magnitude( transformed )
		self.log.info( f"TransformPlot.apply_transform: x{list(x.shape)}, y{list(y.shape)} -> transformed{list(transformed.shape)}  embedding{list(embedding.shape)} ---> x min={embedding.min():.3f}, max={embedding.max():.3f}, mean={embedding.mean():.3f} ---")
		return embedding

	def update_selection_marker(self, freq ) -> float:
		period = 1/freq
		self.selection_marker.set_xdata([freq, freq])
		self.process_event(id="period-update", period=period, ax=str(id(self.ax)), color=self.colors[0])
		return period

	@exception_handled
	def update(self, val=0):
		series_data: Dict[str,Union[np.ndarray,float]] = self.data_loader.get_element(self.element)
		target_period: float = series_data['p']
		tdata: np.ndarray = self.apply_transform(series_data).squeeze()
		x = self.transform.xdata.squeeze()
		y = tdata[None,:] if (tdata.ndim == 1) else tdata
		for ip in range(self.nlines):
			self.plots[ip].set_ydata(y[ip])
			self.plots[ip].set_xdata(x)
		self.ax.set_xlim( x.min(), x.max() )
		self.ax.set_ylim( y.min(), y.max() )
		self.log.info(f"---- TransformPlot {self.tname}[{self.element})] update: y{y.shape}, x range=({x.min():.3f}->{x.max():.3f}) --- ")
		target_freq = self.transform.get_target_freq( target_period )
		self.target_marker.set_xdata([target_freq,target_freq])
		transform_peak_freq = self.transform.xdata[np.argmax(y[0])]
		transform_period = self.update_selection_marker(transform_peak_freq)
		self.ax.title.set_text(f"{self.name}: TP={transform_period:.3f} (F={transform_peak_freq:.3f})")
		self.ax.figure.canvas.draw_idle()


class EvaluatorPlot(SignalPlot):

	def __init__(self, name: str, evaluator: ModelEvaluator, **kwargs):
		SignalPlot.__init__(self, **kwargs)
		self.name = name
		self.evaluator: ModelEvaluator = evaluator
		self.annotations: List[str] = tolower( kwargs.get('annotations',None) )
		self.colors = [ 'red', 'blue', 'magenta', 'cyan', 'darkviolet', 'darkorange', 'saddlebrown', 'darkturquoise', 'green', 'brown', 'purple', 'yellow', 'olive', 'pink', 'gold', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey']
		self.marker_colors = ['black', 'green']
		self.ofac = kwargs.get('upsample_factor',1)
		self.plots: List[Line2D] = []
		self.target_marker: Line2D = None
		self.model_marker: Line2D = None
		self.nfiles = self.evaluator.loader.nfiles
		self.add_param( STIntParam('element', (0,self.nelements)  ) )
		self.add_param( STIntParam('file', (0, self.nfiles), key_press_mode=2) )
		self.transax = None
		self.nlines = -1
		self.transforms = {}

	@property
	def nelements(self) -> int:
		return self.evaluator.nelements

	@property
	def tname(self):
		return self.evaluator.tname

	def _setup(self):
		tdata = self.evaluator.evaluate(self.element).squeeze()
		target_freq = self.evaluator.target_frequency
		model_freq = self.evaluator.model_frequency
		loss =  self.evaluator.lossdata['loss']
		x = self.evaluator.xdata.cpu().numpy()
		y = tdata[None,:] if (tdata.ndim == 1) else tdata
		self.nlines = y.shape[0]
		print( f"PLOT: x{x.shape} y{y.shape}")
		for ip in range(self.nlines):
			self.plots.append( self.ax.plot(x, y[ip], label=f"{self.tname}-{ip}", color=self.colors[ip], marker=".", linewidth=1, markersize=1, alpha=4.0/(ip+4) )[0] )
		self.ax.set_xlim( x.min(), x.max() )
		self.ax.set_ylim( y.min(), y.max() )

		self.target_marker: Line2D = self.ax.axvline( target_freq, 0.0, 1.0, color=self.marker_colors[0], linestyle='-', linewidth=2, alpha=0.7)
		self.model_marker: Line2D  = self.ax.axvline( model_freq,  0.0, 1.0, color=self.marker_colors[1], linestyle='-', linewidth=2, alpha=0.7)
		self.ax.title.set_text(f"{self.name}: target({self.file},{self.element})={target_freq:.3f} model({self.marker_colors[1]})={model_freq:.3f}, loss={sL(loss)}")
		self.ax.title.set_fontsize(8)
		self.ax.title.set_fontweight('bold')
		self.ax.set_xscale('log')
		self.ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
		self.ax.xaxis.set_major_locator(ticker.LogLocator(base=2, numticks=8))
		self.ax.legend(loc="upper right", fontsize=8)

	@exception_handled
	def button_press(self, event: MouseEvent) -> Any:
		if event.inaxes == self.ax and (event.button == MouseButton.RIGHT):
			self.log.info(f"           *** ---- TransformPlot.button_press: selected freq={event.xdata:.2f} mods={event.modifiers} --- ")
			if "shift" in event.modifiers:
				freq, period = event.xdata, 1/event.xdata
				self.log.info(f"           *** ---- TransformPlot.button_press: selected freq={freq:.2f}, period={period:.2f} --- ")
				self.ax.title.set_text(f"{self.name}: TP={period:.3f} (F={freq:.3f})")
				self.model_marker.set_xdata([freq, freq])
				self.process_event( id="period-update", period=period, ax=str(id(self.ax)), color=self.colors[0] )
			elif "ctrl" in event.modifiers:
				for t in self.transforms.values():
					t.process_event( id="crtl-mouse-press", x=event.xdata, y=event.ydata, ax=event.inaxes )
				self.update()

	def key_press(self, event: KeyEvent) -> Any:
		if event.key.startswith( 'ctrl+'):
			for t in self.transforms.values():
				t.process_event( id="KeyEvent", key=event.key, ax=event.inaxes )
			evresult = self.evaluator.process_event( id="KeyEvent", key=event.key, ax=event.inaxes )
			if evresult is not None:
				pass
			self.update()

	@exception_handled
	def update(self, val=0):
		tdata = self.evaluator.evaluate(self.element).squeeze()
		target_freq = self.evaluator.target_frequency
		model_freq = self.evaluator.model_frequency
		loss =  self.evaluator.lossdata['loss']
		x = self.evaluator.xdata.cpu().numpy()
		y = tdata[None,:] if (tdata.ndim == 1) else tdata

		for ip in range(self.nlines):
			self.plots[ip].set_ydata(y[ip])
			self.plots[ip].set_xdata(x)
		self.ax.set_xlim( x.min(), x.max() )
		self.ax.set_ylim( y.min(), y.max() )
		self.log.info(f"---- TransformPlot {self.tname}[{self.element})] update: y{y.shape}, x range=({x.min():.3f}->{x.max():.3f}), model_freq={model_freq:.3f}  ")

		self.target_marker.set_xdata([target_freq,target_freq])
		self.model_marker.set_xdata( [model_freq, model_freq] )
		self.process_event(id="period-update", period=1/model_freq,  ax=str(id(self.ax)), color=self.marker_colors[1])
		self.ax.title.set_text(f"{self.name}({self.file},{self.element}): target_freq={target_freq:.3f} (model_freq={model_freq:.3f}), loss={sL(loss)}")
		self.ax.figure.canvas.draw_idle()

