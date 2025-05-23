import logging, numpy as np
import xarray as xa
import torch
from .param import STIntParam
from astrotime.loaders.base import DataLoader, IterativeDataLoader
from .base import SignalPlot, bounds
from matplotlib.lines import Line2D
from astrotime.util.logging import exception_handled
from astrotime.encoders.embedding import EmbeddingLayer
from typing import List, Optional, Dict, Type, Union, Tuple
log = logging.getLogger()

def tolower(ls: Optional[List[str]]) -> List[str]:
	return [a.lower() for a in ls] if (ls is not None) else []

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


class SignalTransformPlot(SignalPlot):

	def __init__(self, name: str, data_loader: DataLoader, transform: EmbeddingLayer, device: torch.device, **kwargs):
		SignalPlot.__init__(self, **kwargs)
		self.name = name
		self.device = device
		self.transform = transform
		self.data_loader: DataLoader = data_loader
		self.dset_idx = data_loader.dset_idx
		self.annotations: List[str] = tolower( kwargs.get('annotations',None) )
		self.colors = ['blue', 'green'] + [ 'yellow' ] * 16
		self.ofac = kwargs.get('upsample_factor',1)
		self.lines: Dict[str,Line2D] = {}
		self.add_param( STIntParam('element', ( 0, data_loader.nelements )  ) )
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
		element: Dict[str,Union[np.ndarray,float]] = self.data_loader.get_element(self.dset_idx,self.element)
		t: np.ndarray = element['t']
		yt: torch.Tensor = torch.from_numpy(element['y']).to(self.device)
		xt: torch.Tensor = torch.from_numpy(t).to(self.device)
		p: float = element['p']
		embedding: torch.Tensor = self.transform.embed(xt[None,:],yt[None,:])
		f: np.ndarray = self.transform.xdata.cpu().numpy().squeeze()
		trans = self.transform.magnitude(embedding).squeeze()
		return f, trans, p

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


class IterativeDataTransformPlot(SignalPlot):

	def __init__(self, name: str, data_loader: IterativeDataLoader, transform: EmbeddingLayer, **kwargs):
		SignalPlot.__init__(self, **kwargs)
		self.name = name
		self.transform = transform
		self.data_loader: IterativeDataLoader = data_loader
		self.dset_idx = data_loader.dset_idx
		self.annotations: List[str] = tolower( kwargs.get('annotations',None) )
		self.colors = ['blue', 'green'] + [ 'yellow' ] * 16
		self.ofac = kwargs.get('upsample_factor',1)
		self.lines: Dict[str,Line2D] = {}
		self.add_param( STIntParam('dset', (0, data_loader.ndsets)))
		self.add_param( STIntParam('element', (0,data_loader.nelements)  ) )
		self.transax = None

	def set_dset_index(self, dset_index: int ):
		self.dset_idx = dset_index
		self.data_loader.get_dataset( self.dset_idx )

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
		element: Dict[str,Union[np.ndarray,float]] = self.data_loader.get_element(self.dset_idx,self.element)
		ydata: np.ndarray = element['y']
		xdata: np.ndarray = element['t']
		target: float = element['p'] if ('p' in element) else element['period']
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
		self.lines['y'].set_ydata(ydata)
	#	self.lines['target'].remove()
	#	self.lines['target'] = self.ax.axvline(x=1.0/target, color='r', linestyle='-')
		self.ax.set_ylim(*bounds(ydata))
		self.plot.set_xdata(xdata)
		self.ax.set_xlim(*bounds(xdata))

