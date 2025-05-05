import logging, numpy as np
import xarray as xa
from astrotime.plot.param import Number, Parameter, STParam, STFloatParam, STFloatValuesParam, Parameterized
from torch import nn, optim, Tensor, FloatTensor
from omegaconf import DictConfig, OmegaConf
from matplotlib.backend_bases import KeyEvent, MouseEvent, MouseButton
from astrotime.loaders.base import IterativeDataLoader
from astrotime.util.logging import exception_handled
from .MIT import MITDatasetPlot
from typing import List, Optional, Dict, Type, Union, Tuple, Any, Set
log = logging.getLogger()

class MITSyntheticPlot(MITDatasetPlot):

	def __init__(self, name: str, cfg: DictConfig, data_loader: IterativeDataLoader, sector: int, **kwargs):
		MITDatasetPlot.__init__(self, name, data_loader, sector, **kwargs)
		self.arange: Tuple[float,float] = cfg.arange
		self.wrange: Tuple[float, float] = cfg.wrange
		self.nrange: Tuple[float, float] = cfg.nrange
		self.add_param( STFloatParam('width', self.wrange ) )
		self.add_param(STFloatParam('amplitude', self.arange))
		self.add_param(STFloatParam('noise', self.nrange))

	@exception_handled
	def button_press(self, event: MouseEvent) -> Any:
		MITDatasetPlot.button_press(self, event)

	@exception_handled
	def key_press(self, event: KeyEvent) -> Any:
		MITDatasetPlot.key_press(self, event)

	@exception_handled
	def process_ext_event(self, **event_data):
		MITDatasetPlot.process_ext_event(self,**event_data)

	@exception_handled
	def button_release(self, event: MouseEvent) -> Any:
		MITDatasetPlot.button_release(self, event)

	@exception_handled
	def on_motion(self, event: MouseEvent) -> Any:
		MITDatasetPlot.on_motion(self, event)

	@exception_handled
	def _setup(self):
		MITDatasetPlot._setup(self)

	@exception_handled
	def update(self, val=0, **kwargs ):
		MITDatasetPlot.update(self, val, **kwargs)

