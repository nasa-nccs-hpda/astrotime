import logging
from astrotime.plot.param import STFloatParam
from omegaconf import DictConfig
from matplotlib.backend_bases import KeyEvent, MouseEvent
from astrotime.loaders.base import IterativeDataLoader
from astrotime.util.logging import exception_handled
from plot.SCRAP.MIT import MITDatasetPlot
from typing import Tuple, Any

log = logging.getLogger()

class MITSyntheticPlot(MITDatasetPlot):

	def __init__(self, name: str, cfg: DictConfig, data_loader: IterativeDataLoader, sector: int, **kwargs):
		MITDatasetPlot.__init__(self, name, data_loader, sector, **kwargs)
		self.arange: Tuple[float,float]  = cfg.get('arange',(0.01,0.1) )
		self.wrange: Tuple[float, float] = cfg.get('wrange',(0.01,0.1) )
		self.nrange: Tuple[float, float] = cfg.get('nrange',(0.01,0.1) )
		self.add_param( STFloatParam('width', self.wrange ) )
		self.add_param( STFloatParam('amplitude', self.arange ) )
		self.add_param( STFloatParam('noise', self.nrange ) )

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
		MITDatasetPlot.update( self, val, **kwargs )

