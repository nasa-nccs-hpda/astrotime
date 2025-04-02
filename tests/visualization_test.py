from astrotime.util.series import TSet
from astrotime.loaders.MIT import MITLoader
from astrotime.encoders.wavelet import WaveletAnalysisLayer, embedding_space, WaveletSynthesisLayer
from astrotime.plot.MIT import MITDatasetPlot, MITTransformPlot
from astrotime.config.context import astrotime_initialize
from astrotime.plot.base import SignalPlotFigure
import torch
from hydra import initialize, compose

version = "MIT_period.wp"
overrides = []
initialize(version_base=None, config_path="../config" )
cfg = compose( config_name=version, overrides=overrides )
device: torch.device = astrotime_initialize(cfg,version)
sector = cfg.data.sector_range[0]

data_loader = MITLoader( cfg.data )
data_loader.initialize( TSet.Train, test_mode=False )
embedding_space_array, embedding_space_tensor = embedding_space( cfg.transform, device )

dplot = MITDatasetPlot("MIT lightcurves", data_loader, sector )
transforms = dict( analysis  = WaveletAnalysisLayer(  cfg.transform, embedding_space_tensor, device ),
	               synthesis = WaveletSynthesisLayer( cfg.transform, embedding_space_tensor, device ) )
wplot = MITTransformPlot("WWAnalysis Transform", data_loader, transforms, embedding_space_array, sector )

fig = SignalPlotFigure([dplot,wplot])