from astrotime.util.series import TSet
from astrotime.loaders.MIT import MITOctavesLoader
from astrotime.encoders.octaves import OctaveAnalysisLayer, embedding_space
from plot.SCRAP.MIT import MITDatasetPlot, MITTransformPlot
from astrotime.config.context import astrotime_initialize
from astrotime.plot.base import SignalPlotFigure

import torch
from hydra import initialize, compose

version = "MIT_period.octaves"
overrides = []
initialize(version_base=None, config_path="../../../config")
cfg = compose( config_name=version, overrides=overrides )
device: torch.device = astrotime_initialize(cfg,version)
sector = cfg.data.sector_range[0]
refresh = True

data_loader = MITOctavesLoader( cfg.data )
data_loader.initialize( TSet.Train, test_mode=False )
plot_freq_space, embedding_space_tensor = embedding_space( cfg.transform, device )
octave_analysis  = OctaveAnalysisLayer( cfg.transform, embedding_space_tensor, device )

dplot = MITDatasetPlot("MIT lightcurves", data_loader, sector, refresh=refresh )
transforms = dict( octave_analysis_transform=octave_analysis )
wplot = MITTransformPlot("WWAnalysis Transform", data_loader, transforms, plot_freq_space, sector )
fig = SignalPlotFigure([dplot,wplot])
wplot.update()
