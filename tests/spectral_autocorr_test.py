from astrotime.util.series import TSet
from astrotime.loaders.MIT import MITOctavesLoader
from astrotime.encoders.spectral_autocorr import HarmonicsFilterLayer
from astrotime.plot.MIT import MITDatasetPlot, MITTransformPlot
from astrotime.config.context import astrotime_initialize
from astrotime.plot.base import SignalPlotFigure
import torch
from hydra import initialize, compose

version = "MIT_period.octaves"
overrides = []
initialize(version_base=None, config_path="../config" )
cfg = compose( config_name=version, overrides=overrides )
device: torch.device = astrotime_initialize(cfg,version)
sector = cfg.data.sector_range[0]

data_loader = MITOctavesLoader( cfg.data )
data_loader.initialize( TSet.Train )

dplot = MITDatasetPlot("Lightcurves", data_loader, sector )
transforms = dict( analysis  = HarmonicsFilterLayer( cfg.transform, device ) )
wplot = MITTransformPlot("Spectral Autocorrelation", data_loader, transforms, sector )

fig = SignalPlotFigure([dplot,wplot])
wplot.update()