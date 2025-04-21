from astrotime.util.series import TSet
from astrotime.loaders.MIT import MITOctavesLoader
from astrotime.encoders.spectral_autocorr import SpectralAutocorrelationLayer, embedding_space
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
plot_freq_space, embedding_space_tensor = embedding_space( cfg.transform, device )

dplot = MITDatasetPlot("Lightcurves", data_loader, sector )
transforms = dict( analysis  = SpectralAutocorrelationLayer( cfg.transform, embedding_space_tensor, device ) )
wplot = MITTransformPlot("Spectral Autocorrelation", data_loader, transforms, plot_freq_space, sector )

fig = SignalPlotFigure([dplot,wplot])
wplot.update()