from astrotime.util.series import TSet
from astrotime.loaders.MIT import MITLoader
from astrotime.config.context import astrotime_initialize
import torch
import xarray as xa
from hydra import initialize, compose

version = "MIT_period.wp"
overrides = []
initialize(version_base=None, config_path="../config" )
cfg = compose( config_name=version, overrides=overrides )
device: torch.device = astrotime_initialize(cfg,version)
sector = cfg.data.sector_range[0]


data_loader = MITLoader( cfg.data )
data_loader.initialize( TSet.Train )
TICS = data_loader.TICS(sector)

dset: xa.Dataset = data_loader.get_pcross_element(  sector, TICS[0] )