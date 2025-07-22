from astrotime.loaders.MIT import MITElementLoader
from astrotime.config.context import astrotime_initialize
import torch
from hydra import initialize, compose

version = "select_MIT_period"
overrides = [ 'platform.gpu=-1', 'data.snr_min=0', 'data.snr_max=20' ]
initialize(version_base=None, config_path="../../../config")
cfg = compose( config_name=version, overrides=overrides )
device: torch.device = astrotime_initialize(cfg,version+".plot")

data_loader = MITElementLoader( cfg.data, preload=True )
element = data_loader.get_element( 0 )
print( f"elem keys = {element.keys()}")