from astrotime.util.series import TSet
from astrotime.loaders.MIT import MITLoader
from astrotime.config.context import astrotime_initialize
from astrotime.encoders.wavelet import embedding_space
from typing import Any, Dict, List, Optional, Tuple
from astrotime.util.math import shp
import torch, numpy as np
from hydra import initialize, compose

version = "MIT_period.ce"
overrides = []
initialize(version_base=None, config_path="../config" )
cfg = compose( config_name=version, overrides=overrides )
device: torch.device = astrotime_initialize(cfg,version)
sector = cfg.data.sector_range[0]


data_loader = MITLoader( cfg.data )
data_loader.initialize( TSet.Train, test_mode=False )
embedding_space_array, embedding_space_tensor = embedding_space( cfg.transform, device )

batch: Dict[str,np.ndarray] = data_loader.get_batch( sector, 0 )
y: np.ndarray = batch['y']
t: np.ndarray = batch['t']

print( f" y{shp(y)} t{shp(t)}")