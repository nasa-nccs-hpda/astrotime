import xarray

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
TICS: List[str] = data_loader.TICS(sector)
embedding_space_array, embedding_space_tensor = embedding_space( cfg.transform, device )
data_loader.load_sector( sector )
dset: xarray.Dataset = data_loader.dataset
slist = []
for TIC in TICS:
	t: np.ndarray = dset[TIC + ".time"].values
	y: np.ndarray = dset[TIC + ".y"].values
	slist.append( [t.size, np.median(np.diff(t)), np.median(np.diff(y)),  y.max()-y.min()] )
stats = np.array(slist)
print( f"stats{shp(stats)}: ")
# print( f"Length: {nL.min()} -> {nL.max()}, median={np.median(nL)}" )
# print( f"DT: {nT.min():.3f} -> {nT.max():.3f}, median={np.median(nT)}" )
# print( f"Y: {nY.min():.3f} -> {nY.max():.3f}, median range={np.median(nY.max(axis=1)-nY.min(axis=1))}" )


#print( f" y{shp(y)} t{shp(t)}")