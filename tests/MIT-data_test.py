import xarray

from astrotime.util.series import TSet
from astrotime.loaders.MIT import MITLoader
from astrotime.config.context import astrotime_initialize
from astrotime.encoders.wavelet import embedding_space
from typing import Any, Dict, List, Optional, Tuple
from astrotime.util.math import npnorm
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
	nanmask = np.isnan(y)
	t, y = t[~nanmask], npnorm(y[~nanmask])
	dt = np.diff(t)
	slist.append( [t.size, np.median(np.abs(dt)), np.median(np.abs(np.diff(y))), y.std(), dt.std() ] )
stats = np.array(slist)
print( f"stats{shp(stats)}: ")
print( f"Length: {stats[:,0].min()} -> {stats[:,0].max()}, median={np.median(stats[:,0])}" )
print( f"DT: {stats[:,1].min():.7f} -> {stats[:,1].max():.7f}, median={np.median(stats[:,1]):.7f}")
print( f"DY: {stats[:,2].min():.7f} -> {stats[:,2].max():.7f}, median={np.median(stats[:,2]):.7f}" )
print( f"SY: {stats[:,3].min():.7f} -> {stats[:,3].max():.7f}, median={np.median(stats[:,3]):.7f} " )
print( f"SDT: {stats[:,4].min():.7f} -> {stats[:,4].max():.7f}, median={np.median(stats[:,4]):.7f} " )




#print( f" y{shp(y)} t{shp(t)}")