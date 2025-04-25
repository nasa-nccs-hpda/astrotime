from astrotime.util.series import TSet
from typing import List, Optional, Dict, Type, Tuple, Union
from astrotime.util.math import tnorm
from astrotime.loaders.MIT import MITOctavesLoader
from astrotime.encoders.wotan import DetrendTransform
from astrotime.config.context import astrotime_initialize
import torch, xarray as xa, numpy as np
from hydra import initialize, compose

def znorm(ydata: np.ndarray) -> np.ndarray:
	y0,y1 = ydata.min(), ydata.max()
	return (ydata-y0)/(y1-y0)

version = "MIT_period.octaves"
overrides = []
initialize(version_base=None, config_path="../config")
cfg = compose(config_name=version, overrides=overrides)
cfg.platform.gpu = -1
device: torch.device = astrotime_initialize(cfg, version)
sector = cfg.data.sector_range[0]
cfg.transform.detrend_window_length= 0.5
cfg.transform.detrend_method= 'biweight'

data_loader = MITOctavesLoader(cfg.data)
data_loader.initialize(TSet.Train)
transform=DetrendTransform(cfg.transform, device)
TICS = data_loader.TICS(sector)

for element in range(10):
	series_data: xa.Dataset = data_loader.get_dataset_element(sector, TICS[element] )
	slen = transform.cfg.series_length
	ts_tensors: Dict[str, torch.Tensor] = {k: torch.FloatTensor(series_data.data_vars[k].values[:slen]).to(transform.device) for k in ['time', 'y']}
	x, y = ts_tensors['time'].squeeze(), tnorm(ts_tensors['y'].squeeze())
	transformed: torch.Tensor = transform.embed(x, y)
	embedding: np.ndarray = transform.magnitude(transformed)
	print(f"MITTransformPlot.apply_transform: x{list(x.shape)}, y{list(y.shape)} -> embedding{list(embedding.shape)} ---> min={embedding.min():.3f}, max={embedding.max():.3f}, mean={embedding.mean():.3f} ---")

