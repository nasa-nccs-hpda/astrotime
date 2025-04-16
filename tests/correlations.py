
from astrotime.config.context import astrotime_initialize
from typing import List, Optional, Dict, Type, Union, Tuple
import torch, numpy as np
from hydra import initialize, compose

version = "MIT_period.octaves"
overrides = []
initialize(version_base=None, config_path="../config")
cfg = compose(config_name=version, overrides=overrides)
device: torch.device = astrotime_initialize(cfg, version)
pi2: float = 2*np.pi
tbounds: Tuple[float,float] = ( 300.0, 380.0 )
noise_std: float = 0.1
nts: int = 10000
ntaus: int = 50
ntp: int = 5
rwin = 3
trange: float = tbounds[1] - tbounds[0]
tstep: float = trange / nts
noise: np.ndarray = np.random.normal(0.0, noise_std, nts )
t: np.ndarray = np.linspace( tbounds[0], tbounds[1], nts )
taus: np.ndarray = np.linspace( tbounds[0], tbounds[1], ntaus )
ys: np.ndarray = np.sin( pi2*ntp*(t/trange) ) + noise
itaus: np.ndarray = np.searchsorted( t, taus )
yt: np.ndarray = np.stack( [ys,t], axis=1 )
yw: np.ndarray = np.stack( [ yt[itaus-rwin:itaus+rwin+1,:] for i in range(ntaus) ] )

print( f"yw{yw.shape}" )


