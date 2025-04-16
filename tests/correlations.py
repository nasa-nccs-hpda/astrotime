
from astrotime.config.context import astrotime_initialize
from typing import List, Optional, Dict, Type, Union, Tuple
import torch, numpy as np
from hydra import initialize, compose

version = "test.desktop"
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
rwin: int = 3

trange: float = tbounds[1] - tbounds[0]
tstep: float = trange / nts
twbnds = ( tbounds[0]+(rwin+1)*tstep, tbounds[1]-(rwin+1)*tstep )
noise: np.ndarray = np.random.normal(0.0, noise_std, nts )
t: np.ndarray = np.linspace( tbounds[0], tbounds[1], nts )
taus: np.ndarray = np.linspace( twbnds[0], twbnds[1], ntaus )
ys: np.ndarray = np.sin( pi2*ntp*(t/trange) ) + noise
itaus: np.ndarray = np.searchsorted( t, taus )
yt: np.ndarray = np.stack( [ys,t], axis=1 )
ywt: np.ndarray = np.stack( [ yt[itaus[i]-rwin:itaus[i]+rwin+1,:] for i in range(itaus.shape[0]) ] )
yw: np.ndarray = ywt[:,:,0]
tw: np.ndarray = ywt[:,:,1]
dtw: np.ndarray = np.abs(tw-taus[:,None])
alpha = rwin/tstep
w = np.exp( -(alpha*dtw)**2 )

print( f"ys{ys.shape}, t{t.shape} yt{yt.shape} taus{taus.shape} itaus{itaus.shape} ywt{ywt.shape} yw{yw.shape} tw{tw.shape} dtw{dtw.shape}" )
print( f" tstep={tstep:.4f}, 1/tstep={1/tstep:.4f}, dtw range = {dtw.min():.4f} -> {dtw.max():.4f}, w range = {w.min():.5f} -> {w.max():.5f}" )


