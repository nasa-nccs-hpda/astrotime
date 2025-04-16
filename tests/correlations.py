
from astrotime.config.context import astrotime_initialize
from typing import List, Optional, Dict, Type, Union, Tuple
import matplotlib.pyplot as plt
import torch, numpy as np
from hydra import initialize, compose

def znorm(ydata: np.ndarray) -> np.ndarray:
	y0,y1 = ydata.min(), ydata.max()
	return 2*(ydata-y0)/(y1-y0) - 1

def downsample( x: np.ndarray, y: np.ndarray, sparsity: float, dim:int=0 ) -> Tuple[np.ndarray, np.ndarray]:
	mask = np.random.rand(x.shape[dim]) > sparsity
	return np.compress(mask, x, dim), np.compress(mask, y, dim)

version = "test.desktop"
overrides = []
initialize(version_base=None, config_path="../config")
cfg = compose(config_name=version, overrides=overrides)
device: torch.device = astrotime_initialize(cfg, version)

figsize = (16, 8)
pi2: float = 2*np.pi
tbounds: Tuple[float,float] = ( 300.0, 380.0 )
noise_std: float = 0.01
nts: int = 10000
ntaus: int = 100
ntper: int = 5
rwin: int = 8
alpha = 0.6 # 0.8

trange: float = tbounds[1] - tbounds[0]
tstep: float = trange / nts
twbnds = ( tbounds[0]+(rwin+1)*tstep, tbounds[1]-(rwin+1)*tstep )
noise: np.ndarray = np.random.normal(0.0, noise_std, nts )
t: np.ndarray = np.linspace( tbounds[0], tbounds[1], nts )
taus: np.ndarray = np.linspace( twbnds[0], twbnds[1], ntaus )
ys: np.ndarray = np.sin( pi2*ntper*(t/trange) ) + noise
t, ys = downsample( t, ys, 0.5)

itaus: np.ndarray = np.searchsorted( t, taus )
yt: np.ndarray = np.stack( [ys,t], axis=1 )
ywt: np.ndarray = np.stack( [ yt[itaus[i]-rwin:itaus[i]+rwin+1,:] for i in range(itaus.shape[0]) ] )
yw: np.ndarray = ywt[:,:,0]
tw: np.ndarray = ywt[:,:,1]
dtw: np.ndarray = np.abs(tw-taus[:,None])
w = np.exp( -((alpha*dtw)/(rwin*tstep))**2 )
yss = (yw*w).sum(axis=1) /  w.sum(axis=1)
yssc = znorm( np.convolve( yss, yss, mode='same') )

print( f"ys{ys.shape}, t{t.shape} yt{yt.shape} taus{taus.shape} itaus{itaus.shape} ywt{ywt.shape} yw{yw.shape} tw{tw.shape} dtw{dtw.shape}" )
print( f" tstep={tstep:.4f}, 1/tstep={1/tstep:.4f}, dtw{dtw.shape} range = {dtw.min():.4f} -> {dtw.max():.4f}, w{w.shape} range = {w.min():.5f} -> {w.max():.5f}" )
print( f" yss{yss.shape} range = {yss.min():.5f} -> {yss.max():.5f}" )

figure, ax = plt.subplots( 1, 1, figsize=figsize )
plot1 = ax[0,0].plot(taus, yss, label='smoothed', color='red', marker=".", linewidth=3, markersize=6, alpha=1.0)[0]
plot2 = ax[0,0].plot(t, ys, label='signal', color='blue', marker=".", linewidth=1, markersize=2, alpha=0.2)[0]
plot3 = ax[0,1].plot(taus, yssc, label='smoothed-cor', color='red', marker=".", linewidth=2, markersize=4, alpha=1.0)[0]

plt.show()



