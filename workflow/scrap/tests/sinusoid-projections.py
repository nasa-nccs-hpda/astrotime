
from astrotime.config.context import astrotime_initialize
from typing import List, Optional, Dict, Type, Union, Tuple
import matplotlib.pyplot as plt
import torch, numpy as np, math
from hydra import initialize, compose
from scipy.signal import fftconvolve
from astrotime.encoders.wavelet import wavelet_analysis_projection

def rnorm(ydata: np.ndarray) -> np.ndarray:
	y0,y1 = ydata.min(), ydata.max()
	return 2*(ydata-y0)/(y1-y0) - 1

def znorm(ydata: np.ndarray) -> np.ndarray:
	y0,y1 = ydata.min(), ydata.max()
	return (ydata-y0)/(y1-y0)

def downsample( x: np.ndarray, y: np.ndarray, sparsity: float, dim:int=0 ) -> Tuple[np.ndarray, np.ndarray]:
	mask = np.random.rand(x.shape[dim]) > sparsity
	return np.compress(mask, x, dim), np.compress(mask, y, dim)

version = "desktop_period.analysis"
overrides = []
initialize(version_base=None, config_path="../../../config")
cfg = compose(config_name=version, overrides=overrides)
device: torch.device = astrotime_initialize(cfg, version)

figsize = (18, 9)
pi2: float = 2*np.pi
tbounds: Tuple[float,float] = ( 300.0, 380.0 )
noise_std: float = 0.2
nts: int = 10000
ntaus: int = 100
ntper: int = 5
rwin: int = 8
alpha = 0.6 # 0.8
sparsity = 0.5
sfactor = 5

trange: float = tbounds[1] - tbounds[0]
tstep: float = trange / nts
twbnds = ( tbounds[0]+rwin*sfactor*tstep, tbounds[1]-rwin*sfactor*tstep )
noise: np.ndarray = np.random.normal(0.0, noise_std, nts )
t: np.ndarray = np.linspace( tbounds[0], tbounds[1], nts )
taus: np.ndarray = np.linspace( twbnds[0], twbnds[1], ntaus )
dtau = (twbnds[1]-twbnds[0])/ntaus
ctaus = taus - (taus.max()+taus.min())/2
ys: np.ndarray = np.sin( pi2*ntper*(t/trange) ) + noise

t, ys = downsample(t, ys, 0.5)
itaus: np.ndarray = np.searchsorted(t, taus)
yt: np.ndarray = np.stack([ys, t], axis=1)
ywt: np.ndarray = np.stack( [ yt[itaus[i]-rwin:itaus[i]+rwin+1,:] for i in range(itaus.shape[0]) ] )

yw: np.ndarray = ywt[:,:,0]
tw: np.ndarray = ywt[:,:,1]
dtw: np.ndarray = np.abs(tw-taus[:,None])
w = np.exp( -((alpha*dtw)/(rwin*tstep))**2 )
yss = (yw*w).sum(axis=1) /  w.sum(axis=1)
ysscf = rnorm( fftconvolve( yss, yss, mode='same'))

yssfft = znorm( np.absolute( np.fft.rfft(yss) ) )
print( f"yss.size={yss.size}, dtau={dtau}")
yfftfreq = np.fft.rfftfreq(yss.size,dtau)
yfftper = 1 / yfftfreq

analysis_projection = znorm( wavelet_analysis_projection(t,ys,yfftfreq,cfg.transform,device).squeeze() )

# ts: np.ndarray, ys: np.ndarray, fspace: np.ndarray, cfg: DictConfig, device

print( f"ys{ys.shape}, t{t.shape} yt{yt.shape} taus{taus.shape} itaus{itaus.shape} ywt{ywt.shape} yw{yw.shape} tw{tw.shape} dtw{dtw.shape}" )
print( f" tstep={tstep:.4f}, 1/tstep={1/tstep:.4f}, dtw{dtw.shape} range = {dtw.min():.4f} -> {dtw.max():.4f}, w{w.shape} range = {w.min():.5f} -> {w.max():.5f}" )
print( f" yss{yss.shape} range = {yss.min():.5f} -> {yss.max():.5f}" )

figure, axes = plt.subplots( 2, 2, figsize=figsize )
plot1 = axes[0,0].plot(taus, yss, label='interp', color='red', marker=".", linewidth=3, markersize=6, alpha=1.0)[0]
plot2 = axes[0,0].plot(t, ys, label='signal', color='blue', marker=".", linewidth=1, markersize=2, alpha=0.4)[0]
axes[0,0].legend(loc="upper right")
plot3 = axes[0,1].plot(ctaus, ysscf, label='interp-correlation', color='red', marker=".", linewidth=1, markersize=2, alpha=1.0)[0]
axes[0,1].legend(loc="upper right")
plot4 = axes[1,0].plot(yfftper, yssfft, label='interp-fft', color='green', linestyle="--", marker=".", linewidth=2, markersize=4, alpha=1.0)[0]
plot5 = axes[1,0].plot(yfftper, analysis_projection, label='signal-analysis', color='red', linestyle=":", marker="o", linewidth=2, markersize=4, alpha=1.0)[0]
axes[1,0].legend(loc="upper right")

plt.show()



