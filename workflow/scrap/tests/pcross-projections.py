from astrotime.config.context import astrotime_initialize
from typing import List, Optional, Dict, Type, Union, Tuple
import matplotlib.pyplot as plt
import torch, numpy as np, math
from hydra import initialize, compose
from astrotime.util.math import logspace, log2space
from astrotime.encoders.wavelet import wavelet_analysis_projection, embedding_space

def rnorm(ydata: np.ndarray) -> np.ndarray:
	y0,y1 = ydata.min(), ydata.max()
	return 2*(ydata-y0)/(y1-y0) - 1

def znorm(ydata: np.ndarray) -> np.ndarray:
	y0,y1 = ydata.min(), ydata.max()
	return (ydata-y0)/(y1-y0)

def l2norm(ydata: np.ndarray) -> np.ndarray:
	m,s = ydata.mean(), ydata.std()
	return (ydata-m)/s

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
noise_std: float = 0.02
sparsity = 0.5
pts_per_period = 1000
nperiods: int = 15

nts: int = nperiods*pts_per_period
trange: float = tbounds[1] - tbounds[0]
tau_period = trange/nperiods
t: np.ndarray = np.linspace( tbounds[0], tbounds[1], nts )
tstep: float = trange / nts
taus: np.ndarray = np.arange( tbounds[0]+tau_period/2, tbounds[1], tau_period )
tt: np.ndarray = t.reshape(-1,pts_per_period)

sharpness = 30.0
psize = 0.3
dtau: np.ndarray = tt - taus[:,None]
pcross: np.ndarray = 1.0 - psize*np.exp( -(sharpness*dtau/tau_period)**2 ).flatten()
noise: np.ndarray = np.random.normal(0.0, noise_std, nts )
xp, yp  = downsample( t, pcross + noise, 0.5 )

##fspace, tspace = embedding_space( cfg.transform, device )
#pspace: np.ndarray = 1/fspace

fspace = log2space( 0.05, 1.2, 1000 )
pspace = 1/fspace
analysis_projection = znorm( wavelet_analysis_projection( xp, l2norm(yp), fspace, cfg.transform, device).squeeze() )

figure, axes = plt.subplots( 2, 2, figsize=figsize )
plot1 = axes[0,0].plot( xp, yp, label='pcross', color='blue', marker=".", linewidth=1, markersize=2, alpha=0.5)[0]
axes[0,0].set_ylim(0.0,1.1)
axes[0,0].legend(loc="lower right")
plot5 = axes[1,0].plot( pspace, analysis_projection, label='signal-analysis', color='green', linestyle="-", marker=".", linewidth=1, markersize=2, alpha=1.0)[0]
axes[1,0].legend(loc="upper right")
axes[1,0].set_xscale('log')
plt.show()



