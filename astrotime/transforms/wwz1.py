'''
    GPU & multiproc-CPU implementations of the Weighted Wavelet Z-transform (WWZ) for irregularly spaced points.  Based on:
		- Foster, G. Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal 112, 1709 (1996).
		- Witt, A. & Schumann, A. Y. Holocene climate variability on millennial scales recorded in Greenland ice cores. Nonlinear Processes in Geophysics 12, 345–352 (2005).

		Code adapted from Pyleoclim (https://github.com/LinkedEarth/Pyleoclim_util.git)
'''

import time, numpy as np
import torch, warnings
from warnings import catch_warnings, warn
from numpy import sum, pi, cos, sin, arctan2, exp, log, sqrt, dot, arange
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Hashable
from astrotime.util.env import Array, to_torch
import logging
log = logging.getLogger("astrotime")
C0 = 1 / (8 * np.pi ** 2)

def wwz(ys_: Array, ts_: Array, freq_: Array, tau_: Array, c: float = C0, rank: int=0) -> Tuple[Array, Array, Tuple[Array, Array, Array]]:
    '''
        Compute the weighted wavelet amplitude (WWA)
    ---------- Parameters:
    ys   : a time series
    ts   : time axis of the time series
    freq : vector of frequency
    tau  : the evenly-spaced time points, namely the time shift for wavelet analysis
    c    : the decay constant that determines the analytical resolution of frequency for analysis, the smaller the higher resolution;
           the default value 1/(8*np.pi**2) is good for most of the wavelet analysis cases
    ------- Returns:
    wwa   :  the weighted wavelet amplitude
    phase :  the weighted wavelet phase
    coeff :  the wavelet transform coefficients (a0, a1, a2)
    '''
    device: torch.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    ys: torch.Tensor   = to_torch(ys_,device)
    ts: torch.Tensor   = to_torch(ts_,device)
    freq: torch.Tensor = to_torch(freq_,device)
    tau: torch.Tensor  = to_torch(tau_,device)
    nt: int  = tau.shape[0]
    nts: int = ts.shape[0]
    nf: int  = freq.shape[0]
    verbose = False

    omega: torch.Tensor = 2 * np.pi * freq
    omega_ = omega[None,:,None].expand(nt,nf,nts)
    print( f"Transform ts{list(ts.shape)} freq{list(freq.shape)} tau{list(tau.shape)} omega{list(omega.shape)} nt={nt} nf={nf} nts={nts}")
    ts = ts[None,None,:].expand(nt,nf,nts)
    tau = tau[:,None,None].expand(nt,nf,nts)
    if verbose: log.info( f"wwz: ys{list(ys.shape)} ts{list(ts.shape)} freq{list(freq.shape)} tau{list(tau.shape)} omega_{list(omega_.shape)} c={c}" )
    dt = (ts - tau)
    dz = omega_ * dt
    weights = torch.exp(-c * dz ** 2)
    sum_w = torch.sum(weights,dim=-1)
    if verbose: log.info( f"wwz-1: ys{list(ys.shape)} ts{list(ts.shape)} omega{list(omega.shape)} omega_{list(omega_.shape)} dz{list(dz.shape)} weights{list(weights.shape)} sum_w = {list(sum_w.shape)}")

    def w_prod(xs, ys):
        return torch.sum(weights * xs * ys, dim=-1) / sum_w

    theta = omega_ * ts
    sin_basis = torch.sin(theta)
    cos_basis = torch.cos(theta)
    one_v = torch.ones((nt,nf,nts), dtype=torch.float32, device=device)

    sin_one = w_prod(sin_basis, one_v)
    cos_one = w_prod(cos_basis, one_v)
    sin_cos = w_prod(sin_basis, cos_basis)
    sin_sin = w_prod(sin_basis, sin_basis)
    cos_cos = w_prod(cos_basis, cos_basis)
    if verbose: log.info( f"wwz-2: sin_basis{list(sin_basis.shape)} one_v{list(one_v.shape)} sin_one{list(sin_one.shape)}  sin_sin{list(sin_sin.shape)} cos_cos{list(cos_cos.shape)}")

    numerator = 2 * (sin_cos - sin_one * cos_one)
    denominator = (cos_cos - cos_one ** 2) - (sin_sin - sin_one ** 2)
    time_shift = torch.arctan2(numerator, denominator) / (2 * omega)  # Eq. (S5)
    time_shift_ = time_shift[:, :, None].expand(nt, nf, nts)
    if verbose: log.info(f"wwz-3:  numerator{list(numerator.shape)} denominator{list(denominator.shape)}  time_shift{list(time_shift.shape)}  ")

    sin_shift = torch.sin(omega_ * (ts - time_shift_))
    cos_shift = torch.cos(omega_ * (ts - time_shift_))
    sin_tau_center = torch.sin(omega * (time_shift - tau[:,:,0]))
    cos_tau_center = torch.cos(omega * (time_shift - tau[:,:,0]))
    if verbose: log.info(f"wwz-4: sin_shift{list(sin_shift.shape)} cos_shift{list(cos_shift.shape)} sin_tau_center{list(sin_tau_center.shape)} cos_tau_center{list(cos_tau_center.shape)}")

    ys_cos_shift = w_prod(ys, cos_shift)
    ys_sin_shift = w_prod(ys, sin_shift)
    ys_one = w_prod(ys, one_v)
    cos_shift_one = w_prod(cos_shift, one_v)
    sin_shift_one = w_prod(sin_shift, one_v)
    if verbose: log.info(f"wwz-5: ys_cos_shift{list(ys_cos_shift.shape)} cos_shift_one{list(cos_shift_one.shape)} ys_one{list(ys_one.shape)}")

    A = 2 * (ys_cos_shift - ys_one * cos_shift_one)
    B = 2 * (ys_sin_shift - ys_one * sin_shift_one)

    if verbose: log.info(f"wwz-6:  A{list(A.shape)} B{list(B.shape)}")
    a0 = ys_one
    a1 = cos_tau_center * A - sin_tau_center * B   # Eq. (S6)
    a2 = sin_tau_center * A + cos_tau_center * B   # Eq. (S7)

    if verbose: log.info(f"wwz-7: a0{list(a0.shape)} a1{list(a1.shape)} a2{list(a2.shape)}")
    wwp: torch.Tensor = a1**2 + a2**2
    phase: torch.Tensor = torch.arctan2(a2, a1)
    coeff = (a0, a1, a2)
    return wwp, phase, coeff