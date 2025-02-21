'''
        Tensorflow implementations of the Weighted Wavelet Z-transform (WWZ) for irregularly spaced points.  Based on:
		- Foster, G. Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal 112, 1709 (1996).
		- Witt, A. & Schumann, A. Y. Holocene climate variability on millennial scales recorded in Greenland ice cores. Nonlinear Processes in Geophysics 12, 345–352 (2005).

		Code adapted from Pyleoclim (https://github.com/LinkedEarth/Pyleoclim_util.git)
'''

import time, math
import  warnings, torch
from torch import Tensor
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Hashable
from astrotime.util.logging import lgm

C0 = 1 / (8 * math.pi ** 2)

def wwz(ys: Tensor, ts: Tensor, freq: Tensor, tau: Tensor, device: torch.device, c: float = C0) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor, Tensor]]:
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

    nb: int  = ts.shape[0]
    nts: int = ts.shape[1]
    nf: int  = freq.shape[0]
    if tau is None: tau = 0.5 *(ts[:,nts/2] + ts[:,nts/2+1])
    lgm().debug(f"wwz-0: nb={nb} nts={nts} nf={nf}")

    tau: Tensor   = tau[:,None,None]
    omega = freq * 2.0 * math.pi
    omega_: Tensor =  omega[None,:,None]         # broadcast-to(nb,nf,nts)
    ts: Tensor    = ts[:,None,:]                               # broadcast-to(nb,nf,nts)
    ys: Tensor    = ys[:,None,:]                                      # broadcast-to(nb,nf,nts)
    lgm().debug( f"wwz(nb,nf,nts): ({nb},{nf},{nts}) ys{list(ys.shape)} ts{list(ts.shape)} freq{list(freq.shape)} tau{list(tau.shape)} omega_{list(omega.shape)} omega_{list(omega_.shape)} c={c}" )
    dt: Tensor = (ts - tau)
    dz: Tensor = omega_ * dt
    weights: Tensor= torch.exp(-c * dz ** 2)
    sum_w: Tensor = torch.sum(weights, dim=-1)
    lgm().debug( f"wwz-1: dt{list(dt.shape)} dz{list(dz.shape)} weights{list(weights.shape)} sum_w = {list(sum_w.shape)}")

    def w_prod(xs: Tensor, ys: Tensor) ->  Tensor:
        return torch.sum(weights * xs * ys, dim=-1) / sum_w

    theta: Tensor = omega_ * ts
    sin_basis: Tensor = torch.sin(theta)
    cos_basis: Tensor = torch.cos(theta)
    one_v: Tensor  = torch.ones( (nb,nf,nts) ).to(device)

    sin_one: Tensor  = w_prod(sin_basis, one_v)
    cos_one: Tensor  = w_prod(cos_basis, one_v)
    sin_cos: Tensor  = w_prod(sin_basis, cos_basis)
    sin_sin: Tensor  = w_prod(sin_basis, sin_basis)
    cos_cos: Tensor  = w_prod(cos_basis, cos_basis)
    lgm().debug( f"wwz-2: sin_basis{list(sin_basis.shape)} one_v{list(one_v.shape)} sin_one{list(sin_one.shape)}  sin_sin{list(sin_sin.shape)} cos_cos{list(cos_cos.shape)}")

    numerator: Tensor  = 2 * (sin_cos - sin_one * cos_one)
    denominator: Tensor  = (cos_cos - cos_one ** 2) - (sin_sin - sin_one ** 2)
    time_shift: Tensor  = torch.atan2(numerator, denominator) / (2 * omega)  # Eq. (S5)
    time_shift_: Tensor  = time_shift[:,None]  #  broadcast-to(nb,nf,nts)
    lgm().debug(f"wwz-3:  numerator{list(numerator.shape)} denominator{list(denominator.shape)}  time_shift{list(time_shift.shape)}  ")

    sin_shift: Tensor = torch.sin(omega_ * (ts - time_shift_))
    cos_shift: Tensor = torch.cos(omega_ * (ts - time_shift_))
    sin_tau_center: Tensor = torch.sin(omega * (time_shift - tau[:,:,0]))
    cos_tau_center: Tensor = torch.cos(omega * (time_shift - tau[:,:,0]))
    lgm().debug(f"wwz-4: sin_shift{list(sin_shift.shape)} cos_shift{list(cos_shift.shape)} sin_tau_center{list(sin_tau_center.shape)} cos_tau_center{list(cos_tau_center.shape)}")

    ys_cos_shift: Tensor = w_prod(ys, cos_shift)
    ys_sin_shift: Tensor = w_prod(ys, sin_shift)
    ys_one: Tensor = w_prod(ys, one_v)
    cos_shift_one: Tensor = w_prod(cos_shift, one_v)
    sin_shift_one: Tensor = w_prod(sin_shift, one_v)
    lgm().debug(f"wwz-5: ys_cos_shift{list(ys_cos_shift.shape)} cos_shift_one{list(cos_shift_one.shape)} ys_one{list(ys_one.shape)}")

    A: Tensor = 2 * (ys_cos_shift - ys_one * cos_shift_one)
    B: Tensor = 2 * (ys_sin_shift - ys_one * sin_shift_one)

    lgm().debug(f"wwz-6:  A{list(A.shape)} B{list(B.shape)}")
    a0: Tensor = ys_one
    a1: Tensor = cos_tau_center * A - sin_tau_center * B   # Eq. (S6)
    a2: Tensor = sin_tau_center * A + cos_tau_center * B   # Eq. (S7)

    lgm().debug(f"wwz-7: a0{list(a0.shape)} a1{list(a1.shape)} a2{list(a2.shape)}")
    wwp: Tensor = a1**2 + a2**2
    phase: Tensor = torch.atan2(a2, a1)
    coeff: Tuple[Tensor,Tensor,Tensor] = (a0, a1, a2)
    return wwp, phase, coeff
