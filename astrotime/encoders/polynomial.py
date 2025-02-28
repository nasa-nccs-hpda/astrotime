from astrotime.encoders.expansion import Expansion
import random, time, torch, numpy as np
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor, device
from omegaconf import DictConfig, OmegaConf
from astrotime.util.math import tmean, tstd, tmag, npnorm, shp
from numpy.polynomial.polynomial import Polynomial
import logging
log = logging.getLogger("astrotime")


class PolyExpansion(Expansion):

	def __init__(self, cfg: DictConfig, device: device):
		super(Expansion, self).__init__(cfg, device)
		self.degree = cfg.degree
		self.xstep = None

	def get_expansion_coeff(self, x: np.ndarray, y: np.ndarray ) -> Tuple[np.ndarray,np.ndarray]:
		coeffs, xs = [], []
		for ipt in range(1,npts):
			x0 = xrng[0] + ipt*self.xstep
			domain = [x0-dr,x0+dr]
			mask = np.abs(x-x0) < dr
			poly: Polynomial = Polynomial.fit( x[mask], y[mask], self.degree, domain )
			coeffs.append( poly.coef )
			xs.append( x0 )
		return np.concatenate(xs), np.concatenate(coeffs)