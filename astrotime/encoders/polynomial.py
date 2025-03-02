from astrotime.encoders.expansion import Expansion
import random, time, torch, numpy as np
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor, device
from omegaconf import DictConfig, OmegaConf
from numpy.polynomial.polynomial import Polynomial
from astrotime.util.math import shp
import logging
log = logging.getLogger("astrotime")


class PolyExpansion(Expansion):

	def __init__(self, cfg: DictConfig, device: device):
		super(Expansion, self).__init__(cfg, device)
		self.degree = cfg.degree

	def get_expansion_coeff(self, x: np.ndarray, y: np.ndarray ) -> Tuple[np.ndarray,np.ndarray]:
		coeffs, xs = [], []
		dr = self.xstride*self.cfg.domain_scale/2
		for ipt in range(1,int(self.nstrides)):
			x0 = x[0] + ipt*self.xstride
			domain = [x0-dr,x0+dr]
			mask = np.abs(x-x0) < dr
			poly: Polynomial = Polynomial.fit( x[mask], y[mask], self.degree, domain )
			coeffs.append( poly.coef )
			xs.append( x0 )
		X,C = np.array(xs), np.concatenate(coeffs)
		print( f"PolyExpansion: X{shp(X)} C{shp(C)}")
		return X,C