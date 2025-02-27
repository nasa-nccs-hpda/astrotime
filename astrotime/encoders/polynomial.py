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

	def get_expansion_coeff(self, x: np.ndarray, y: np.ndarray ) -> np.ndarray:
		poly: Polynomial = Polynomial.fit(x1,y1,self.degree,domain)
		return poly.coef