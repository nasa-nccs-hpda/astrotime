from astrotime.encoders.expansion import Expansion
import random, time, torch, numpy as np
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor, device
from .embedding import EmbeddingLayer
from omegaconf import DictConfig, OmegaConf
from numpy.polynomial.polynomial import Polynomial
from astrotime.util.math import shp
import logging
log = logging.getLogger()

class PolyEmbeddingLayer(EmbeddingLayer):

	def __init__(self, cfg, device: device):
		EmbeddingLayer.__init__(self,cfg,device)

	def embed(self, ts: torch.Tensor, ys: torch.Tensor) -> Tensor:
		print(f"     MODEL INPUT T: ts{list(ts.shape)}: ({ts.min().item():.2f}, {ts.max().item():.2f}, {ts.mean().item():.2f}, {ts.std().item():.2f}) ")
		print(f"     MODEL INPUT Y: ys{list(ys.shape)}: ({ys.min().item():.4f}, {ys.max().item():.4f}, {ys.mean().item():.4f}, {ys.std().item():.4f}) ")
		return ys

	@property
	def nfeatures(self) -> int:
		return 1


class PolyExpansion(Expansion):

	def __init__(self, cfg: DictConfig, device: device):
		super(PolyExpansion, self).__init__(cfg, device)
		self.degree = cfg.degree

	@property
	def nfeatures(self):
		return self.degree + 1

	def get_expansion_coeff(self, x: np.ndarray, y: np.ndarray ) -> Tuple[np.ndarray,np.ndarray]:
		# print( f"PolyExpansion input: x{shp(x)} y{shp(y)}, x: ({x[0]:.4f},{x[-1]:.4f}): dx={(x[-1]-x[0])/x.shape[0]:.4f}")
		coeffs, xs = [], []
		dr = self._xstride*self.cfg.domain_scale/2
		for ipt in range(0,int(self.output_series_length)):
			x0 = x[0] + (ipt+0.5)*self._xstride
			domain = [x0-dr,x0+dr]
			mask = np.abs(x-x0) < dr
			#print(f" ** x-mask{shp(x[mask])}, y-mask{shp(y[mask])}, x0={x0:.4f}, dr={dr:.4f}")
			poly: Polynomial = Polynomial.fit( x[mask], y[mask], self.degree, domain )
			coeffs.append( poly.coef )

			xs.append( x0 )
		X,C = np.array(xs), np.concatenate(coeffs)
		#print( f"PolyExpansion output: X{shp(X)} C{shp(C)}")
		return X, C