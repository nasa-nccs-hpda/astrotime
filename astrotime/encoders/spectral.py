import random, time, numpy as np
import torch, math, copy
from astropy.utils.masked.function_helpers import ones_like
from astrotime.util.math import shp
from typing import List, Tuple, Mapping, Callable
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, device, nn
from astrotime.encoders.embedding import EmbeddingLayer
from astrotime.util.math import l2space
from astrotime.util.logging import elapsed
from astrotime.util.tensor_ops import check_nan

def tnorm(x: Tensor, dim: int=-1) -> Tensor:
	m: Tensor = x.mean( dim=dim, keepdim=True)
	s: Tensor = torch.std( x, dim=dim, keepdim=True)
	return (x - m) / s

def embedding_space( cfg: DictConfig, device: device ) -> Tuple[np.ndarray,Tensor]:
	nfspace = l2space( cfg.base_freq, cfg.noctaves, cfg.nfreq_oct )
	tfspace = torch.FloatTensor( nfspace ).to(device)
	return nfspace, tfspace

def spectral_projection( x: Tensor, y: Tensor ) -> Tensor:
	yn: Tensor = tnorm(y)
	pw1: Tensor = torch.sin(x)
	pw2: Tensor = torch.cos(x)
	p1: Tensor = torch.sum( yn * pw1, dim=-1)
	p2: Tensor = torch.sum( yn * pw2, dim=-1)
	mag: Tensor =  torch.sqrt( p1**2 + p2**2 )
	return tnorm(mag)

class SpectralProjection(EmbeddingLayer):

	def __init__(self, cfg, embedding_space: Tensor, device: device):
		EmbeddingLayer.__init__( self, 'spectral_projection', cfg, embedding_space, device )
		self.init_log(f"SpectralProjection: nfreq={self.nfreq} ")
		self.subbatch_size: int = cfg.get('subbatch_size',-1)
		self.noctaves: int = self.cfg.noctaves
		self.nfreq_oct: int = self.cfg.nfreq_oct
		self.fold_octaves = self.cfg.fold_octaves

	@property
	def xdata(self) -> Tensor:
		return self._embedding_space

	@property
	def output_series_length(self) -> int:
		return self.nfreq_oct if self.fold_octaves else self.nf

	def sbatch(self, ts: torch.Tensor, ys: torch.Tensor, subbatch: int) -> tuple[Tensor,Tensor]:
		sbr = [ subbatch*self.subbatch_size, (subbatch+1)*self.subbatch_size ]
		return ts[sbr[0]:sbr[1]], ys[sbr[0]:sbr[1]]

	def embed(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs) -> Tensor:
		if ys.ndim == 1:
			result = self.embed_subbatch( ts[None,:], ys[None,:] )
		elif self.subbatch_size <= 0:
			result = self.embed_subbatch( ts, ys )
		else:
			nsubbatches = math.ceil(ys.shape[0]/self.subbatch_size)
			subbatches = [ self.embed_subbatch( *self.sbatch(ts,ys,i), **kwargs ) for i in range(nsubbatches) ]
			result = torch.concat( subbatches, dim=0 )
		return result

	def embed_subbatch(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs ) -> Tensor:
		t0 = time.time()
		self.init_log(f"WaveletAnalysisLayer shapes: ts{list(ts.shape)} ys{list(ys.shape)}")
		omega = self._embedding_space * 2.0 * math.pi
		omega_: Tensor = omega[None, :, None]  # broadcast-to(self.batch_size,self.nfreq,slen)
		ts: Tensor = ts[:, None, :]  # broadcast-to(self.batch_size,self.nfreq,slen)
		dz: Tensor = omega_ * ts
		mag: Tensor =  spectral_projection( dz, ys )
		embedding: Tensor = mag.reshape( [mag.shape[0], self.noctaves, self.nfreq_oct] ) if self.fold_octaves else torch.unsqueeze(mag, 1)
		self.init_log(f" Completed embedding{list(embedding.shape)} in {elapsed(t0):.5f} sec: nfeatures={embedding.shape[1]}")
		self.init_state = False
		return embedding

	@property
	def nf(self):
		return self.noctaves * self.nfreq_oct

	@property
	def nfeatures(self):
		return 1

	def magnitude(self, embedding: Tensor) -> np.ndarray:
		self.init_log(f" -> Embedding magnitude{embedding.shape}")
		return embedding.cpu().numpy()
