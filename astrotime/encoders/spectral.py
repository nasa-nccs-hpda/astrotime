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
	check_nan(f'yn', yn)
	pw1: Tensor = torch.sin(x)
	pw2: Tensor = torch.cos(x)
	p1: Tensor = torch.sum( yn * pw1, dim=-1)
	p2: Tensor = torch.sum( yn * pw2, dim=-1)
	mag: Tensor =  torch.sqrt( p1**2 + p2**2 )
	check_nan(f'mag', mag)
	rv = tnorm(mag)
	for i in range(mag.shape[0]):
		print( f" mag-{i}[{mag.shape[1]}] std={torch.std( mag[i] )}" )
	check_nan(f'rv', rv)
	return rv

class SpectralProjection(EmbeddingLayer):

	def __init__(self, cfg, embedding_space: Tensor, device: device):
		EmbeddingLayer.__init__( self, 'spectral_projection', cfg, embedding_space, device )
		self.init_log(f"SpectralProjection: nfreq={self.nfreq} ")
		self.subbatch_size: int = cfg.get('subbatch_size',-1)
		self.noctaves: int = self.cfg.noctaves
		self.nfreq_oct: int = self.cfg.nfreq_oct
		self.fold_octaves = self.cfg.fold_octaves
		self.f0 = self.cfg.base_freq
		self.focused_octaves = self.cfg.get('focused_octaves',self.noctaves)
		self.expspace: Tensor = torch.pow(2.0, torch.tensor(range(self.focused_octaves * self.nfreq_oct)).to(self.device) / self.nfreq_oct )

	@property
	def output_channels(self):
		return self.focused_octaves if self.fold_octaves else 1

	@property
	def xdata(self) -> Tensor:
		return self._embedding_space

	@property
	def output_series_length(self) -> int:
		return self.nfreq_oct if self.fold_octaves else self.nf

	def sbatch(self, ts: torch.Tensor, ys: torch.Tensor, subbatch: int) -> tuple[Tensor,Tensor,Tensor]:
		sbr = [ subbatch*self.subbatch_size, (subbatch+1)*self.subbatch_size ]
		octaves = None if self._octaves is None else self._octaves[sbr[0]:sbr[1]]
		return ts[sbr[0]:sbr[1]], ys[sbr[0]:sbr[1]], octaves

	def embed(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs) -> Tensor:
		if ys.ndim == 1:
			result = self.embed_subbatch( 0, ts[None,:], ys[None,:], self._octaves )
		elif self.subbatch_size <= 0:
			result = self.embed_subbatch( 0, ts, ys, self._octaves  )
		else:
			nsubbatches = math.ceil(ys.shape[0]/self.subbatch_size)
			subbatches = [ self.embed_subbatch(i, *self.sbatch(ts,ys,i), **kwargs ) for i in range(nsubbatches) ]
			result = torch.concat( subbatches, dim=0 )
			# print(f" embedding{list(result.shape)}: ({result.min():.3f} -> {result.max():.3f})")
		embedding =  torch.unsqueeze(result, 1) if result.ndim == 2 else result
		return embedding

	def get_omega(self, octaves:torch.Tensor=None ):
		if octaves is None:
			omega = self._embedding_space * 2.0 * math.pi
			return omega[None, :, None] # broadcast-to(self.batch_size,self.nfreq,slen)
		else:
			base_f: torch.Tensor = self.f0 * torch.pow(2, octaves)
			omg = base_f[:,None,None] * self.expspace[None,:,None]
			return omg

	def embed_subbatch(self, ibatch: int, ts: torch.Tensor, ys: torch.Tensor,  octaves:torch.Tensor=None, **kwargs ) -> Tensor:
		t0 = time.time()
		check_nan(f'ts-sb{ibatch}', ts)
		check_nan(f'ys-sb{ibatch}', ys)
		self.init_log(f"SpectralProjection shapes: ts{list(ts.shape)} ys{list(ys.shape)}")
		ts: Tensor = ts[:, None, :]  # broadcast-to(self.batch_size,self.nfreq,slen)
		dz: Tensor =  ts * self.get_omega(octaves)
		check_nan(f'dz-sb{ibatch}', dz)
		mag: Tensor =  spectral_projection( dz, ys )
		embedding: Tensor = mag.reshape( [mag.shape[0], self.focused_octaves, self.nfreq_oct] ) if self.fold_octaves else torch.unsqueeze(mag, 1)
		self.init_log(f" Completed embedding{list(embedding.shape)} in {elapsed(t0):.5f} sec: nfeatures={embedding.shape[1]}")
		self.init_state = False
		check_nan(f'embed_subbatch-{ibatch}', embedding)
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
