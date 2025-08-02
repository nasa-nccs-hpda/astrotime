from typing import List, Tuple, Mapping, Optional
import logging, torch, numpy as np, math, time
from omegaconf import DictConfig
from torch import Tensor
from astrotime.util.math import l2space

class Transform(torch.nn.Module):

	def __init__(self, name: str, cfg: DictConfig):
		torch.nn.Module.__init__(self)
		self.name = name
		self.requires_grad_(False)
		self.cfg: DictConfig = cfg

	def process_event(self, **kwargs ):
		pass

	@property
	def xdata(self, **kwargs ) -> np.ndarray:
		raise NotImplementedError("Transform.xdata not implemented")

	def embed(self, xs: Tensor, ys: Tensor, **kwargs) -> Tensor:
		raise NotImplementedError("Transform.embed() not implemented")

	def magnitude(self, embedding: Tensor) -> np.ndarray:
		raise NotImplementedError("Transform.magnitude() not implemented")


class EmbeddingLayer(Transform):

	def __init__(self, name: str, cfg, embedding_space: Tensor ):
		Transform.__init__(self, name, cfg )
		self.nfreq: int = embedding_space.shape[0]
		self.batch_size: int = cfg.batch_size
		self._embedding_space: Tensor = embedding_space
		self._result: torch.Tensor = None
		self._octaves: torch.Tensor = None

	@property
	def output_channels(self):
		return 1

	def set_octave_data(self, octaves: torch.Tensor):
		self._octaves = octaves

	def get_octave_data(self) -> torch.Tensor:
		return self._octaves

	def forward(self, batch: torch.Tensor ) -> torch.Tensor:
		xs: torch.Tensor = batch[:, 0, :]
		ys: torch.Tensor = batch[:, 1:, :]
		self._result: torch.Tensor = self.embed(xs,ys)
		return self._result

	def get_result(self) -> np.ndarray:
		return self._result.cpu().numpy()

	def get_result_tensor(self) -> torch.Tensor:
		return self._result

	def get_target_freq( self, target_period: float ) -> float:
		return 1/target_period

	def embed(self, xs: Tensor, ys: Tensor, **kwargs) -> Tensor:
		raise NotImplementedError("EmbeddingLayer.embed() not implemented")

	def magnitude(self, embedding: Tensor) -> np.ndarray:
		raise NotImplementedError("EmbeddingLayer.embed() not implemented")

	@property
	def xdata(self) -> Tensor:
		return self._embedding_space

	@property
	def projection_dim(self) -> int:
		raise NotImplementedError("EmbeddingLayer.projection_dim not implemented")

	@property
	def output_series_length(self) -> int:
		return self.nfreq

	@property
	def nfeatures(self) -> int:
		return 1

def tnorm(x: Tensor, dim: int=-1) -> Tensor:
	m: Tensor = x.mean( dim=dim, keepdim=True)
	s: Tensor = torch.std( x, dim=dim, keepdim=True)
	return (x - m) / (s + 0.0001)

def embedding_space( cfg: DictConfig ) -> Tuple[np.ndarray,Tensor]:
	nfspace = l2space( cfg.base_freq, cfg.noctaves, cfg.nfreq_oct )
	tfspace = torch.FloatTensor( nfspace )
	return nfspace, tfspace


class SpectralProjection(EmbeddingLayer):

	def __init__(self, cfg, embedding_space: Tensor):
		EmbeddingLayer.__init__( self, 'spectral_projection', cfg, embedding_space )
		self.init_log(f"SpectralProjection: nfreq={self.nfreq} ")
		self.subbatch_size: int = cfg.get('subbatch_size',-1)
		self.noctaves: int = self.cfg.noctaves
		self.nfreq_oct: int = self.cfg.nfreq_oct
		self.fold_octaves = self.cfg.fold_octaves
		self.f0 = self.cfg.base_freq
		self.device = None
		self.focused_octaves = self.cfg.get('focused_octaves',self.noctaves)
		self.expspace: Tensor = torch.pow(2.0, torch.tensor(range(self.focused_octaves * self.nfreq_oct)) / self.nfreq_oct )

	def set_device(self, device):
		self.device = device

	def spectral_projection(self, x: Tensor, y: Tensor) -> Tensor:
		yn: Tensor = tnorm(y)
		pw1: Tensor = torch.sin(x).to( self.device, non_blocking=True )
		pw2: Tensor = torch.cos(x).to( self.device, non_blocking=True )
		p1: Tensor = torch.sum(yn * pw1, dim=-1)
		p2: Tensor = torch.sum(yn * pw2, dim=-1)
		mag: Tensor = torch.sqrt(p1 ** 2 + p2 ** 2)
		rv = tnorm(mag)
		return rv

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
			result = self.embed_subbatch( ts[None,:], ys[None,:], self._octaves )
		elif self.subbatch_size <= 0:
			result = self.embed_subbatch( ts, ys, self._octaves  )
		else:
			nsubbatches = math.ceil(ys.shape[0]/self.subbatch_size)
			subbatches = [ self.embed_subbatch( *self.sbatch(ts,ys,i), **kwargs ) for i in range(nsubbatches) ]
			result = torch.concat( subbatches, dim=0 )
		embedding =  torch.unsqueeze(result, 1) if result.ndim == 2 else result
		return embedding

	def get_omega(self, octaves:torch.Tensor=None ):
		if octaves is None:
			omega = self._embedding_space * 2.0 * math.pi
			omg = omega[None, :, None] # broadcast-to(self.batch_size,self.nfreq,slen)
		else:
			base_f: torch.Tensor = self.f0 * torch.pow(2, octaves)
			omg = base_f[:,None,None] * self.expspace[None,:,None]
		return omg.to( self.device, non_blocking=True )

	def embed_subbatch(self, ts: torch.Tensor, ys: torch.Tensor,  octaves:torch.Tensor=None, **kwargs ) -> Tensor:
		ts: Tensor = ts[:, None, :]  # broadcast-to(self.batch_size,self.nfreq,slen)
		om: Tensor = self.get_omega(octaves)
		dz: Tensor =  ts * om
		mag: Tensor =  self.spectral_projection( dz, ys )
		embedding: Tensor = mag.reshape( [mag.shape[0], self.focused_octaves, self.nfreq_oct] ) if self.fold_octaves else torch.unsqueeze(mag, 1)
		return embedding

	@property
	def nf(self):
		return self.noctaves * self.nfreq_oct

	@property
	def nfeatures(self):
		return 1

	def magnitude(self, embedding: Tensor) -> np.ndarray:
		return embedding.cpu().numpy()




