import torch
from torch import Tensor, device
import numpy as np
import logging
from .base import Transform

def tnorm(x: Tensor, dim: int=-1) -> Tensor:
	m: Tensor = x.mean( dim=dim, keepdim=True)
	s: Tensor = torch.std( x, dim=dim, keepdim=True)
	return (x - m) / (s + 0.0001)

class EmbeddingLayer(Transform):

	def __init__(self, name: str, cfg, embedding_space: Tensor, device: device ):
		Transform.__init__(self, name, cfg, device )
		self.nfreq: int = embedding_space.shape[0]
		self.noctaves: int = self.cfg.noctaves
		self.nfreq_oct: int = self.cfg.nfreq_oct
		self.batch_size: int = cfg.batch_size
		self._embedding_space: Tensor = embedding_space.to(self.device)
		self.init_state: bool = True
		self._result: torch.Tensor = None
		self._octaves: torch.Tensor = None


	@property
	def output_channels(self):
		return 1

	def set_octave_data(self, octaves: torch.Tensor):
		self._octaves = octaves

	def get_octave_data(self) -> torch.Tensor:
		return self._octaves

	def init_log(self, msg: str):
		if self.init_state: self.log.info(msg)

	def forward(self, batch: torch.Tensor ) -> torch.Tensor:
		xs: torch.Tensor = torch.unsqueeze(batch[0, :],0) if batch.ndim == 2 else batch[:, 0, :]
		ys: torch.Tensor = torch.unsqueeze(batch[1, :],0) if batch.ndim == 2 else batch[:, 1:, :]
		yn: Tensor = tnorm(ys)
		self._result: torch.Tensor = self.embed(xs,yn)
		self.init_state = False
		return tnorm(self._result)

	def get_result(self) -> np.ndarray:
		return self._result.cpu().numpy()

	def get_result_tensor(self) -> torch.Tensor:
		return self._result

	def get_target_freq( self, target_period: float ) -> float:
		return 1/target_period

	def embed(self, xs: Tensor, ys: Tensor, **kwargs) -> Tensor:
		raise NotImplementedError("EmbeddingLayer.embed() not implemented")

	def full_embedding(self, xs: Tensor, ys: Tensor ) -> Tensor:
		raise NotImplementedError("EmbeddingLayer.full_embedding() not implemented")

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

