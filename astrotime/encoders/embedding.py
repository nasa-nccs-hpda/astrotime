import torch
from torch import Tensor, device
import numpy as np
import logging
from .base import Transform

class EmbeddingLayer(Transform):

	def __init__(self, name: str, cfg, embedding_space: Tensor, device: device ):
		Transform.__init__(self, name, cfg, device )
		self.nfreq: int = embedding_space.shape[0]
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
		self.log.debug(f"WaveletEmbeddingLayer shapes:")
		xs: torch.Tensor = batch[:, 0, :]
		ys: torch.Tensor = batch[:, 1:, :]
		self._result: torch.Tensor = self.embed(xs,ys)
		self.init_state = False
		return self._result

	def get_result(self) -> np.ndarray:
		return self._result.cpu().numpy()

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

