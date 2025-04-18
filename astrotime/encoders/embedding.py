import torch
from torch import Tensor, device
import numpy as np
import logging

class EmbeddingLayer(torch.nn.Module):

	def __init__(self, cfg, embedding_space: Tensor, device: device):
		torch.nn.Module.__init__(self)
		self.requires_grad_(False)
		self.device = device
		self.cfg = cfg
		self.series_length = cfg.series_length
		self.nfreq = embedding_space.shape[0]
		self.batch_size = cfg.batch_size
		self.time_scale = self.cfg.time_scale
		self.log = logging.getLogger()
		self._embedding_space: Tensor = embedding_space.to(self.device)
		self.log.info(f"EmbeddingLayer: series_length={self.series_length} batch_size={self.batch_size} ")
		self.init_state = True

	def process_event(self, **kwargs ):
		pass

	def init_log(self, msg: str):
		if self.init_state: self.log.info(msg)

	def forward(self, input: torch.Tensor ) -> torch.Tensor:
		self.log.debug(f"WaveletEmbeddingLayer shapes:")
		xs: torch.Tensor = input[:, 0, :]
		ys: torch.Tensor = input[:, 1:, :]
		result: torch.Tensor = self.embed(xs,ys)
		self.init_state = False
		return result

	def get_target_freq( self, target_period: float ) -> float:
		return 1/target_period

	def embed(self, xs: Tensor, ys: Tensor ) -> Tensor:
		raise NotImplementedError("EmbeddingLayer.embed() not implemented")

	def magnitude(self, embedding: Tensor) -> np.ndarray:
		raise NotImplementedError("EmbeddingLayer.embed() not implemented")

	def xdata(self) -> Tensor:
		return self._embedding_space

	@property
	def projection_dim(self) -> int:
		raise NotImplementedError("EmbeddingLayer.projection_dim not implemented")

	@property
	def output_series_length(self):
		raise NotImplementedError("EmbeddingLayer.output_series_length not implemented")

