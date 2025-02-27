import torch
from torch import Tensor, device
import logging
log = logging.getLogger("astrotime")

class EmbeddingLayer(torch.nn.Module):

	def __init__(self, cfg, device: device):
		torch.nn.Module.__init__(self)
		self.requires_grad_(False)
		self.device = device
		self.cfg = cfg
		self.series_length = cfg.series_length
		self.batch_size = cfg.batch_size
		self.nfeatures = cfg.nfeatures
		log.info(f"EmbeddingLayer: series_length={self.series_length} batch_size={self.batch_size} ")

	def forward(self, input: torch.Tensor ):
		log.debug(f"WaveletEmbeddingLayer shapes:")
		ys: torch.Tensor = input[:, 1:, :]
		xs: torch.Tensor = input[:, 0, :]
		return self.embed(xs,ys)

	def embed(self, xs: Tensor, ys: Tensor ) -> Tensor:
		raise NotImplementedError("EmbeddingLayer.embed() not implemented")

