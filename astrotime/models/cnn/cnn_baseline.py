from torch import nn
import torch, math
from omegaconf import DictConfig, OmegaConf
from astrotime.encoders.embedding import EmbeddingLayer
from typing import Any, Dict, List, Optional, Tuple, Mapping
from astrotime.models.spectral.peak_finder import SpectralPeakSelector

def harmonic( y: float, t: float) -> float:
	if y > t: return round(y / t)
	else:     return 1 / round(t / y)

class ExpU(nn.Module):

	def __init__(self, cfg: DictConfig) -> None:
		super().__init__()
		self.f0: float = cfg.base_freq

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		result = self.f0 * ( torch.pow(2,x) - 1 )
		return result

class ExpLoss(nn.Module):
	def __init__(self, cfg: DictConfig):
		super(ExpLoss, self).__init__()
		self.f0: float = cfg.base_freq

	def forward(self, product: torch.Tensor, target: torch.Tensor)-> torch.Tensor:
		result = torch.abs( torch.log2( (product+self.f0)/(target+self.f0) ) ).mean()
		return result

class ExpHLoss(nn.Module):
	def __init__(self, cfg: DictConfig):
		super(ExpHLoss, self).__init__()
		self.f0: float = cfg.base_freq

	def forward(self, product: torch.Tensor, target: torch.Tensor)-> torch.Tensor:
		h: float = harmonic( product.item(), target.item() )
		result = torch.abs( torch.log2( (product+self.f0)/(h*target+self.f0) ) ).mean()
		return result

def add_cnn_block( cfg: DictConfig, model: nn.Sequential, nchannels: int, num_input_features: int ) -> int:
	block_input_channels = num_input_features if (num_input_features > 0) else nchannels
	in_channels = block_input_channels
	out_channels = nchannels

	for iL in range( cfg.num_cnn_layers ):
		out_channels = out_channels + cfg.cnn_expansion_factor
		model.append( nn.Conv1d( in_channels, out_channels, kernel_size=cfg.kernel_size, stride=cfg.stride, padding='same') )
		model.append(nn.ELU())
		in_channels = out_channels
	model.append(nn.ELU())
	model.append( nn.BatchNorm1d(out_channels) )
	model.append( nn.MaxPool1d(cfg.pool_size) )
	print(f"CNN: add_cnn_block: in_channels={block_input_channels}, out_channels={out_channels}")
	return out_channels

def add_dense_block( cfg: DictConfig, model: nn.Sequential, in_channels:int ):
	print(f"CNN: add_dense_block: in_channels={in_channels}, hidden_channels={cfg.dense_channels}, out_channels={cfg.out_channels}")
	model.append( nn.Flatten() )
	model.append( nn.Linear( in_channels, cfg.dense_channels ) )  # 64
	model.append( nn.ELU() )
	model.append( nn.Linear( cfg.dense_channels, cfg.out_channels ) )


def get_model_from_cfg( cfg: DictConfig, device: torch.device, embedding_layer: EmbeddingLayer, scale: nn.Module = None  ) -> nn.Module:
	model: nn.Sequential = nn.Sequential( embedding_layer )
	cnn_channels = cfg.cnn_channels
	num_input_features = embedding_layer.nfeatures
	for iblock in range(cfg.num_blocks):
		cnn_channels = add_cnn_block( cfg, model, cnn_channels, num_input_features )
		num_input_features = -1
	reduced_series_len = embedding_layer.output_series_length // int( math.pow(cfg.pool_size, cfg.num_blocks) )
	print(f"CNN: reduced_series_len={reduced_series_len}, cnn_channels={cnn_channels}, output_series_length={embedding_layer.output_series_length}")
	add_dense_block( cfg, model, cnn_channels*reduced_series_len  )
	if scale is not None: model.append(scale)
	return model.to(device)

def get_spectral_peak_selector_from_cfg( cfg: DictConfig, device: torch.device, embedding_layer: EmbeddingLayer  ) -> nn.Module:
	model: nn.Sequential = nn.Sequential( embedding_layer )
	model.append( SpectralPeakSelector( cfg, device, embedding_layer.xdata ) )
	return model.to(device)
