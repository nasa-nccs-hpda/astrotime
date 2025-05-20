from torch import nn
import torch, math
from omegaconf import DictConfig, OmegaConf
from astrotime.encoders.embedding import EmbeddingLayer
from typing import Any, Dict, List, Optional, Tuple, Mapping
import torch.nn.functional as F

class ScaledRelu(nn.Module):
	def __init__(self, cfg: DictConfig) -> None:
		super().__init__()
		self.cfg = cfg

	def forward(self, x):
	#	return self.cfg.base_freq + F.relu(x)
		return  F.relu(x)

class MAELoss(nn.Module):
	def __init__(self, cfg: DictConfig):
		super(MAELoss, self).__init__()
		self.cfg = cfg

	def forward(self, product: torch.Tensor, target: torch.Tensor)-> torch.Tensor:
		return torch.abs( product-target ).mean()

class TScaleLoss(nn.Module):
	def __init__(self, cfg: DictConfig):
		super(TScaleLoss, self).__init__()
		self.cfg = cfg

	def forward(self, product: torch.Tensor, target: torch.Tensor)-> torch.Tensor:
		return torch.abs( (product-target)/target ).mean()

class FScaleLoss(nn.Module):
	def __init__(self, cfg: DictConfig):
		super(FScaleLoss, self).__init__()
		self.cfg = cfg

	def forward(self, product: torch.Tensor, target: torch.Tensor)-> torch.Tensor:
		return torch.abs( torch.log2( product/target) ).mean()

class FScaleLog(nn.Module):
	def __init__(self, cfg: DictConfig):
		super(FScaleLog, self).__init__()
		self.cfg = cfg

	def forward(self, tensor: torch.Tensor) -> torch.Tensor:
		return self.cfg.base_freq * torch.pow(2.0,tensor)

class FScaleLin(nn.Module):
	def __init__(self, cfg: DictConfig):
		super(FScaleLin, self).__init__()
		self.cfg = cfg

	def forward(self, tensor: torch.Tensor) -> torch.Tensor:
		return self.cfg.base_freq + tensor


def add_cnn_block( model: nn.Sequential, nchannels: int, num_input_features: int, cfg: DictConfig ) -> int:
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

def add_dense_block( model: nn.Sequential, in_channels:int, cfg: DictConfig ):
	print(f"CNN: add_dense_block: in_channels={in_channels}, hidden_channels={cfg.dense_channels}, out_channels={cfg.out_channels}")
	model.append( nn.Flatten() )
	model.append( nn.Linear( in_channels, cfg.dense_channels ) )  # 64
	model.append( nn.ELU() )
	model.append( nn.Linear( cfg.dense_channels, cfg.out_channels ) )
	# model.append( ScaledRelu(cfg) )
	model.append( nn.ELU() )

def get_model_from_cfg( cfg: DictConfig, device: torch.device, embedding_layer: EmbeddingLayer  ) -> nn.Module:
	model: nn.Sequential = nn.Sequential( embedding_layer )
	cnn_channels = cfg.cnn_channels
	num_input_features = embedding_layer.nfeatures
	for iblock in range(cfg.num_blocks):
		cnn_channels = add_cnn_block( model, cnn_channels, num_input_features, cfg )
		num_input_features = -1
	reduced_series_len = embedding_layer.output_series_length // int( math.pow(cfg.pool_size, cfg.num_blocks) )
	print(f"CNN: reduced_series_len={reduced_series_len}, cnn_channels={cnn_channels}, output_series_length={embedding_layer.output_series_length}")
	add_dense_block( model, cnn_channels*reduced_series_len, cfg )
	return model.to(device)
