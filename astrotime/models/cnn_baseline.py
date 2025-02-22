from torch import nn
import torch, math
from omegaconf import DictConfig, OmegaConf
from astrotime.encoders.base import Encoder
from typing import Any, Dict, List, Optional, Tuple, Mapping

def add_cnn_block( model: nn.Sequential, nchannels: int, num_input_features: int, cfg: DictConfig ) -> int:
	in_channels = num_input_features if (num_input_features > 0) else nchannels
	out_channels = nchannels
	for iL in range( cfg.num_cnn_layers ):
		out_channels = out_channels + cfg.cnn_expansion_factor
		model.append( nn.Conv1d( in_channels, out_channels, kernel_size=cfg.kernel_size, stride=cfg.stride, padding='same') )
		model.append(nn.ELU())
		in_channels = out_channels
	model.append(nn.ELU())
	model.append( nn.BatchNorm1d(out_channels) )
	model.append( nn.MaxPool1d(cfg.pool_size) )
	return out_channels

def add_dense_block( model: nn.Sequential, in_channels:int, cfg: DictConfig ):
	model.append( nn.Flatten() )
	model.append( nn.Linear( in_channels, cfg.dense_channels ) )  # 64
	model.append( nn.ELU() )
	model.append( nn.Linear( cfg.dense_channels, cfg.out_channels ) )

def get_model_from_cfg( cfg: DictConfig, encoder: Encoder, device: torch.device ) -> nn.Module:
	model: nn.Sequential = nn.Sequential()
	cnn_channels = cfg.cnn_channels
	for iblock in range(cfg.num_blocks):
		cnn_channels = add_cnn_block( model, cnn_channels, encoder.nfeatures, cfg )
	reduced_series_len = encoder.series_length // int( math.pow(cfg.pool_size, cfg.num_blocks) )
	add_dense_block( model, cnn_channels*reduced_series_len, cfg )
	return model.to(device)
