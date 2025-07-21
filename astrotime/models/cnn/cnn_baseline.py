from torch import nn
import torch, math, logging
from omegaconf import DictConfig, OmegaConf
from astrotime.encoders.embedding import EmbeddingLayer
from astrotime.trainers.loss import ExpLoss, ExpU
from astrotime.util.math import shp
from typing import Any, Dict, List, Optional, Tuple, Mapping

def add_cnn_block( cfg: DictConfig, model: nn.Sequential, nchannels: int, num_input_features: int ) -> int:
	log = logging.getLogger()
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

def add_dense_block( model: nn.Sequential, in_channels:int, hidden_channels:int, out_channels:int  ):
	log = logging.getLogger()
	print(f"CNN: add_dense_block: in_channels={in_channels}, hidden_channels={hidden_channels}, out_channels={out_channels}")
	model.append( nn.Flatten() )
	model.append( nn.Linear( in_channels, hidden_channels ) )  # 64
	model.append( nn.ELU() )
	model.append( nn.Linear( hidden_channels, out_channels  ) )

def get_model_from_cfg( gcfg: DictConfig, embedding_layer: EmbeddingLayer  ) -> nn.Module:
	log = logging.getLogger()
	mtype, cfg, dcfg = gcfg.model.mtype, gcfg.model, gcfg.data
	model: nn.Sequential = nn.Sequential( embedding_layer )
	num_input_channels = embedding_layer.output_channels
	if mtype.startswith("cnn"):
		cnn_channels = cfg.cnn_channels
		for iblock in range(cfg.num_blocks):
			cnn_channels = add_cnn_block( cfg, model, cnn_channels, num_input_channels )
			num_input_channels = -1
		reduced_series_len = embedding_layer.output_series_length // int( math.pow(cfg.pool_size, cfg.num_blocks) )
		log.info(f"CNN: reduced_series_len={reduced_series_len}, cnn_channels={cnn_channels}, output_series_length={embedding_layer.output_series_length}")
		add_dense_block( model, cnn_channels*reduced_series_len, cfg.dense_channels, cfg.out_channels  )
	elif mtype.startswith("dense"):
		in_channels = embedding_layer.output_series_length
		for iL, lsize in enumerate( cfg.layer_sizes):
			model.append( nn.Linear(in_channels, lsize))
			model.append(nn.ELU())
			in_channels = lsize
		model.append( nn.Linear(in_channels, 1) )

	if 'regression' in mtype:
		if 'octave' in mtype: model.append( nn.Sigmoid() )
		else:                 model.append( ExpU(dcfg) )
	return model

def get_spectral_peak_selector_from_cfg( cfg: DictConfig, device: torch.device, embedding_layer: EmbeddingLayer ) -> nn.Module:
	from astrotime.models.spectral.peak_finder import SpectralPeakSelector
	model: nn.Sequential = nn.Sequential( embedding_layer )
	model.append( SpectralPeakSelector( cfg, device, embedding_layer.xdata ) )
	return model.to(device)
