import os, glob, hydra, torch
from omegaconf import DictConfig, OmegaConf
# from typing import Any, Dict, List, Tuple, Type, Optional, Union, Hashable
import xarray as xa, numpy as np
from torch import nn
from astrotime.loaders.sinusoid import ncSinusoidLoader
from astrotime.encoders.baseline import ValueEncoder, ValueEmbeddingLayer
from astrotime.trainers.signal_trainer import SignalTrainer
from astrotime.models.cnn_baseline import get_model_from_cfg
from astrotime.config.context import astrotime_initialize
version = "sinusoid_period.baseline_small"

@hydra.main(version_base=None, config_path="../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version )
	sinusoid_loader = ncSinusoidLoader( cfg.data )
	encoder = ValueEncoder( cfg.transform, device )
	embedding = ValueEmbeddingLayer( cfg.transform, device )
	model: nn.Module = get_model_from_cfg( cfg.model, device, embedding, encoder )

	trainer = SignalTrainer( cfg.train, sinusoid_loader, encoder, model )
	trainer.initialize_checkpointing(version)
	trainer.train()

if __name__ == "__main__":
	my_app()
