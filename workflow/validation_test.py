import os, glob, hydra, torch, logging
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Hashable
from torch import nn
import numpy as np
from astrotime.encoders.baseline import ValueEncoder
from astrotime.loaders.sinusoid import ncSinusoidLoader
from astrotime.encoders.wavelet import WaveletEmbeddingLayer
from astrotime.trainers.signal_trainer import SignalTrainer
from astrotime.models.cnn_baseline import get_model_from_cfg
from astrotime.config.context import astrotime_initialize

@hydra.main(version_base=None, config_path="../config", config_name="sinusoid_period.wwz")
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg )
	encoder = ValueEncoder( cfg.transform, device )
	sinusoid_loader = ncSinusoidLoader( cfg.data )
	embedding = WaveletEmbeddingLayer( cfg.transform, device)
	model: nn.Module = get_model_from_cfg( cfg.model, device, embedding )

	trainer = SignalTrainer( cfg.train, sinusoid_loader, encoder, model )
	trainer.initialize_checkpointing()
	losses: np.ndarray = trainer.exec_validation()
	print( f"Validation Loss{list(losses.shape)}: mean={losses.mean():.3f} ({losses.min():.3f} -> {losses.max():.3f})")

if __name__ == "__main__":
	my_app()
