import os, glob, hydra, torch
from omegaconf import DictConfig, OmegaConf
# from typing import Any, Dict, List, Tuple, Type, Optional, Union, Hashable
import xarray as xa, numpy as np
from torch import nn
from astrotime.loaders.sinusoid import ncSinusoidLoader
import logging
from astrotime.util.logging import lgm, exception_handled, log_timing
from astrotime.encoders.wavelet import WaveletEncoder
from astrotime.trainers.signal_trainer import SignalTrainer
from astrotime.models.cnn_baseline import get_model_from_cfg

@hydra.main(version_base=None, config_path="../config", config_name="sinusoid_period")
def my_app(cfg: DictConfig) -> None:
	print( cfg )
	log_level = logging.INFO
	lgm().init_logging( cfg.train, log_level )
	device: torch.device = torch.device(f"cuda:{cfg.platform.gpu}" if (torch.cuda.is_available() and (cfg.platform.gpu >= 0)) else "cpu")

	sinusoid_loader = ncSinusoidLoader( cfg.data )
	encoder = WaveletEncoder( device, cfg.transform )
	model: nn.Module = get_model_from_cfg( cfg.model )

	trainer = SignalTrainer( sinusoid_loader, encoder, model, cfg.train )
	trainer.train()

if __name__ == "__main__":
	my_app()
