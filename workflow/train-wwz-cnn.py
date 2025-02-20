import os, glob, hydra, torch
from omegaconf import DictConfig, OmegaConf
# from typing import Any, Dict, List, Tuple, Type, Optional, Union, Hashable
# import xarray as xr, numpy as np
# from torch import nn
from astrotime.loaders.sinusoid import ncSinusoidLoader
# from astrotime.encoders.wavelet import WaveletEncoder
# from astrotime.trainers.signal_trainer import SignalTrainer
# from astrotime.models.cnn_baseline import get_model_from_cfg

@hydra.main(version_base=None, config_path="../config", config_name="sinusoid_period")
def my_app(cfg: DictConfig) -> None:
	device: torch.device = torch.device(f"cuda:{cfg.platform.gpu}" if (torch.cuda.is_available() and (cfg.platform.gpu >= 0)) else "cpu")
	print( f"Config = {cfg}" )
	sinusoid_loader = ncSinusoidLoader( cfg.data )

	# encoder = WaveletEncoder( device, series_length, nfreq, fbounds, fscale, nfeatures, int(max_series_length*(1-sparsity)) )
	#
	# model: nn.Module = get_model_from_cfg( cc.cfg.model )
	# trainer = SignalTrainer( signal, encoder, model, clargs, cc.cfg.training )
	# trainer.train()

if __name__ == "__main__":
	my_app()
