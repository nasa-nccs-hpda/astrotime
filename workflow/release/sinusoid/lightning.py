import hydra, torch
from omegaconf import DictConfig
from torch import nn
from astrotime.util.series import TSet
from typing import List, Optional, Dict, Type, Union, Tuple
import pytorch_lightning as PL
from astrotime.models.cnn.lightning import PLSpectralCNN
from astrotime.config.context import astrotime_initialize
from astrotime.datasets.sinusoids import SinusoidDataLoader
version = "sinusoid_period"

@hydra.main(version_base=None, config_path="../../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	astrotime_initialize( cfg, version )

	loader = SinusoidDataLoader(cfg.data)
	model = PLSpectralCNN(cfg)
	trainer = PL.Trainer()

	trainer.fit(model=model, train_dataloaders=loader, ckpt_path=model.ckpt_path(version) )
