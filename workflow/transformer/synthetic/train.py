import hydra, torch
from omegaconf import DictConfig
from torch import nn
from astrotime.util.series import TSet
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.trainers.loss import ExpLoss, ExpU
from astrotime.models.transformer.trainer import IterativeTrainer
from astrotime.config.context import astrotime_initialize
from astrotime.loaders.synthetic import SyntheticElementLoader
version = "synthetic_period_transformer"

@hydra.main(version_base=None, config_path="../../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version )

	data_loader = SyntheticElementLoader(cfg.data, TSet.Train)
	trainer = IterativeTrainer( cfg, device, data_loader, activation=ExpU(cfg.data), loss=ExpLoss(cfg.data), verbose=False )
	trainer.test_learning(version)
#	trainer.compute(version)

if __name__ == "__main__":
	my_app()