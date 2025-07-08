import hydra, torch
from omegaconf import DictConfig
from torch import nn
from astrotime.util.series import TSet
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.models.transformer.trainer import IterativeTrainer
from astrotime.config.context import astrotime_initialize
from astrotime.loaders.synthetic import SyntheticElementLoader
version = "synthetic_period_transformer"

@hydra.main(version_base=None, config_path="../../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version )

	data_loader = SyntheticElementLoader(cfg.data, TSet.Train)
	trainer = IterativeTrainer( cfg, device, data_loader, verbose=True )
	trainer.compute(version)

if __name__ == "__main__":
	my_app()