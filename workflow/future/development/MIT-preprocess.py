import hydra, torch
from omegaconf import DictConfig
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.util.series import TSet
from astrotime.loaders.MIT import MITLoader
from astrotime.config.context import astrotime_initialize
version = "select_MIT_period"

@hydra.main(version_base=None, config_path="../../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	astrotime_initialize( cfg, version )
	data_loader = MITLoader(cfg.data)
	data_loader.initialize(TSet.Train)
	data_loader.preprocess()

if __name__ == "__main__":
	my_app()