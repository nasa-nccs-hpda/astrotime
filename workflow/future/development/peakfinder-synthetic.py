import hydra, torch
from omegaconf import DictConfig
from torch import nn
from astrotime.util.series import TSet
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.encoders.spectral import SpectralProjection, embedding_space
from astrotime.trainers.iterative_trainer import IterativeTrainer
from astrotime.models.cnn.cnn_baseline import get_spectral_peak_selector_from_cfg
from astrotime.config.context import astrotime_initialize
from astrotime.loaders.synthetic import SyntheticElementLoader
version = "synthetic_period_cnn"
mtype = "peakfinder"

@hydra.main(version_base=None, config_path="../../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version+".pf" )
	embedding_space_array, embedding_space_tensor = embedding_space(cfg.transform, device)
	data_loader = SyntheticElementLoader( cfg.data )
	embedding = SpectralProjection(cfg.transform, embedding_space_tensor, device )

	model: nn.Module = get_spectral_peak_selector_from_cfg( cfg.model, device, embedding )
	trainer = IterativeTrainer( cfg, device, data_loader, model, embedding, mtype=mtype )
	trainer.evaluate(version)

if __name__ == "__main__":
	my_app()