import hydra, torch
from omegaconf import DictConfig
from torch import nn
from astrotime.loaders.MIT import MITElementLoader
from astrotime.encoders.spectral import SpectralProjection, embedding_space
from astrotime.trainers.iterative_trainer import IterativeTrainer
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg
from astrotime.config.context import astrotime_initialize
version = "MIT_period"

@hydra.main(version_base=None, config_path="../../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version )
	cfg.data['snr_min'] = 80.0
	embedding_space_array, embedding_space_tensor = embedding_space(cfg.transform, device)
	embedding = SpectralProjection( cfg.transform, embedding_space_tensor, device )
	model: nn.Module = get_model_from_cfg(cfg, embedding).to(device)

	data_loader = MITElementLoader(cfg.data)
	trainer = IterativeTrainer( cfg, device, data_loader, model, embedding )
	trainer.evaluate(version)

if __name__ == "__main__":
	my_app()