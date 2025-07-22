import hydra, torch
from omegaconf import DictConfig
from torch import nn
from astrotime.util.series import TSet
from astrotime.encoders.spectral import SpectralProjection, embedding_space
from astrotime.trainers.iterative_trainer import IterativeTrainer
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg
from astrotime.config.context import astrotime_initialize
from astrotime.loaders.MIT import MITElementLoader
version = "MIT_period"

@hydra.main(version_base=None, config_path="../../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version+".eval" )
	cfg.data['snr_min'] = 80.0

	embedding_space_array, embedding_space_tensor = embedding_space(cfg.transform, device)
	data_loader = MITElementLoader(cfg.data, TSet.Validation)
	embedding = SpectralProjection( cfg.transform, embedding_space_tensor, device )
	model: nn.Module = get_model_from_cfg( cfg,  embedding ).to(device)

	trainer = IterativeTrainer( cfg.train, device, data_loader, model, embedding )

	for cpversion in [None, "synthetic_period", version]:
		print( f" ---- Evaluating model, saved weights version = {cpversion} ---- ")
		trainer.evaluate(cpversion)

if __name__ == "__main__":
	my_app()