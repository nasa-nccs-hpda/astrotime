import hydra, torch
from omegaconf import DictConfig
from torch import nn
from astrotime.loaders.MIT import MITElementLoader
from astrotime.encoders.wavelet import WaveletAnalysisLayer, embedding_space
from astrotime.trainers.iterative_trainer import IterativeTrainer
from astrotime.trainers.loss import ExpHLoss
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg, ExpU
from astrotime.config.context import astrotime_initialize
version = "select_MIT_period"

@hydra.main(version_base=None, config_path="../../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version )
	espace = embedding_space(cfg.transform, device)[1]

	data_loader = MITElementLoader(cfg.data)
	embedding = WaveletAnalysisLayer( 'analysis', cfg.transform, espace, device )
	model: nn.Module = get_model_from_cfg( cfg.model,  embedding, activation=ExpU(cfg.data) ).to(device)

	trainer = IterativeTrainer( cfg.train, device, data_loader, model, ExpHLoss(cfg.data) )
	trainer.train(version)

if __name__ == "__main__":
	my_app()