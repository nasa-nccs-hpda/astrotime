import hydra, torch
from omegaconf import DictConfig
from torch import nn
from astrotime.encoders.polynomial import PolyExpansion
from astrotime.loaders.sinusoid import ncSinusoidLoader
from astrotime.encoders.baseline import ValueEmbeddingLayer
from astrotime.trainers.signal_trainer import SignalTrainer
from models.cnn.cnn_baseline import get_model_from_cfg
from astrotime.config.context import astrotime_initialize
version = "sinusoid_period.poly"

@hydra.main(version_base=None, config_path="../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version )
	encoder = PolyExpansion( cfg.transform, device )
	sinusoid_loader = ncSinusoidLoader( cfg.data )
	embedding = ValueEmbeddingLayer( cfg.transform, device)
	model: nn.Module = get_model_from_cfg( cfg.model, device, embedding, encoder )

	trainer = SignalTrainer( cfg.train, sinusoid_loader, encoder, model )
	trainer.initialize_checkpointing(version)
	trainer.train()

if __name__ == "__main__":
	my_app()
