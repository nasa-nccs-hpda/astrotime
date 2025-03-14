import hydra, torch
from omegaconf import DictConfig
from torch import nn
from astrotime.encoders.baseline import ValueEncoder
from astrotime.loaders.sinusoid import ncSinusoidLoader
from astrotime.encoders.wavelet import WaveletProjectionLayer
from astrotime.trainers.signal_trainer import SignalTrainer
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg
from astrotime.config.context import astrotime_initialize
version = "sinusoid_period.wpk"

@hydra.main(version_base=None, config_path="../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version )
	encoder = ValueEncoder( cfg.transform, device )
	sinusoid_loader = ncSinusoidLoader( cfg.data )
	embedding = WaveletProjectionLayer( cfg.transform, device)
	model: nn.Module = get_model_from_cfg( cfg.model, device, embedding )

	trainer = SignalTrainer( cfg.train, sinusoid_loader, encoder, model )
	trainer.initialize_checkpointing(version)
	trainer.compute()

if __name__ == "__main__":
	my_app()