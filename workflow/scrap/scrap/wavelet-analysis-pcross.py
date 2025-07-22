import hydra, torch
from omegaconf import DictConfig
from torch import nn
from astrotime.encoders.baseline import ValueEncoder
from astrotime.loaders.MIT import MITOctavesLoader
from astrotime.encoders.wavelet import WaveletAnalysisLayer, embedding_space d
from astrotime.trainers.signal_trainer import SignalTrainer
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg
from astrotime.config.context import astrotime_initialize
version = "MIT_period.octaves"

@hydra.main(version_base=None, config_path="../../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version )
	sector = cfg.data.sector_range[0]
	plot_freq_space, embedding_space_tensor = embedding_space(cfg.transform, device)

	encoder = ValueEncoder( cfg.transform, device )
	data_loader = MITOctavesLoader( cfg.data )
	embedding = WaveletAnalysisLayer( cfg.transform, embedding_space_tensor, device)
	model: nn.Module = get_model_from_cfg( cfg.model, device, embedding )

	trainer = SignalTrainer( cfg.train, data_loader, encoder, model )
	trainer.initialize_checkpointing(version)
	trainer.compute()

if __name__ == "__main__":
	my_app()