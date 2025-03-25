import hydra, torch
from omegaconf import DictConfig
from torch import nn
from astrotime.encoders.baseline import IterativeEncoder
from astrotime.loaders.MIT import MITLoader
from astrotime.encoders.wavelet import WaveletAnalysisLayer
from astrotime.trainers.iterative_trainer import IterativeTrainer
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg
from astrotime.config.context import astrotime_initialize
version = "MIT_period.wp"

@hydra.main(version_base=None, config_path="../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version )
	encoder = IterativeEncoder( cfg.transform, device )
	loader = MITLoader( cfg.data )
	embedding = WaveletAnalysisLayer( cfg.transform, device)
	model: nn.Module = get_model_from_cfg( cfg.model, device, embedding )

	trainer = IterativeTrainer( cfg.train, loader, encoder, model )
	trainer.initialize_checkpointing(version)
	trainer.compute()

if __name__ == "__main__":
	my_app()