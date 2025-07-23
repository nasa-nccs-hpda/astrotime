import hydra, torch
from omegaconf import DictConfig
from torch import nn
from astrotime.encoders.baseline import IterativeEncoder
from astrotime.loaders.MIT import MITOctavesLoader
from astrotime.encoders.octaves import OctaveAnalysisLayer, embedding_space
from astrotime.encoders.embedding import EmbeddingLayer
from astrotime.trainers.iterative_trainer import IterativeTrainer
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg
from astrotime.config.context import astrotime_initialize
version = "MIT_period.octaves.pcross"

@hydra.main(version_base=None, config_path="../../../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version )
	freq_space = embedding_space(cfg.transform, device)[1]
	encoder = IterativeEncoder( cfg.transform, device )
	loader = MITOctavesLoader( cfg.data )
	embedding: EmbeddingLayer = OctaveAnalysisLayer( cfg.transform, freq_space, device )
	model: nn.Module = get_model_from_cfg( cfg.model, device, embedding )

	trainer = IterativeTrainer( cfg.train, loader, encoder, model )
	trainer.initialize_checkpointing(version)
	trainer.train()

if __name__ == "__main__":
	my_app()