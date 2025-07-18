import hydra, torch
from omegaconf import DictConfig
from torch import nn
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.encoders.baseline import ValueEncoder
from astrotime.util.series import TSet
from astrotime.loaders.MIT import MITLoader
from astrotime.encoders.wavelet import WaveletAnalysisLayer, embedding_space
from astrotime.trainers.iterative_trainer import IterativeTrainer
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg
from astrotime.config.context import astrotime_initialize
version = "MIT_period.synthetic.folded"

@hydra.main(version_base=None, config_path="../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version )
	embedding_space_array, embedding_space_tensor = embedding_space(cfg.transform, device)
	cfg.data['test_mode'] = 'planet_crossing'

	encoder = ValueEncoder( cfg.transform, device )
	data_loader = MITLoader(cfg.data)
	data_loader.initialize(TSet.Train)

	embedding = WaveletAnalysisLayer( 'analysis', cfg.transform, embedding_space_tensor, device )
	model: nn.Module = get_model_from_cfg( cfg.model, device, embedding )

	trainer = IterativeTrainer( cfg.train, data_loader, encoder, model )
	trainer.initialize_checkpointing(version)
	trainer.train()

if __name__ == "__main__":
	my_app()