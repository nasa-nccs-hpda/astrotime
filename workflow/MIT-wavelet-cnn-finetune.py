import hydra, torch
from omegaconf import DictConfig
from torch import nn
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.util.series import TSet
from astrotime.loaders.MIT import MITLoader
from astrotime.encoders.wavelet import WaveletAnalysisLayer, embedding_space
from astrotime.trainers.iterative_trainer import IterativeTrainer
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg, ExpLoss, ExpU
from astrotime.config.context import astrotime_initialize
version = "MIT_period"
ckp_version = "synthetic_period"

@hydra.main(version_base=None, config_path="../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version )
	embedding_space_array, embedding_space_tensor = embedding_space(cfg.transform, device)

	data_loader = MITLoader(cfg.data)
	data_loader.initialize(TSet.Train)

	embedding = WaveletAnalysisLayer( 'analysis', cfg.transform, embedding_space_tensor, device )
	model: nn.Module = get_model_from_cfg( cfg.model, device, embedding, ExpU(cfg) )

	trainer = IterativeTrainer( cfg.train, device, data_loader, model, ExpLoss(cfg) )
	trainer.compute(ckp_version)

if __name__ == "__main__":
	my_app()