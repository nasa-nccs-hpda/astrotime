import hydra, torch
from omegaconf import DictConfig
from torch import nn
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.encoders.correlation import AutoCorrelationLayer
from astrotime.encoders.wavelet import WaveletAnalysisLayer, embedding_space
from astrotime.trainers.iterative_trainer import IterativeTrainer
from astrotime.trainers.loss import ExpLoss, ExpHLoss
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg, ExpU
from astrotime.config.context import astrotime_initialize
from astrotime.loaders.synthetic import SyntheticElementLoader

version = "synthetic_period_autocorr"

@hydra.main(version_base=None, config_path="../../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version )
	embedding_space_array, embedding_space_tensor = embedding_space(cfg.transform, device)

	data_loader = SyntheticElementLoader(cfg.data)

	embedding = AutoCorrelationLayer( 'analysis', cfg.transform, embedding_space_tensor, device )
	model: nn.Sequential = nn.Sequential( embedding )

	trainer = IterativeTrainer( cfg.train, device, data_loader, model, ExpHLoss(cfg.data) )
	trainer.test_model()

if __name__ == "__main__":
	my_app()