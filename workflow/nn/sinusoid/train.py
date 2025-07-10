import hydra, torch
from omegaconf import DictConfig
from torch import nn
import numpy as np
from astrotime.util.series import TSet
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.trainers.filters import RandomDownsample, Norm
from astrotime.loaders.sinusoid import SinusoidElementLoader
from astrotime.encoders.wavelet import WaveletAnalysisLayer, embedding_space
from astrotime.trainers.iterative_trainer import IterativeTrainer
from astrotime.trainers.loss import ExpLoss, ExpU
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg
from astrotime.config.context import astrotime_initialize
version = "sinusoid_period"

@hydra.main(version_base=None, config_path="../../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version )
	embedding_space_array, embedding_space_tensor = embedding_space(cfg.transform, device)
	transforms = [ RandomDownsample(cfg.transform), Norm(cfg.transform) ]
	data_loader = SinusoidElementLoader(cfg.data, TSet.Train)

	embedding = WaveletAnalysisLayer( 'analysis', cfg.transform, embedding_space_tensor, device )
	model: nn.Module = get_model_from_cfg( cfg.model,  embedding, activation=ExpU(cfg.data) ).to(device)

	trainer = IterativeTrainer( cfg.train, device, data_loader, model, embedding, ExpLoss(cfg.data), transforms )
	trainer.compute(version)

if __name__ == "__main__":
	my_app()