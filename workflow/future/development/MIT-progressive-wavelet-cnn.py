import hydra, torch
from omegaconf import DictConfig
from torch import nn
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.util.series import TSet
from astrotime.loaders.MIT import MITElementLoader
from astrotime.encoders.wavelet import WaveletAnalysisLayer, embedding_space
from astrotime.trainers.iterative_trainer import IterativeTrainer
from astrotime.trainers.loss import ExpHLoss, ExpU
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg
from astrotime.config.context import astrotime_initialize
version = "progressive_MIT_period"
ckp_version = "select_MIT_period"

@hydra.main(version_base=None, config_path="../../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version )
	cfg.data['snr_min'] = 50.0
	cfg.data['snr_max'] = 1e9

	embedding_space_array, embedding_space_tensor = embedding_space(cfg.transform, device)
	data_loader = MITElementLoader(cfg.data)

	embedding = WaveletAnalysisLayer( 'analysis', cfg.transform, embedding_space_tensor, device )
	model: nn.Module = get_model_from_cfg( cfg.model,  embedding, activation=ExpU(cfg.data) ).to(device)

	trainer = IterativeTrainer( cfg.train, device, data_loader, model, ExpHLoss(cfg.data) )
	trainer.train(version,ckp_version)

if __name__ == "__main__":
	my_app()