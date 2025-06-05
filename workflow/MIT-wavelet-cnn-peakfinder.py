import hydra, torch
from omegaconf import DictConfig
from torch import nn
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.loaders.MIT import MITElementLoader
from astrotime.encoders.wavelet import WaveletAnalysisLayer, embedding_space
from astrotime.models.spectral.peak_finder import Evaluator
from astrotime.models.cnn.cnn_baseline import get_spectral_peak_selector_from_cfg, ElemExpHLoss
from astrotime.config.context import astrotime_initialize
version = "select_MIT_period"

@hydra.main(version_base=None, config_path="../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version )
	embedding_space_array, embedding_space_tensor = embedding_space(cfg.transform, device)

	data_loader = MITElementLoader(cfg.data)
	embedding = WaveletAnalysisLayer( 'analysis', cfg.transform, embedding_space_tensor, device )
	model: nn.Module = get_spectral_peak_selector_from_cfg( cfg.model, device, embedding )

	evel = Evaluator( cfg.train, device, data_loader, model, ElemExpHLoss(cfg.data) )
	evel.evaluate()

if __name__ == "__main__":
	my_app()