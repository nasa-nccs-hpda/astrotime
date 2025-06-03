import hydra, torch
from omegaconf import DictConfig
from torch import nn
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.util.series import TSet
from astrotime.encoders.wavelet import WaveletAnalysisLayer, embedding_space
from astrotime.models.cnn.cnn_baseline import get_spectral_peak_selector_from_cfg, ExpLoss, ExpHLoss
from astrotime.config.context import astrotime_initialize
from astrotime.loaders.synthetic import SyntheticElementLoader
from astrotime.models.spectral.peak_finder import Evaluator
version = "synthetic_period"

@hydra.main(version_base=None, config_path="../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize( cfg, version )
	embedding_space_array, embedding_space_tensor = embedding_space(cfg.transform, device)

	data_loader = SyntheticElementLoader(cfg.data)
	data_loader.initialize(TSet.Train)

	embedding = WaveletAnalysisLayer( 'analysis', cfg.transform, embedding_space_tensor, device )
	model: nn.Module = get_spectral_peak_selector_from_cfg( cfg.model, device, embedding )

	evel = Evaluator( cfg.train, device, data_loader, model, ExpLoss(cfg.data) )
	evel.evaluate()

if __name__ == "__main__":
	my_app()