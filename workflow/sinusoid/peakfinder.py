import hydra, torch
from omegaconf import DictConfig
from torch import nn
from astrotime.util.series import TSet
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.trainers.filters import RandomDownsample, Norm
from astrotime.encoders.wavelet import WaveletAnalysisLayer, embedding_space
from astrotime.trainers.iterative_trainer import IterativeTrainer
from astrotime.trainers.loss import ExpLoss
from astrotime.models.cnn.cnn_baseline import get_spectral_peak_selector_from_cfg
from astrotime.config.context import astrotime_initialize
from astrotime.loaders.sinusoid import SinusoidElementLoader
version = "sinusoid_period"

@hydra.main(version_base=None, config_path="../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:

	device: torch.device = astrotime_initialize( cfg, version+".pf" )
	embedding_space_array, embedding_space_tensor = embedding_space(cfg.transform, device)
	data_loader = SinusoidElementLoader(cfg.data, TSet.Train)
	transforms = [RandomDownsample(cfg.transform), Norm(cfg.transform)]
	embedding = WaveletAnalysisLayer( 'analysis', cfg.transform, embedding_space_tensor, device )

	print( f"PeakFinder Validation({version}):")
	model: nn.Module = get_spectral_peak_selector_from_cfg( cfg.model, device, embedding )
	trainer = IterativeTrainer( cfg.train, device, data_loader, model, embedding, ExpLoss(cfg.data), transforms )
	trainer.evaluate()

if __name__ == "__main__":
	my_app()