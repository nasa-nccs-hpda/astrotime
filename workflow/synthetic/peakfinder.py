import hydra, torch
from omegaconf import DictConfig
from torch import nn
from astrotime.util.series import TSet
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.encoders.wavelet import WaveletAnalysisLayer, embedding_space
from astrotime.trainers.iterative_trainer import IterativeTrainer
from astrotime.trainers.loss import ExpLoss, ExpHLoss, ExpU
from astrotime.models.cnn.cnn_baseline import get_spectral_peak_selector_from_cfg
from astrotime.config.context import astrotime_initialize
from astrotime.loaders.synthetic import SyntheticElementLoader

version = "synthetic_period"
use_hloss = False
reduce_type = 0

@hydra.main(version_base=None, config_path="../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:

	device: torch.device = astrotime_initialize( cfg, version+".pf" )
	embedding_space_array, embedding_space_tensor = embedding_space(cfg.transform, device)

	data_loader = SyntheticElementLoader(cfg.data, TSet.Validation)

	embedding = WaveletAnalysisLayer( 'analysis', cfg.transform, embedding_space_tensor, device )

	print( f"PeakFinder Validation({version}):")
	for reduce_type in [0,1]:
		for use_hloss in [False, True]:
			print(f" ------ reduce_type={reduce_type}, hloss={use_hloss}  ------ ")
			model: nn.Module = get_spectral_peak_selector_from_cfg( cfg.model, device, embedding, reduce_type=reduce_type )
			lossf = ExpHLoss if use_hloss else ExpLoss
			trainer = IterativeTrainer( cfg.train, device, data_loader, model, embedding, lossf(cfg.data) )
			trainer.evaluate(version, with_checkpoint=False)

if __name__ == "__main__":
	my_app()