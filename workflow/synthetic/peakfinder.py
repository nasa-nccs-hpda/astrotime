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
hloss_maxh = 5

@hydra.main(version_base=None, config_path="../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:

	device: torch.device = astrotime_initialize( cfg, version+".pf" )
	embedding_space_array, embedding_space_tensor = embedding_space(cfg.transform, device)
	data_loader = SyntheticElementLoader(cfg.data, TSet.Validation)
	embedding = WaveletAnalysisLayer( 'analysis', cfg.transform, embedding_space_tensor, device )

	print( f"PeakFinder Validation({version}):")
	for reduce_type in [0,1]:
		for use_hloss in [False, True]:
			print(f" ------ reduce_type={reduce_type}, hloss={use_hloss}, hloss_maxh={hloss_maxh}  ------ ")
			model: nn.Module = get_spectral_peak_selector_from_cfg( cfg.model, device, embedding, reduce_type=reduce_type )
			lossf = ExpHLoss(cfg.data, maxh=hloss_maxh) if use_hloss else ExpLoss(cfg.data)
			trainer = IterativeTrainer( cfg.train, device, data_loader, model, embedding, lossf )
			trainer.evaluate(version, with_checkpoint=False)

if __name__ == "__main__":
	my_app()