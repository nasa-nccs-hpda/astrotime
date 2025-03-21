import hydra, torch
from numpy.ma.core import shape
from omegaconf import DictConfig
from torch import nn
import numpy as np
from astrotime.loaders.MIT import MITLoader
from astrotime.encoders.baseline import ValueEncoder, ValueEmbeddingLayer
from astrotime.trainers.signal_trainer import SignalTrainer
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg
from astrotime.config.context import astrotime_initialize
version = "MIT_period.wp"

@hydra.main(version_base=None, config_path="../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	device: torch.device = astrotime_initialize(cfg, version)
	MIT_loader = MITLoader(cfg.data)
	MIT_loader.load_sector( MIT_loader.sector_range[0], refresh=True )

#	times = []
#	for signal in MIT_loader.dataset.data_vars.values():
#		time_coord = signal.coords["time"].values
#		times.append( time_coord )
#	time = np.stack(times,axis=0)
# dt: np.ndarray = np.diff(time,axis=1)
# print( f" *** times{list(time.shape)} dt{list(dt.shape)}")
# #print( f" diff: median={np.median(dt)}, max={np.max(dt)},  min={np.min(dt)}")
# for it in range(15):
# 	idx = it*100
# 	print( f" diff-{idx}: mean={np.mean(dt[idx,:])}, max={np.max(dt[idx,:])},  min={np.min(dt[idx,:])}")
#	threshold = 0.002
#	breaks: np.ndarray = (dt > threshold)
#	nbreaks: np.ndarray = np.count_nonzero(breaks, axis=1)
#	print(f" nbreaks{list(nbreaks.shape)}: mean={np.mean(nbreaks)}, max={np.max(nbreaks)},  min={np.min(nbreaks)}")

if __name__ == "__main__":
	my_app()
