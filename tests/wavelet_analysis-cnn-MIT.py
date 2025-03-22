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
	MIT_loader.load_sector( MIT_loader.sector_range[0] )
	times: np.ndarray = MIT_loader.dataset.coords["time"].values
	dt: np.ndarray = np.diff(times)
	print( f" *** times{times.shape} dt{dt.shape}")
	print( f" diff: median={np.median(dt)}, max={np.max(dt)},  min={np.min(dt)}")

if __name__ == "__main__":
	my_app()
