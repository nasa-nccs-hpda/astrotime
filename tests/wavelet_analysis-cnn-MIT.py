import hydra, torch
from numpy.ma.core import shape
from omegaconf import DictConfig
from typing import List, Optional, Dict, Type, Any
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

	diffs: List[np.ndarray] = []
	tlen = []
	for TIC, xsignal in MIT_loader.dataset.data_vars.items():
		if TIC.endswith(".time"):
			time_coord = xsignal.values
			diff = np.diff(time_coord)
			diffs.append( diff*1000 )
			tlen.append( (time_coord[-1]-time_coord[0])*1000 )
	cdiff: np.ndarray = np.concatenate(diffs)
	tlens: np.ndarray = np.array(tlen)
	print( f" *** diffs(x1000): range=({cdiff.min():.3f},{cdiff.max():.3f}) median={np.median(cdiff):.3f}")
	print( f" *** tlens(x1000): range=({tlens.min():.3f},{tlens.max():.3f}) median={np.median(tlens):.3f}")
	threshold = 2000.0
	breaks: np.ndarray = (cdiff > threshold)
	nbreaks = np.count_nonzero(breaks)
	print(f" Threshold={threshold:.3f}: nbreaks/signal: {nbreaks/len(diffs):.3f}")

#print( f" diff: median={np.median(dt)}, max={np.max(dt)},  min={np.min(dt)}")
# for it in range(15):
# 	idx = it*100
# 	print( f" diff-{idx}: mean={np.mean(dt[idx,:])}, max={np.max(dt[idx,:])},  min={np.min(dt[idx,:])}")
#	threshold = 0.002
#	breaks: np.ndarray = (dt > threshold)
#	nbreaks: np.ndarray = np.count_nonzero(breaks, axis=1)
#	print(f" nbreaks{list(nbreaks.shape)}: mean={np.mean(nbreaks)}, max={np.max(nbreaks)},  min={np.min(nbreaks)}")

if __name__ == "__main__":
	my_app()
