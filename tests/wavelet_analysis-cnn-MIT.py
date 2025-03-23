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
	threshold = 1.0
	for elem, (TIC, xsignal) in enumerate(MIT_loader.dataset.data_vars.items()):
		if TIC.endswith(".time"):
			time_coord: np.ndarray = xsignal.values.squeeze()
			diff: np.ndarray = np.diff(time_coord)
			diffs.append( diff )
			tlen.append( (time_coord[-1]-time_coord[0]) )
			break_indies: np.ndarray = np.argwhere( diff > threshold ).squeeze()
			time_blocks: List[np.ndarray] = np.array_split( time_coord, break_indies)
			block_sizes =  np.diff(break_indies)
			idx_largest_block = np.argmax(block_sizes)
			if elem % 100 == 0:
				print( f"Largest block: {idx_largest_block}")
				print(f"block_sizes: {block_sizes}")
				print(f"time_block_sizees: {[t.size for t in time_blocks]}")
	cdiff: np.ndarray = np.concatenate(diffs)
	tlens: np.ndarray = np.array(tlen)
	print( f" *** diffs: range=({cdiff.min():.4f},{cdiff.max():.4f}) median={np.median(cdiff):.4f}")
	print( f" *** tlens: range=({tlens.min():.1f},{tlens.max():.1f}) median={np.median(tlens):.1f}")

	breaks: np.ndarray = (cdiff > threshold)
	nbreaks = np.count_nonzero(breaks)
	print(f" Threshold={threshold:.2f}: nbreaks/signal: {nbreaks/len(diffs):.1f}")


if __name__ == "__main__":
	my_app()
