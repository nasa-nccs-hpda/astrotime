import hydra, torch
from omegaconf import DictConfig
from typing import List, Optional, Dict, Type, Any
import numpy as np
from astrotime.loaders.MIT import MITLoader
import xarray as xa
from astrotime.config.context import astrotime_initialize
version = "MIT_period.wp"

def get_blocks_test( cfg, dataset: xa.Dataset, TICS: List[str] ):
	diffs: List[np.ndarray] = []
	tlen = []
	block_size_list = []
	threshold = cfg.block_gap_threshold
	for elem, TIC in enumerate(TICS):
		time_coord: np.ndarray = dataset.data_vars[TIC+".time"].values.squeeze()
		diff: np.ndarray = np.diff(time_coord)
		diffs.append( diff )
		tlen.append( (time_coord[-1]-time_coord[0]) )
		break_indices: np.ndarray = np.nonzero( diff > threshold )[0]
		print(f" ************** TIC[{elem}: {TIC} -------------------------------------------------------------- ")
		print(f"break_indices: {break_indices[:16]}")
		print(f"#time_coords: {time_coord.size}")
		if break_indices.size == 0:
			largest_block = time_coord
		elif break_indices.size == 1:
			largest_block = time_coord[0:break_indices[0]] if (break_indices[0] >= time_coord.size//2) else time_coord[break_indices[0]:]
			print(f" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ")
			print(f"largest_block.size: {largest_block.size}")
		else:
			time_blocks: List[np.ndarray] = np.array_split( time_coord, break_indices)
			bsizes: np.array =  np.array( [break_indices[0]] + np.diff(break_indices).tolist() + [time_coord.size-break_indices[-1]] )
			idx_largest_block: int = int(np.argmax(bsizes))
			largest_block: np.array = time_blocks[idx_largest_block]
			if elem % 100 == 0:
				print(f"#time_blocks: {len(time_blocks)}")
				print(f"bsizes: {bsizes[:16]}")
				print(f"Largest block: {idx_largest_block}")
				print(f"Block size: {bsizes[idx_largest_block]} {largest_block.size} ")
		block_size_list.append(largest_block.size)
	cdiff: np.ndarray = np.concatenate(diffs)
	tlens: np.ndarray = np.array(tlen)
	block_sizes: np.ndarray = np.array(block_size_list)
	print(f" ==================================================================================== ")
	print( f" *** diffs: range=({cdiff.min():.4f},{cdiff.max():.4f}) median={np.median(cdiff):.4f}")
	print( f" *** tlens: range=({tlens.min():.1f},{tlens.max():.1f}) median={np.median(tlens):.1f}")
	print(f" *** block_sizes: range=({block_sizes.min()},{block_sizes.max()}) median={np.median(block_sizes)}")

@hydra.main(version_base=None, config_path="../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	MIT_loader = MITLoader(cfg.data)
	sector_index = MIT_loader.sector_range[0]+3
	TICs: List[str] = MIT_loader.TICS(sector_index)

	MIT_loader.get_period_range(sector_index)

	get_blocks_test( cfg.data, MIT_loader.dataset, TICs[:16] )

	for TIC in TICs[:16]:
		block = MIT_loader.get_largest_block( TIC )
		print( f"{TIC}: largest_block{block.shape}")




if __name__ == "__main__":
	my_app()
