import sys, torch
from astrotime.util.series import TSet
from astrotime.loaders.base import IterativeDataLoader, RDict
from typing import List, Optional, Dict, Type, Union, Tuple

def snr_analysis( loader: IterativeDataLoader, device: torch.device ):
	with device:
		loader.initialize(TSet.Train)
		loader.init_epoch()
		try:
			for ibatch in range(0, sys.maxsize):
				batch: RDict = loader.get_next_batch()
				if batch['y'].shape[0] > 0:
					snr = batch['sn']
					print( f" batch snr{snr.shape}")

		except StopIteration:
			print(f"Completed evaluation")