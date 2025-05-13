import sys, torch, numpy as np
from astrotime.util.series import TSet
from astrotime.loaders.base import IterativeDataLoader, RDict
from typing import List, Optional, Dict, Type, Union, Tuple


def snr_analysis( loader: IterativeDataLoader ) -> np.ndarray:
	loader.initialize(TSet.Train)
	loader.init_epoch()
	snrl = []
	try:
		for ibatch in range(0, sys.maxsize):
			batch: RDict = loader.get_next_batch()
			if batch['y'].shape[0] > 0:
				snrl.append( batch['sn'].flatten() )
	except StopIteration: pass
	return np.concatenate(snrl)



