import numpy as np
from astrotime.loaders.base import DataLoader
from typing import List, Optional, Dict, Type

# Create and compile the model
seq_length = 1000

class SinusoidLoader(DataLoader):

	def __init__(self, data_dir: str):
		super().__init__()
		self.data_dir = data_dir

	def get_dataset( self, dset_idx: int ) -> Dict[ str, np.ndarray]:
		data = np.load( f"{self.data_dir}/sinusoids_{dset_idx}.npz", allow_pickle=True)
		sinusoids: np.ndarray = data['sinusoids']
		times: np.ndarray     = data['times']
		periods: np.ndarray   = data['periods']
		return dict( y=sinusoids, x=times, target=periods )