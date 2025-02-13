import numpy as np, xarray as xa
from astrotime.loaders.base import DataLoader
from typing import List, Optional, Dict, Type

# Create and compile the model
seq_length = 1000

class WaveletLoader(DataLoader):

	def __init__(self, data_dir: str, nfeatures: int):
		super().__init__()
		self.data_dir = data_dir
		self.nfeatures = nfeatures

	def get_dataset( self, dset_idx: int ) -> Dict[ str, np.ndarray]:
		dset = xa.open_dataset( f"{self.data_dir}/wwz-{dset_idx}.nc")
		y: np.ndarray  = dset['batch'].values[:, 0:self.nfeatures, :].transpose(0, 2, 1)
		x: np.ndarray     = dset['freq'].values
		target: np.ndarray   = dset['target'].values
		return dict( y=y, x=x, target=target )