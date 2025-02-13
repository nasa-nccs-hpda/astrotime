import numpy as np, xarray as xa
from astrotime.loaders.base import DataLoader
from typing import List, Optional, Dict, Type

# Create and compile the model
seq_length = 1000

class WaveletLoader(DataLoader):

	def __init__(self, data_dir: str):
		super().__init__()
		self.data_dir = data_dir

	def get_dataset( self, dset_idx: int ) -> Dict[ str, np.ndarray]:
		dset = xa.open_dataset( f"{self.data_dir}/wwz-{dset_idx}.nc")
		y: xa.DataArray  = dset['batch']
		x: xa.DataArray     = dset['freq']
		target: xa.DataArray   = dset['target']
		return dict( y=y.values, x=x.values, target=target )