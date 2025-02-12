import numpy as np
from astrotime.loaders.base import DataLoader

# Create and compile the model
seq_length = 1000

class SinusoidLoader(DataLoader):

	def __init__(self, data_dir: str):
		super().__init__()
		self.data_dir = data_dir

	def get_dataset( self, dset_idx: int ):
		data = np.load( f"{self.data_dir}/sinusoids_{dset_idx}.npz", allow_pickle=True)
		sinusoids = data['sinusoids']
		times = data['times']
		periods = data['periods']
		return dict( y=sinusoids, t=times, target=periods )