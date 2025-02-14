import numpy as np
from astrotime.loaders.base import DataLoader
from typing import List, Optional, Dict, Type
import tensorflow as tf

# Create and compile the model
seq_length = 1000

class SinusoidLoader(DataLoader):

	def __init__(self, device: str, data_dir: str):
		super().__init__()
		self.data_dir = data_dir
		self.device = tf.device( device )

	def get_dataset( self, dset_idx: int ) -> Dict[ str, tf.Tensor ]:
		data = np.load( f"{self.data_dir}/sinusoids_{dset_idx}.npz", allow_pickle=True)
		with self.device:
			sinusoids: tf.Tensor = tf.convert_to_tensor( data['sinusoids'], dtype=tf.float32 )
			times: tf.Tensor = tf.convert_to_tensor( data['times'] )
			periods: tf.Tensor = tf.convert_to_tensor( data['periods'] )
		return dict( y=sinusoids, x=times, target=periods )