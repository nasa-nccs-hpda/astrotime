import numpy as np, xarray as xa
from typing import List, Optional, Dict, Type, Tuple
from astrotime.encoders.base import Encoder
import tensorflow as tf
from astrotime.util.logging import lgm, exception_handled, log_timing, shp

class DataLoader:

	def __init__(self):
		pass

	def get_dataset( self, dset_idx: int ) -> Dict[ str, np.ndarray]:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_dataset' method")

	def get_batch(self, batch_index: int) -> xa.Dataset:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'get_batch' method")

	@property
	def nbatches(self) -> int:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'nbatches' property")

	@property
	def batch_size(self) -> int:
		raise NotImplementedError(f"The class '{self.__class__.__name__}' does not implement the 'batch_size' property")

	@property
	def nelements(self) -> int:
		return self.nbatches * self.batch_size

class DataGenerator(Sequence):

	def __init__(self, loader: DataLoader, encoder: Encoder):
		Sequence.__init__(self)
		self.loader: DataLoader = loader
		self.encoder: Encoder = encoder
		self.nbatches: int = loader.nbatches
		self.batch_size = self.loader.batch_size

	def on_epoch_end(self):
		pass

	@exception_handled
	def __getitem__(self, batch_index):
		dset: xa.Dataset = self.loader.get_batch( batch_index )
		x, y = dset['t'].values, dset['y'].values
		lgm().log(f" DataPreprocessor:get_batch({batch_index}: x{shp(x)} y{shp(y)}")
		X, Y = self.encoder.encode_batch( x, y )
		target: tf.Tensor = tf.convert_to_tensor( dset['p'].values[:,None] )
		lgm().log( f"  ENCODED --->  y{Y.shape} target{target.shape}")
		return Y, target

	def __len__(self):
		return self.nbatches