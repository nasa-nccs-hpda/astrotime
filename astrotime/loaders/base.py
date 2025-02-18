import numpy as np, xarray as xa
from typing import List, Optional, Dict, Type, Tuple
from astrotime.encoders.base import Encoder
import tensorflow as tf

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

class DataPreprocessor:

	def __init__(self, loader: DataLoader, encoder: Encoder):
		self.loader: DataLoader = loader
		self.encoder: Encoder = encoder
		self.nbatches: int = loader.nbatches
		self.batch_index: int = 0

	@property
	def shape(self) -> List[int]:
		return [self.loader.batch_size, self.encoder.series_len]

	def __call__(self)-> Tuple[tf.Tensor,tf.Tensor]:
		dset: xa.Dataset = self.loader.get_batch( self.batch_index )
		X, Y = self.encoder.encode_batch( dset['t'].values, dset['y'].values )
		target: tf.Tensor = tf.convert_to_tensor( dset['p'].values[:,None] )
		print( f"DataPreprocessor:get_batch({self.batch_index}: x{X.shape} y{Y.shape} target{target.shape}")
		return Y, target

	def __iter__(self):
		return self

	def __next__(self):
		self.batch_index = self.batch_index + 1
		if self.batch_index >= self.nbatches:
			self.batch_index = 0
			raise StopIteration
		return self.__call__()

	def get_dataset(self) -> tf.data.Dataset:
		output_sig = ( tf.TensorSpec(shape=self.shape+[1], dtype=tf.float32), tf.TensorSpec(shape=self.shape[:1]+[1], dtype=tf.float32) )
		return tf.data.Dataset.from_generator( self, output_signature=output_sig )


