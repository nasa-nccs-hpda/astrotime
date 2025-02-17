import keras, os

class CheckpointCallback(keras.callbacks.ModelCheckpoint):

	def __init__(self, model_name: str, checkpoint_filepath: str ):
		self.filepath = checkpoint_filepath + "/" + model_name + '.weights.h5'
		keras.callbacks.ModelCheckpoint.__init__( self, self.filepath, save_weights_only=True )

	def load_weights(self, model):
		if os.path.exists(self.filepath):
			model.load_weights(self.filepath)
			print(f"Loaded weights from checkpoint: {self.filepath}")
		else:
			print("No checkpoint found.  Training from scratch.")


