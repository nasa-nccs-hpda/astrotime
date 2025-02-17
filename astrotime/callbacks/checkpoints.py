import keras

class CheckpointCallback(keras.callbacks.ModelCheckpoint):

	def __init__(self, model_name: str, checkpoint_filepath: str ):
		self.filepath = checkpoint_filepath + "/" + model_name + '.keras'
		keras.callbacks.ModelCheckpoint.__init__(self, self.filepath, monitor = 'val_accuracy', mode = 'max', save_best_only = True)


