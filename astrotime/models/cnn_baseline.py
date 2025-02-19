from keras import layers
import keras

def get_model( input_shape,  **kwargs ) -> keras.Model:
	model_layers = [
		keras.Input(shape=input_shape),
		layers.Conv1D(68, kernel_size=3, activation='elu', padding='same'),
		layers.Conv1D(72, kernel_size=3, activation='elu', padding='same'),
		layers.Conv1D(76, kernel_size=3, activation='elu', padding='same'),
		layers.BatchNormalization(),
		layers.MaxPool1D(2),
		layers.Conv1D(80, kernel_size=3, activation='elu', padding='same'),
		layers.Conv1D(84, kernel_size=3, activation='elu', padding='same'),
		layers.Conv1D(88, kernel_size=3, activation='elu', padding='same'),
		layers.BatchNormalization(),
		layers.MaxPool1D(2),
		layers.Conv1D(92, kernel_size=3, activation='elu', padding='same'),
		layers.Conv1D(96, kernel_size=3, activation='elu', padding='same'),
		layers.Conv1D(100, kernel_size=3, activation='elu', padding='same'),
		layers.BatchNormalization(),
		layers.MaxPool1D(2),
		layers.Conv1D(104, kernel_size=3, activation='elu', padding='same'),
		layers.Conv1D(108, kernel_size=3, activation='elu', padding='same'),
		layers.Conv1D(112, kernel_size=3, activation='elu', padding='same'),
		layers.BatchNormalization(),
		layers.MaxPool1D(2),
		layers.Conv1D(116, kernel_size=3, activation='elu', padding='same'),
		layers.Conv1D(120, kernel_size=3, activation='elu', padding='same'),
		layers.Conv1D(124, kernel_size=3, activation='elu', padding='same'),
		layers.BatchNormalization(),
		layers.MaxPool1D(2),
		layers.Conv1D(128, kernel_size=3, activation='elu', padding='same'),
		layers.Conv1D(132, kernel_size=3, activation='elu', padding='same'),
		layers.Conv1D(136, kernel_size=3, activation='elu', padding='same'),
		layers.BatchNormalization(),
		layers.MaxPool1D(2),
		layers.Conv1D(140, kernel_size=3, activation='elu', padding='same'),
		layers.Conv1D(144, kernel_size=3, activation='elu', padding='same'),
		layers.Conv1D(148, kernel_size=3, activation='elu', padding='same'),
		layers.BatchNormalization(),
		layers.MaxPool1D(2),
		layers.Flatten(),
		layers.Dense(64, activation='elu'),
		layers.Dense(1)
	]
	model = keras.Sequential(model_layers)
	model.compile(**kwargs)
	return model