import numpy as np
import tensorflow as tf
from keras import layers
import keras
from sklearn.preprocessing import MinMaxScaler

class SinusoidPeriodModel(keras.Model):
	def __init__(self, seq_length, embedding_dim=64):
		super(SinusoidPeriodModel, self).__init__()
		self.seq_length = seq_length
		self.conv1 = layers.Conv1D(68, kernel_size=3, activation='elu', padding='same')
		self.conv2 = layers.Conv1D(72, kernel_size=3, activation='elu', padding='same')
		self.conv3 = layers.Conv1D(76, kernel_size=3, activation='elu', padding='same')
		self.batchnorm1 = layers.BatchNormalization()
		self.maxpool1 = layers.MaxPool1D(2)
		self.conv4 = layers.Conv1D(80, kernel_size=3, activation='elu', padding='same')
		self.conv5 = layers.Conv1D(84, kernel_size=3, activation='elu', padding='same')
		self.conv6 = layers.Conv1D(88, kernel_size=3, activation='elu', padding='same')
		self.batchnorm2 = layers.BatchNormalization()
		self.maxpool2 = layers.MaxPool1D(2)
		self.conv7 = layers.Conv1D(92, kernel_size=3, activation='elu', padding='same')
		self.conv8 = layers.Conv1D(96, kernel_size=3, activation='elu', padding='same')
		self.conv9 = layers.Conv1D(100, kernel_size=3, activation='elu', padding='same')
		self.batchnorm3 = layers.BatchNormalization()
		self.maxpool3 = layers.MaxPool1D(2)
		self.conv10 = layers.Conv1D(104, kernel_size=3, activation='elu', padding='same')
		self.conv11 = layers.Conv1D(108, kernel_size=3, activation='elu', padding='same')
		self.conv12 = layers.Conv1D(112, kernel_size=3, activation='elu', padding='same')
		self.batchnorm4 = layers.BatchNormalization()
		self.maxpool4 = layers.MaxPool1D(2)
		self.conv13 = layers.Conv1D(116, kernel_size=3, activation='elu', padding='same')
		self.conv14 = layers.Conv1D(120, kernel_size=3, activation='elu', padding='same')
		self.conv15 = layers.Conv1D(124, kernel_size=3, activation='elu', padding='same')
		self.batchnorm5 = layers.BatchNormalization()
		self.maxpool5 = layers.MaxPool1D(2)
		self.conv16 = layers.Conv1D(128, kernel_size=3, activation='elu', padding='same')
		self.conv17 = layers.Conv1D(132, kernel_size=3, activation='elu', padding='same')
		self.conv18 = layers.Conv1D(136, kernel_size=3, activation='elu', padding='same')
		self.batchnorm6 = layers.BatchNormalization()
		self.maxpool6 = layers.MaxPool1D(2)
		self.conv19 = layers.Conv1D(140, kernel_size=3, activation='elu', padding='same')
		self.conv20 = layers.Conv1D(144, kernel_size=3, activation='elu', padding='same')
		self.conv21 = layers.Conv1D(148, kernel_size=3, activation='elu', padding='same')
		self.batchnorm7 = layers.BatchNormalization()
		self.maxpool7 = layers.MaxPool1D(2)

		self.flatten = layers.Flatten()
		self.dense1 = layers.Dense(64, activation='elu')
		self.dense2 = layers.Dense(1)

	def call(self, inputs):
		sinusoid = inputs

		# Ensure inputs have the correct shape
		x = tf.expand_dims(sinusoid, axis=-1)  # (batch_size, seq_length, 1)

		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.batchnorm1(x)
		x = self.maxpool1(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.conv6(x)
		x = self.batchnorm2(x)
		x = self.maxpool2(x)
		x = self.conv7(x)
		x = self.conv8(x)
		x = self.conv9(x)
		x = self.batchnorm3(x)
		x = self.maxpool3(x)
		x = self.conv10(x)
		x = self.conv11(x)
		x = self.conv12(x)
		x = self.batchnorm4(x)
		x = self.maxpool4(x)
		x = self.conv13(x)
		x = self.conv14(x)
		x = self.conv15(x)
		x = self.batchnorm5(x)
		x = self.maxpool5(x)
		x = self.conv16(x)
		x = self.conv17(x)
		x = self.conv18(x)
		x = self.batchnorm6(x)
		x = self.maxpool6(x)
		x = self.conv19(x)
		x = self.conv20(x)
		x = self.conv21(x)
		x = self.batchnorm7(x)
		x = self.maxpool7(x)

		x = self.flatten(x)
		x = self.dense1(x)
		period = self.dense2(x)
		return period

# Create and compile the model
seq_length = 1000
data_dir = "/explore/nobackup/projects/ilab/data/astro_sigproc/sinusoids/npz/"

data = np.load( data_dir+'sinusoids_0.npz', allow_pickle=True)

sinusoids = data['sinusoids']
periods = data['periods']

Xs = []

for s in sinusoids:
	scaler = MinMaxScaler()
	Xs.append(scaler.fit_transform(s[:1000].reshape(-1, 1))[:, 0])

Xs = np.array(Xs)

val_data = np.load( data_dir+'sinusoids_1.npz', allow_pickle=True)

val_sinusoids = data['sinusoids']
val_periods = data['periods']

val_Xs = []

for s in val_sinusoids:
	scaler = MinMaxScaler()
	val_Xs.append(scaler.fit_transform(s[:1000].reshape(-1, 1))[:, 0])

val_Xs = np.array(val_Xs)

class ShapePrinter(keras.callbacks.Callback):
	def __init__(self, input_shapes):
		super().__init__()
		self.input_shapes = input_shapes

	def on_train_begin(self, logs=None):
		print("Training is starting. Printing initial shapes:")
		self.print_shapes()

	def on_epoch_begin(self, epoch, logs=None):
		if epoch == 0:
			print(f"Epoch {epoch} is starting. Printing shapes:")
			self.print_shapes()

	def print_shapes(self):
		sinusoid_shape = self.input_shapes
		print(f"Shape of sinusoid: {sinusoid_shape}")

		# Create dummy inputs with the correct shape
		sinusoid = tf.zeros((1,) + sinusoid_shape[1:])

		# Expand dimensions
		x = tf.expand_dims(sinusoid, axis=-1)
		print(f"Shape after expand_dims: sinusoid {sinusoid.shape}")

		x = self.model.conv1(x)
		print(f"Shape after conv1: {x.shape}")

		x = self.model.conv2(x)
		print(f"Shape after conv2: {x.shape}")

		x = self.model.conv3(x)
		print(f"Shape after conv3: {x.shape}")

		x = self.model.maxpool1(x)
		print(f"Shape after maxpool1: {x.shape}")

		x = self.model.conv4(x)
		print(f"Shape after conv4: {x.shape}")

		x = self.model.conv5(x)
		print(f"Shape after conv5: {x.shape}")

		x = self.model.conv6(x)
		print(f"Shape after conv6: {x.shape}")

		x = self.model.maxpool2(x)
		print(f"Shape after maxpool2: {x.shape}")

		x = self.model.conv7(x)
		print(f"Shape after conv7: {x.shape}")

		x = self.model.conv8(x)
		print(f"Shape after conv8: {x.shape}")

		x = self.model.conv9(x)
		print(f"Shape after conv9: {x.shape}")

		x = self.model.maxpool3(x)
		print(f"Shape after maxpool3: {x.shape}")

		x = self.model.conv10(x)
		print(f"Shape after conv10: {x.shape}")

		x = self.model.conv11(x)
		print(f"Shape after conv11: {x.shape}")

		x = self.model.conv12(x)
		print(f"Shape after conv12: {x.shape}")

		x = self.model.maxpool4(x)
		print(f"Shape after maxpool4: {x.shape}")

		x = self.model.conv13(x)
		print(f"Shape after conv13: {x.shape}")

		x = self.model.conv14(x)
		print(f"Shape after conv14: {x.shape}")

		x = self.model.conv15(x)
		print(f"Shape after conv15: {x.shape}")

		x = self.model.maxpool5(x)
		print(f"Shape after maxpool5: {x.shape}")

		x = self.model.conv16(x)
		print(f"Shape after conv16: {x.shape}")

		x = self.model.conv17(x)
		print(f"Shape after conv17: {x.shape}")

		x = self.model.conv18(x)
		print(f"Shape after conv18: {x.shape}")

		x = self.model.maxpool6(x)
		print(f"Shape after maxpool6: {x.shape}")

		x = self.model.conv19(x)
		print(f"Shape after conv19: {x.shape}")

		x = self.model.conv20(x)
		print(f"Shape after conv20: {x.shape}")

		x = self.model.conv21(x)
		print(f"Shape after conv21: {x.shape}")

		x = self.model.maxpool7(x)
		print(f"Shape after maxpool6: {x.shape}")

		x = self.model.flatten(x)
		print(f"Shape after flatten: {x.shape}")

		x = self.model.dense1(x)
		print(f"Shape after dense1: {x.shape}")

		period = self.model.dense2(x)
		print(f"Shape of output (period): {period.shape}")

# Create the model
model = SinusoidPeriodModel(seq_length=1000)  # Assuming seq_length is 1000

# Compile the model
model.compile(optimizer='rmsprop', loss='mae')

# Create an instance of the ShapePrinter callback
shape_printer = ShapePrinter(input_shapes=Xs.shape)

# Train the model with the callback
history = model.fit(
	Xs,
	periods,
	validation_data=(val_Xs, val_periods),
	epochs=1000,  # Reduced number of epochs
	batch_size=64,
	shuffle=True,
	callbacks=[shape_printer],
	verbose=1
)

