import tensorflow as tf
import keras

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
