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
		if epoch < 4:
			print(f"Epoch {epoch} is starting. Printing shapes:")
			self.print_shapes()

	def print_shapes(self):
		sinusoid_shape = self.input_shapes
		print(f"Shape of sinusoid: {sinusoid_shape}")

		# Create dummy inputs with the correct shape
		x: tf.Tensor = tf.zeros((1,) + sinusoid_shape[1:])

		if x.ndim == 2:
			x = tf.expand_dims(x, axis=-1)
		print(f"-> Shape after expand_dims: sinusoid {x.shape}")

		x = self.model.conv1(x)
		print(f"-> Shape conv1: {x.shape}")

		x = self.model.conv2(x)
		print(f"-> Shape conv2: {x.shape}")

		x = self.model.conv3(x)
		print(f"-> Shape conv3: {x.shape}")

		x = self.model.maxpool1(x)
		print(f"-> Shape maxpool1: {x.shape}")

		x = self.model.conv4(x)
		print(f"-> Shape conv4: {x.shape}")

		x = self.model.conv5(x)
		print(f"-> Shape conv5: {x.shape}")

		x = self.model.conv6(x)
		print(f"-> Shape conv6: {x.shape}")

		x = self.model.maxpool2(x)
		print(f"-> Shape maxpool2: {x.shape}")

		x = self.model.conv7(x)
		print(f"-> Shape conv7: {x.shape}")

		x = self.model.conv8(x)
		print(f"-> Shape conv8: {x.shape}")

		x = self.model.conv9(x)
		print(f"-> Shape conv9: {x.shape}")

		x = self.model.maxpool3(x)
		print(f"-> Shape maxpool3: {x.shape}")

		x = self.model.conv10(x)
		print(f"-> Shape conv10: {x.shape}")

		x = self.model.conv11(x)
		print(f"-> Shape conv11: {x.shape}")

		x = self.model.conv12(x)
		print(f"-> Shape conv12: {x.shape}")

		x = self.model.maxpool4(x)
		print(f"-> Shape maxpool4: {x.shape}")

		x = self.model.conv13(x)
		print(f"-> Shape conv13: {x.shape}")

		x = self.model.conv14(x)
		print(f"-> Shape conv14: {x.shape}")

		x = self.model.conv15(x)
		print(f"-> Shape conv15: {x.shape}")

		x = self.model.maxpool5(x)
		print(f"-> Shape maxpool5: {x.shape}")

		x = self.model.conv16(x)
		print(f"-> Shape conv16: {x.shape}")

		x = self.model.conv17(x)
		print(f"-> Shape conv17: {x.shape}")

		x = self.model.conv18(x)
		print(f"-> Shape conv18: {x.shape}")

		x = self.model.maxpool6(x)
		print(f"-> Shape maxpool6: {x.shape}")

		x = self.model.conv19(x)
		print(f"-> Shape conv19: {x.shape}")

		x = self.model.conv20(x)
		print(f"-> Shape conv20: {x.shape}")

		x = self.model.conv21(x)
		print(f"-> Shape conv21: {x.shape}")

		x = self.model.maxpool7(x)
		print(f"-> Shape maxpool6: {x.shape}")

		x = self.model.flatten(x)
		print(f"-> Shape flatten: {x.shape}")

		x = self.model.dense1(x)
		print(f"-> Shape dense1: {x.shape}")

		period = self.model.dense2(x)
		print(f"Shape of output (period): {period.shape}")
