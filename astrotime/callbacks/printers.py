import tensorflow as tf
import keras, logging
from astrotime.util.logging import lgm, exception_handled, log_timing

class ShapePrinter(keras.callbacks.Callback):

	def __init__(self, input_shapes):
		super().__init__()
		self.input_shapes = input_shapes

	def on_train_begin(self, logs=None):
		lgm().log("Training is starting. Printing initial shapes:")
		self.print_shapes()

	def on_epoch_begin(self, epoch, logs=None):
		if lgm().get_level() == logging.DEBUG:
			lgm().log(f"Epoch {epoch} is starting. Printing shapes:")
			self.print_shapes()

	def print_shapes(self):
		sinusoid_shape = self.input_shapes
		lgm().log(f"Shape of sinusoid: {sinusoid_shape}")

		# Create dummy inputs with the correct shape
		x: tf.Tensor = tf.zeros((1,) + sinusoid_shape[1:])

		if x.ndim == 2:
			x = tf.expand_dims(x, axis=-1)
		lgm().log(f"-> Shape after expand_dims: sinusoid {x.shape}")

		x = self.model.conv1(x)
		lgm().log(f"-> Shape conv1: {x.shape}")

		x = self.model.conv2(x)
		lgm().log(f"-> Shape conv2: {x.shape}")

		x = self.model.conv3(x)
		lgm().log(f"-> Shape conv3: {x.shape}")

		x = self.model.maxpool1(x)
		lgm().log(f"-> Shape maxpool1: {x.shape}")

		x = self.model.conv4(x)
		lgm().log(f"-> Shape conv4: {x.shape}")

		x = self.model.conv5(x)
		lgm().log(f"-> Shape conv5: {x.shape}")

		x = self.model.conv6(x)
		lgm().log(f"-> Shape conv6: {x.shape}")

		x = self.model.maxpool2(x)
		lgm().log(f"-> Shape maxpool2: {x.shape}")

		x = self.model.conv7(x)
		lgm().log(f"-> Shape conv7: {x.shape}")

		x = self.model.conv8(x)
		lgm().log(f"-> Shape conv8: {x.shape}")

		x = self.model.conv9(x)
		lgm().log(f"-> Shape conv9: {x.shape}")

		x = self.model.maxpool3(x)
		lgm().log(f"-> Shape maxpool3: {x.shape}")

		x = self.model.conv10(x)
		lgm().log(f"-> Shape conv10: {x.shape}")

		x = self.model.conv11(x)
		lgm().log(f"-> Shape conv11: {x.shape}")

		x = self.model.conv12(x)
		lgm().log(f"-> Shape conv12: {x.shape}")

		x = self.model.maxpool4(x)
		lgm().log(f"-> Shape maxpool4: {x.shape}")

		x = self.model.conv13(x)
		lgm().log(f"-> Shape conv13: {x.shape}")

		x = self.model.conv14(x)
		lgm().log(f"-> Shape conv14: {x.shape}")

		x = self.model.conv15(x)
		lgm().log(f"-> Shape conv15: {x.shape}")

		x = self.model.maxpool5(x)
		lgm().log(f"-> Shape maxpool5: {x.shape}")

		x = self.model.conv16(x)
		lgm().log(f"-> Shape conv16: {x.shape}")

		x = self.model.conv17(x)
		lgm().log(f"-> Shape conv17: {x.shape}")

		x = self.model.conv18(x)
		lgm().log(f"-> Shape conv18: {x.shape}")

		x = self.model.maxpool6(x)
		lgm().log(f"-> Shape maxpool6: {x.shape}")

		x = self.model.conv19(x)
		lgm().log(f"-> Shape conv19: {x.shape}")

		x = self.model.conv20(x)
		lgm().log(f"-> Shape conv20: {x.shape}")

		x = self.model.conv21(x)
		lgm().log(f"-> Shape conv21: {x.shape}")

		x = self.model.maxpool7(x)
		lgm().log(f"-> Shape maxpool6: {x.shape}")

		x = self.model.flatten(x)
		lgm().log(f"-> Shape flatten: {x.shape}")

		x = self.model.dense1(x)
		lgm().log(f"-> Shape dense1: {x.shape}")

		period = self.model.dense2(x)
		lgm().log(f"Shape of output (period): {period.shape}")
