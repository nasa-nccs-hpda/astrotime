import tensorflow as tf
from keras import layers
import numpy as np
import keras
from astrotime.util.logging import lgm, exception_handled, log_timing

class SinusoidPeriodModel(keras.Model):

	def __init__(self):
		keras.Model.__init__(self)
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

	@exception_handled
	def call(self, *args, **kwargs):
		if len(args) >= 1:
			x = args[0]
			lgm().log(f" * Processing x{x.shape}")
			if x.ndim == 2:
				x = tf.expand_dims(x, axis=-1)
			lgm().log( f"   ----> x{x.shape}")
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