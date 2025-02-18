import tensorflow as tf
import keras
from keras import layers
import numpy as np
import time

class ncTrainer(object):

	def __init__(self, model, **kwargs):
		self.batch_size = 64
		self.nepochs = 1000
		self.model = model
		self.optimizer = keras.optimizers.SGD(learning_rate=kwargs.get('lr',1e-3))
		self.loss_fn = keras.losses.MeanAbsoluteError()

	@tf.function
	def train_step(self, x, y):
		with tf.GradientTape() as tape:
			logits = self.model(x, training=True)
			loss_value = self.loss_fn(y, logits)
		grads = tape.gradient(loss_value, self.model.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
		self.train_acc_metric.update_state(y, logits)
		return loss_value

	def train(self, train_dataset):
		for epoch in range(self.nepochs):
			print("\nStart of epoch %d" % (epoch,))

			for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
				loss_value = self.train_step(x_batch_train, y_batch_train)

				if step % 200 == 0:
					print( "Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)) )
					print("Seen so far: %d samples" % ((step + 1) * self.batch_size))
