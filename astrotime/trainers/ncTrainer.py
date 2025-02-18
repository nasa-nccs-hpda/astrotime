import tensorflow as tf
import keras
from keras import layers
import numpy as np
import time


class ncTrainer(object):
	def __init__(self):
		self.batch_size = 64
		self.nepochs = 1000
		self.train_acc_metric = keras.metrics.MeanAbsoluteError()
		self.val_acc_metric = keras.metrics.MeanAbsoluteError()

	def train(self, model, optimizer, loss_fn, train_dataset, val_dataset):
		for epoch in range(self.nepochs):
			print("\nStart of epoch %d" % (epoch,))
			start_time = time.time()

			# Iterate over the batches of the dataset.
			for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
				with tf.GradientTape() as tape:
					y_batch_predict = model(x_batch_train, training=True)
					loss_value = loss_fn(y_batch_train, y_batch_predict)
				grads = tape.gradient(loss_value, model.trainable_weights)
				optimizer.apply_gradients(zip(grads, model.trainable_weights))

				# Update training metric.
				self.train_acc_metric.update_state(y_batch_train, y_batch_predict)

				# Log every 200 batches.
				if step % 200 == 0:
					print(
						"Training loss (for one batch) at step %d: %.4f"
						% (step, float(loss_value))
					)
					print("Seen so far: %d samples" % ((step + 1) * self.batch_size))

			# Display metrics at the end of each epoch.
			train_acc = self.train_acc_metric.result()
			print("Training acc over epoch: %.4f" % (float(train_acc),))

			# Reset training metrics at the end of each epoch
			self.train_acc_metric.reset_states()

			# Run a validation loop at the end of each epoch.
			for x_batch_val, y_batch_val in val_dataset:
				val_logits = model(x_batch_val, training=False)
				# Update val metrics
				self.val_acc_metric.update_state(y_batch_val, val_logits)
			val_acc = self.val_acc_metric.result()
			self.val_acc_metric.reset_states()
			print("Validation acc: %.4f" % (float(val_acc),))
			print("Time taken: %.2fs" % (time.time() - start_time))