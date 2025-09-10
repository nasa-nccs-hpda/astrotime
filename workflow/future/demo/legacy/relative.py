import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
from tqdm import tqdm
import random
import os
from scipy.stats import norm
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt

def float_to_binary_precise(num, places=64):
	getcontext().prec = places
	decimal_num = Decimal(str(num))
	integer_part = int(decimal_num)
	fractional_part = decimal_num - integer_part

	integer_binary = bin(integer_part)[2:]  # remove '0b' prefix

	fractional_binary = ""
	for _ in range(places):
		fractional_part *= 2
		bit = int(fractional_part)
		fractional_binary += str(bit)
		fractional_part -= bit

	# return f"{integer_binary}.{fractional_binary}"
	return fractional_binary

def float_to_binary_array_not_IEEE(x):
	binary_str = float_to_binary_precise(x, places=64)
	return np.array([int(bit) for bit in binary_str], dtype=np.float64)

class FirstLastElementConstraint(tf.keras.constraints.Constraint):
	def __call__(self, w):
		input_dim = w.shape[-2]
		output_dim = w.shape[-1]
		mask = tf.concat([
			tf.ones((1, input_dim, output_dim)),
			tf.zeros((tf.shape(w)[0] - 2, input_dim, output_dim)),
			tf.ones((1, input_dim, output_dim))
		], axis=0)
		return w * mask

def create_rel_pos_model(dropout_frac, num_conv):
	binary_times_input = tf.keras.Input(shape=(64,), name="binary_times_input")

	binary_times_reshape = tf.keras.layers.Reshape((64, 1))(binary_times_input)

	pos_layers = []
	for i in range(num_conv):
		conv = tf.keras.layers.Conv1D(8, i + 2, strides=1, padding='valid', activation='tanh', kernel_constraint=FirstLastElementConstraint() if i > 0 else None)
		pos = conv(binary_times_reshape)
		pos = tf.keras.layers.Flatten()(pos)
		pos = tf.keras.layers.Dense(32, activation='tanh')(pos)
		pos_layers.append(pos)

	pos = tf.keras.layers.Concatenate(axis=-1)(pos_layers)
	pos = tf.keras.layers.Dropout(dropout_frac)(pos)
	pos = tf.keras.layers.BatchNormalization()(pos)

	x = pos
	for _ in range(6):
		x = tf.keras.layers.Dense(512, activation='tanh')(x)
		x = tf.keras.layers.Dropout(dropout_frac)(x)
		x = tf.keras.layers.BatchNormalization()(x)

	x = tf.keras.layers.Dense(512, activation='tanh')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	outputs = tf.keras.layers.Dense(1, activation='linear')(x)

	model = tf.keras.Model(inputs=binary_times_input, outputs=outputs)
	return model

# load the data
data = np.load('jordan_data.npz', allow_pickle=True)
signals = data['signals']
times = data['times']

# each index is a different time series... pick an index from 0-99
index = 1

X = times[index].copy()
# Normalize
X = X / X.max()
# Use new base 2 embedding
X = np.array([float_to_binary_array_not_IEEE(X[i]) for i in range(len(X))])
Y = signals[index]
validation_split = int(0.8 * X.shape[0])

# separate into train and test
Xtrain = X[:validation_split]
Xval = X[validation_split:]
Ytrain = Y[:validation_split]
Yval = Y[validation_split:]

# num_conv is the maximum size of the kernel... It can be anything up to 63... a high number is probably overkill
model = create_rel_pos_model(dropout_frac=0.5, num_conv=62)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss='mae')

# save the weights at the lowest val_loss
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("embed_time_predict_rel_pos.weights.h5", save_best_only=True, save_weights_only=True, monitor='val_loss')

# train
history = model.fit(
	Xtrain,
	Ytrain,
	epochs=10000,
	validation_data=(Xval, Yval),
	callbacks=[checkpoint_callback],
	batch_size=512,
	shuffle=True
)

# plot

model.load_weights('embed_time_predict_rel_pos.weights.h5')
p0 = model.predict(Xtrain, batch_size=256)
p1 = model.predict(Xval, batch_size=256)

plt.figure(figsize=(15, 5))
plt.plot(times[index], Y, label='truth')
plt.plot(times[index][:validation_split], p0[:, 0], label='train prediction')
plt.plot(times[index][validation_split:], p1[:, 0], label='val prediction')

plt.savefig('timepred35.png')
plt.close()