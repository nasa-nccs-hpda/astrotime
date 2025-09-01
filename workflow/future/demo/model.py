import time, os, math, numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
data_dir = "/explore/nobackup/projects/ilab/data/astrotime/demo"

def get_demo_data( ):
	return np.load(f'{data_dir}/jordan_data.npz', allow_pickle=True)

def get_ckp_file( expt_index, signal_index ):
	return f"{data_dir}/embed_time_predict.e{expt_index}.s{signal_index}.weights.h5"

def create_small_model(nfeatures: int, dropout_frac: float):
	binary_times_input = tf.keras.Input(shape=(nfeatures,), name="time_features")

	x = tf.keras.layers.Dense(512, activation='tanh')(binary_times_input)
	x = tf.keras.layers.Dropout(dropout_frac)(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Dense(512, activation='tanh')(x)
	x = tf.keras.layers.Dropout(dropout_frac)(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Dense(512, activation='tanh')(x)
	x = tf.keras.layers.Dropout(dropout_frac)(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Dense(512, activation='tanh')(x)
	x = tf.keras.layers.BatchNormalization()(x)

	outputs = tf.keras.layers.Dense(1, activation='linear')(x)
	model = tf.keras.Model(inputs=binary_times_input, outputs=outputs)
	return model

def get_features( T: np.ndarray, feature_type: int = 0 ) -> np.ndarray:
	features = []
	t, tL = T-T[0], T[-1]-T[0]
	features.append(t/tL)
	for ibase, npow in [ (2,12), (3,8), (5,5), (6,4), (7,3) ]:
		for ip in range(1,npow+1):
			base = tL/math.pow( ibase, ip )
			features.append( np.mod(t,base)/base )
	return np.stack(features, axis=1)