import time, os, math, numpy as np
from typing import List, Optional, Dict, Type, Union, Tuple
import matplotlib.pyplot as plt
from decimal import getcontext, Decimal
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

	return fractional_binary

def get_features( T: np.ndarray, feature_type: int ) -> np.ndarray:
	features = []
	t, tL = T-T[0], T[-1]-T[0]
	if feature_type == 0:
		for x in T.tolist():
			binary_str = np.binary_repr(np.float64(x).view(np.int64), width=64)
			features.append( np.array([int(bit) for bit in binary_str], dtype=np.float64) )
		return np.stack(features, axis=0)
	elif feature_type == 1:
		for x in T.tolist():
			binary_str = float_to_binary_precise(x/tL, places=64)
			features.append( np.array([int(bit) for bit in binary_str], dtype=np.float64) )
		return np.stack(features, axis=0)
	elif feature_type == 2:
		for ip in range(1,12):
			alpha = math.pi*math.pow(2,ip)/tL
			features.append( np.sin(alpha*t) )
			features.append( np.cos(alpha*t) )
		return np.stack(features, axis=1)
	elif feature_type == 3:
		features.append(t/tL)
		for ibase, npow in [ (2,12), (3,8), (5,5), (6,4), (7,3) ]:
			for ip in range(1,npow+1):
				base = tL/math.pow( ibase, ip )
				features.append( np.mod(t,base)/base )
		return np.stack(features, axis=1)
	else:
		raise ValueError(f"Invalid feature_type: {feature_type}")

def alpha( ip: int, ipsel: int ):
	return 1.0 if ip == ipsel else 0.1

def select_feature( plots: List[plt.Line2D], fig, sval: float):
	for ip in range(len(plots)):
		plots[ip].set_alpha( alpha(ip,int(sval)) )
	fig.canvas.draw_idle()



