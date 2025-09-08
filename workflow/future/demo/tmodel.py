import time, os, math, pickle, numpy as np
from argparse import Namespace
from typing import List, Optional, Dict, Type, Union, Tuple
import matplotlib.pyplot as plt
from decimal import getcontext, Decimal
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
data_dir = "/explore/nobackup/projects/ilab/data/astrotime/demo"
args_path = f"{data_dir}/args.pkl"

def get_demo_data( ):
	return np.load(f'{data_dir}/jordan_data.npz', allow_pickle=True)

def get_ckp_file( expt_index, signal_index ):
	return f"{data_dir}/embed_time_predict.e{expt_index}.s{signal_index}.weights.h5"

def parse_args( parser ) -> Namespace:
	args: Namespace = parser.parse_args()
	afile = open(args_path, 'wb')
	pickle.dump(args, afile)
	afile.close()
	print(f"\nRunning with args: {args}\n")
	return args

def load_args( ) -> Namespace:
	afile = open(args_path, 'rb')
	args = pickle.load(afile)
	afile.close()
	print(f"Running with args: {args}")
	return args

def tnorm(x: np.ndarray, dim: int=0) -> np.ndarray:
	m: np.ndarray = x.mean( axis=dim, keepdims=True )
	s: np.ndarray = x.std( axis=dim, keepdims=True )
	return (x - m) / s

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

def float_to_binary_precise(fval: float, places=64) -> str:
	return bin(int(fval * pow(2, places)))[2:].rjust(places, '0')

def get_features( T: np.ndarray, feature_type: int, nf: int = 64 ) -> np.ndarray:
	features = []
	t, tL = T-T[0], T[-1]-T[0]
	ts: np.ndarray = (t/tL)*0.9
	if feature_type == 0:
		for x in ts.tolist():
			binary_str: str = float_to_binary_precise(x, places=nf)
			features.append( np.array([int(bit) for bit in binary_str], dtype=np.float64) )
		return np.stack(features, axis=0)
	elif feature_type == 1:
		pbase = 1.2
		for ip in range(nf):
			omega = math.pi*math.pow(pbase,ip+1)
			features.append( np.sin(omega*ts) )
			features.append( np.cos(omega*ts) )
		return np.stack(features, axis=1)
	else:
		raise ValueError(f"Invalid feature_type: {feature_type}")

def alpha( ip: int, ipsel: int ):
	return 1.0 if ip == ipsel else 0.05

def select_feature( plots: List[plt.Line2D], fig, sval: float):
	for ip in range(len(plots)):
		plots[ip].set_alpha( alpha(ip,int(sval)) )
	fig.canvas.draw_idle()



