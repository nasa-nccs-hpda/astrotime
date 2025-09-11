import time, os, math, pickle, logging, numpy as np
from argparse import Namespace
from typing import List, Optional, Dict, Type, Union, Tuple
import matplotlib.pyplot as plt
import tensorflow as tf
from pexpect.pxssh import ExceptionPxssh
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
data_dir = os.environ.get('ASTROTIME_DATA_DIR', "/explore/nobackup/projects/ilab/data/astrotime/demo")
log_file = f"{data_dir}/astrotime.log"
args_path = f"{data_dir}/args.pkl"
logging.basicConfig( filename=log_file, level=logging.INFO,  format='%(asctime)s - %(levelname)s - %(message)s',  filemode='w' )

def smooth( data: np.ndarray, window_width: int ) -> np.ndarray:
	if window_width > 0:
		cumsum_vec = np.cumsum( np.insert(data, 0, 0) )
		return (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
	else: return data

def get_demo_data( ):
	return np.load(f'{data_dir}/jordan_data.npz', allow_pickle=True)

def log( msg: str ):
	logging.info( msg )

def error( msg: str ):
	logging.exception( f"{msg}" )

def get_ckp_file( args: Namespace ):
	return f"{data_dir}/streamed_time_predict.s{args.signal}.f{args.feature_type}.nf{args.nfeatures}.bs{args.batch_size}.weights.h5"

def parse_args( parser ) -> Namespace:
	args: Namespace = parser.parse_args()
	afile = open(args_path, 'wb')
	pickle.dump(args, afile)
	afile.close()
	print(f" ***** Running with args: {args}")
	print(f" ***** log_file: {log_file}")
	return args

def load_args( ) -> Namespace:
	afile = open(args_path, 'rb')
	args = pickle.load(afile)
	afile.close()
	print(f" ***** Running with args: {args}")
	print(f" ***** log_file: {log_file}")
	return args

def tnorm(x: np.ndarray, dim: int=0) -> np.ndarray:
	m: np.ndarray = x.mean( axis=dim, keepdims=True )
	s: np.ndarray = x.std( axis=dim, keepdims=True )
	return (x - m) / s


def create_streams_model(nfeatures, dropout_frac, n_streams):
	times_input = tf.keras.Input(shape=(nfeatures,), name="times_input")

	def apply_relpos(xx):
		x = tf.keras.layers.Dense(512, activation='elu')(xx)
		x = tf.keras.layers.Dropout(dropout_frac)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Dense(512, activation='elu')(x)
		x = tf.keras.layers.Dropout(dropout_frac)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Dense(512, activation='elu')(x)
		x = tf.keras.layers.Dropout(dropout_frac)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Dense(512, activation='elu')(x)
		x = tf.keras.layers.Dropout(dropout_frac)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Dense(512, activation='elu')(x)
		x = tf.keras.layers.Dropout(dropout_frac)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Dense(512, activation='elu')(x)
		return x


	streams = [apply_relpos(times_input) for i in range(n_streams)]

	x = tf.keras.layers.Concatenate(axis=-1)(streams)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Dropout(dropout_frac)(x)
	x = tf.keras.layers.Dense(512, activation='elu')(x)
	x = tf.keras.layers.BatchNormalization()(x)

	outputs = tf.keras.layers.Dense(1, activation='linear')(x)
	model = tf.keras.Model(inputs=times_input, outputs=outputs)
	return model

def create_small_model(nfeatures: int, dropout_frac: float):
	time_features = tf.keras.Input(shape=(nfeatures,), name="time_features")

	x = tf.keras.layers.Dense(512, activation='tanh')(time_features)
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
	model = tf.keras.Model(inputs=time_features, outputs=outputs)
	return model

def create_dense_model(nfeatures: int, dropout_frac: float, n_streams: int ):
	times_input = tf.keras.Input(shape=(nfeatures,), name="times_input")

	def apply_relpos(xx):
		x = tf.keras.layers.Dense(512, activation='elu')(xx)
		x = tf.keras.layers.Dropout(dropout_frac)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Dense(512, activation='elu')(x)
		x = tf.keras.layers.Dropout(dropout_frac)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Dense(512, activation='elu')(x)
		x = tf.keras.layers.Dropout(dropout_frac)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Dense(512, activation='elu')(x)
		x = tf.keras.layers.Dropout(dropout_frac)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Dense(512, activation='elu')(x)
		x = tf.keras.layers.Dropout(dropout_frac)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Dense(512, activation='elu')(x)
		return x

	streams = [apply_relpos(times_input) for i in range(n_streams)]

	x = tf.keras.layers.Concatenate(axis=-1)(streams)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Dropout(dropout_frac)(x)
	x = tf.keras.layers.Dense(512, activation='elu')(x)
	x = tf.keras.layers.BatchNormalization()(x)

	outputs = tf.keras.layers.Dense(1, activation='linear')(x)
	model = tf.keras.Model(inputs=times_input, outputs=outputs)
	return model

def float_to_binary(fval: float, places) -> str:
	return bin(int(fval * pow(2, places)))[2:].rjust(places, '0')

def get_features( T: np.ndarray, feature_type: int, args: Namespace ) -> np.ndarray:
	features = []
	dt = T.max()/T.shape[-1]
	ts: np.ndarray = T/(T.max()+dt)
	if feature_type == 0:
		for x in ts.tolist():
			binary_str: str = float_to_binary(x, args.nfeatures)
			bin_feature = np.array( [int(bit) for bit in binary_str], dtype=np.float64 )
			features.append( smooth( bin_feature, args.smooth_win ) )
		return np.stack(features, axis=0)
	elif feature_type == 1:
		pmin = 2*args.minp_factor/T.shape[-1]
		sfactor = math.exp( math.log(1/pmin)/args.nfeatures )
		omega = 2*math.pi
		for ip in range(args.nfeatures):
			features.append( np.sin(omega*ts) )
			features.append( np.cos(omega*ts) )
			omega = omega*sfactor
		print(f"Using sfactor: {sfactor}, T{T.shape}, nf={args.nfeatures}, Pmin={2*math.pi/omega}, mpf={args.minp_factor}")
		return np.stack(features, axis=1)
	else:
		raise ValueError(f"Invalid feature_type: {feature_type}")
