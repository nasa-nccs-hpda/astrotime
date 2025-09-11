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
			features.append( np.array([int(bit) for bit in binary_str], dtype=np.float64) )
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

def alpha( ip: int, ipsel: int ):
	return 1.0 if ip == ipsel else 0.05

def select_feature( plots: List[plt.Line2D], fig, sval: float):
	try:
		log(f"Selecting feature {sval:.3f}, isel={int(sval)} ")
		for ip in range(len(plots)):
			a = alpha(ip, int(sval))
			log(f"  ** f{ip}: a= {a:.3f} ")
			plots[ip].set_alpha(a)
		fig.canvas.draw_idle()
	except:
		error( f"Error while selecting feature:  " )




