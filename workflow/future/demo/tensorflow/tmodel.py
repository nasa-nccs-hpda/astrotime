import time, os, math, pickle, logging, numpy as np, shutil
from argparse import Namespace
from typing import List, Optional, Dict, Type, Tuple, Union
data_dir = os.environ.get('ASTROTIME_DATA_DIR', "/explore/nobackup/projects/ilab/data/astrotime/demo")
log_file = f"{data_dir}/astrotime.log"
current_args_path = f"{data_dir}/args.pkl"
logging.basicConfig( filename=log_file, level=logging.INFO,  format='%(asctime)s - %(levelname)s - %(message)s',  filemode='w' )

def args_path( signal: int, ftype: int ) -> str:
	return f"{data_dir}/args-{signal}-{ftype}.pkl"

def attribution_path( args, signal_index: int ) -> str:
	results_dir = f"{args.data_dir}/attribution"
	os.makedirs(results_dir, exist_ok=True)
	return f"{results_dir}/att_data_{signal_index}.nc"

def upscale2x( T: np.ndarray, Y: np.ndarray ) -> Tuple[np.ndarray,np.ndarray]:
	T_interp = (T[1:] + T[:1])/2
	T_new = np.empty((T.size + T_interp.size,), dtype=T.dtype)
	T_new[0::2] = T; T_new[1::2] = T_interp
	return T_new, np.interp(T_new, T, Y)

def downscale2x( T: np.ndarray, Y: np.ndarray ) -> Tuple[np.ndarray,np.ndarray]:
	return  T[::2], Y[::2]

def upscale( T: np.ndarray, Y: np.ndarray, upscale_factor: int ) -> Tuple[np.ndarray,np.ndarray]:
	for i in range(upscale_factor):
		T, Y = upscale2x(T, Y)
	return T, Y

def downscale( T: np.ndarray, Y: np.ndarray, downscale_factor: int ) -> Tuple[np.ndarray,np.ndarray]:
	for i in range(downscale_factor):
		T, Y = downscale2x(T, Y)
	return T, Y

def get_demo_data( ):
	return np.load(f'{data_dir}/jordan_data.npz', allow_pickle=True)

def log( msg: str ):
	logging.info( msg )

def error( msg: str ):
	logging.exception( f"{msg}" )

def mask_feature( x: np.ndarray, iFeature: int ) -> np.ndarray:
	xf: np.ndarray = x.copy()
	xf[:,iFeature] = 0
	return xf

def get_ckp_file( args: Namespace, cptype: str ):
	base_path = f"{data_dir}/streamed_time_predict.s{args.signal}.f{args.feature_type}.nf{args.nfeatures}.bs{args.batch_size}"
	return f"{base_path}.{cptype}.weights.h5"

def parse_args( parser  ) -> Namespace:
	args: Namespace = parser.parse_args()
	return save_args(args)

def save_args( args: Namespace  ) -> Namespace:
	apath = args_path(args.signal,args.feature_type)
	afile = open( apath, 'wb' )
	pickle.dump(args, afile)
	afile.close()
	shutil.copyfile( apath, current_args_path )
	print(f" ***** Running with args: {args}")
	print(f" ***** log_file: {log_file}")
	return args

def load_args( signal: int = -1, ftype: int = -1 ) -> Namespace:
	apath = current_args_path if ftype < 0 else args_path(signal,ftype)
	afile = open(apath, 'rb')
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
	import tensorflow as tf
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

def float_to_binary_array(x: float, places: int) -> np.array:
	binary_str: str = float_to_binary( x, places )
	return np.array( [int(bit) for bit in binary_str], dtype=np.float64 )

def get_features( T: np.ndarray,  args: Namespace ) -> np.ndarray:
	features = []
	feature_type: int = args.feature_type
	tm = T[-1]*(1+(1.0/T.size))
	ts: np.ndarray = T/tm
	if feature_type == 0:
		return np.stack( [ float_to_binary_array(x,args.nfeatures) for x in ts.tolist() ], axis=0 )
	elif feature_type in (1,2,3,5):
		omega = 2*math.pi
		for ip in range(args.nfeatures):
			f = np.cos if (feature_type==2) else np.sin
			features.append( f(omega*ts) )
			if feature_type == 5:
				features.append( np.cos(omega*ts) )
			omega = omega*2
		sf = np.stack(features, axis=1)
		if feature_type==3:
			fbin = np.stack( [ float_to_binary_array(x,args.nfeatures) for x in ts.tolist() ], axis=0 )
			return  0.5*(1-fbin) + 0.5*( 0.5 + 0.5*sf )
		else:
			return sf
	elif feature_type == 4:
		p = 1.0
		for ip in range(args.nfeatures):
			features.append( np.mod(ts,p)/p )
			p = p/2
		return np.stack(features, axis=1)
	elif feature_type in (6,7):
		omega = 2*math.pi
		f = np.cos if (feature_type == 6) else np.sin
		for ip in range(1,args.nfeatures+1):
			features.append( f(ip*omega*ts) )
		sf = np.stack(features, axis=1)
		return sf
	else:
		raise ValueError(f"Invalid feature_type: {feature_type}")

def get_grad_attribution( model, X: np.ndarray, Y: np.ndarray ) -> np.ndarray:
	import tensorflow as tf
	with tf.GradientTape() as tape:
		y_pred = model(X, training=False)
		loss = tf.keras.losses.mean_squared_error(Y, y_pred)
	grads = tape.gradient(loss, model.trainable_variables)
	return np.stack( [ g.numpy().flatten() for g in grads ], axis=1 )

def mse( Y: np.ndarray, P: np.ndarray ) -> float:
	E = P - Y
	return np.mean( E*E )

def mae( Y: np.ndarray, P: np.ndarray ) -> float:
	return np.mean( np.abs( P - Y ))

def get_masked_attribution( model, X ) -> Tuple[np.ndarray,np.ndarray]:
	P = model.predict(X)
	A, R = [], [ P.flatten() ]
	for iF in range(X.shape[1]):
		print( f"Computing masked attribution for feature {iF} ... ", flush=True )
		Xm = mask_feature(X, iF)
		Y = model.predict(Xm)
		A.append( np.mean( np.abs( P - Y )) )
		R.append( Y.flatten() )
	Ap =  np.array(A)
	Ar =  np.stack(R, axis=0)
	return Ap/Ap.mean(), Ar
