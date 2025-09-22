import numpy as np, os, time
import xarray as xa
import tensorflow as tf
from tensorflow import keras
from argparse import Namespace
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
opts.defaults( opts.Curve(width=1000, height=600, line_width=1) )
import tmodel
use_current = True

signal_index=2
data=tmodel.get_demo_data()
signals = data['signals']
times = data['times']
T: np.ndarray = times[signal_index].copy()
Y: np.ndarray = signals[signal_index]
validation_split = int(0.8 * T.shape[0])
Tt = T[:validation_split]
avars, args = {}, None

for feature_type in range(5):
	args: Namespace = tmodel.load_args(signal_index, feature_type)
	latest_ckp_file = tmodel.get_ckp_file( args, "latest" )
	if os.path.exists(latest_ckp_file):
		print( f"\n\t\t * Computing attribution for Feature type {feature_type} *" )
		X: np.ndarray = tmodel.get_features( T, feature_type, args )
		Xt = X[:validation_split]

		strategy = tf.distribute.MirroredStrategy([f"GPU:{i}" for i in args.devices])
		with strategy.scope():
			model = tmodel.create_streams_model( X.shape[1], 0.0, n_streams=args.nstreams )
			model.compile( optimizer=tf.keras.optimizers.Adam( learning_rate=0.01 ), loss=args.loss )
		assert os.path.exists(latest_ckp_file), f"Checkpint file '{latest_ckp_file}' not found."
		print( f"Loading checkpoint from '{latest_ckp_file}'")
		model.load_weights(latest_ckp_file)

		A, P = tmodel.get_masked_attribution( model, Xt )
		avars[ f"AF{feature_type}" ] = xa.DataArray( P, name=f"AF{feature_type}", dims=["feature","time"], coords={"time":Tt, "feature":np.arange(P.shape[0])}, attrs=dict(scores=A) )

xa.Dataset( avars ).to_netcdf( tmodel.attribution_path( args, signal_index) )







