import numpy as np, os, time
import xarray as xa
import tensorflow as tf
from tensorflow import keras
from argparse import Namespace
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
	print(f"\n\t\t * Computing attribution for Feature type {feature_type} *")
	args: Namespace = tmodel.load_args(signal_index, feature_type)
	latest_ckp_file = tmodel.get_ckp_file( args, "latest" )
	if os.path.exists(latest_ckp_file):
		X: np.ndarray = tmodel.get_features( T, args )
		Xt = X[:validation_split]

		strategy = tf.distribute.MirroredStrategy([f"GPU:{i}" for i in args.devices])
		with strategy.scope():
			model = tmodel.create_streams_model( X.shape[1], 0.0, n_streams=args.nstreams )
			model.compile( optimizer=tf.keras.optimizers.Adam( learning_rate=0.01 ), loss=args.loss )
		try:
			model.load_weights(latest_ckp_file)
			print(f"Loading checkpoint from '{latest_ckp_file}'")
			A, P = tmodel.get_masked_attribution( model, Xt )
			avars[ f"AF{feature_type}" ] = xa.DataArray( P, name=f"AF{feature_type}", dims=["feature","time"], coords={"time":Tt, "feature":np.arange(P.shape[0])}, attrs=dict(scores=A) )
		except OSError as e:
			print(f" ---> Unable to read checkpoint file '{latest_ckp_file}', skipping this feature type.")
	else:
		print(f" ---> Checkpoint file '{latest_ckp_file}' does not exist, skipping this feature type.")

att_path = tmodel.attribution_path( args, signal_index )
xa.Dataset( avars ).to_netcdf( att_path )
print(f"\n  *** Saved attribution datset to '{att_path}' *** ")







