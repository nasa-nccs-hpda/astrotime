import numpy as np, os, time
import tensorflow as tf
from tensorflow import keras
from argparse import Namespace
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tmodel, argparse
default_data_dir = "/explore/nobackup/projects/ilab/data/astrotime/demo"
def intlist(arg:str): return list(map(int, arg.split(",")))

parser = argparse.ArgumentParser( prog='timehascome', usage='python train.py --help', description='Trains time-aware CNN on demo data.')
parser.add_argument('-s',  '--signal',        type=int, default=2)
parser.add_argument('-f',  '--feature_type',  type=int, default=1)
parser.add_argument('-ne', '--nepochs',       type=int, default=2000)
parser.add_argument('-nf', '--nfeatures',     type=int, default=32)
parser.add_argument('-bs', '--batch_size',    type=int, default=512)
parser.add_argument('-l',  '--loss',          type=str, default="mae")
parser.add_argument('-ns', '--nstreams',      type=int, default=10)
parser.add_argument('-sw', '--smooth_win',    type=int, default=0)
parser.add_argument('-r',  '--refresh',       action='store_true')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
parser.add_argument('-pf', '--minp_factor',   type=float, default=2.0)
parser.add_argument('-do', '--dropout_frac',  type=float, default=0.5)
parser.add_argument('-dd',  '--data_dir',     type=str, default=default_data_dir)
parser.add_argument('-dv-dv',  '--devices',      type=intlist, default="0")
args: Namespace = tmodel.parse_args(parser)

signal_index=args.signal
feature_type=args.feature_type

data=tmodel.get_demo_data()
signals = data['signals']
times = data['times']
T: np.ndarray = times[signal_index].copy()
X: np.ndarray = tmodel.get_features( T, feature_type, args )
Y: np.ndarray = signals[signal_index]

strategy = tf.distribute.MirroredStrategy([f"GPU:{i}" for i in args.devices])
print(f"Number of devices: {strategy.num_replicas_in_sync}")
with strategy.scope():
    model = tmodel.create_streams_model( X.shape[1], dropout_frac=args.dropout_frac, n_streams=args.nstreams )
    model.compile( optimizer=tf.keras.optimizers.Adam( learning_rate=args.learning_rate ), loss=args.loss )

latest_ckp_file = tmodel.get_ckp_file( args, "latest" )
if args.refresh and os.path.exists(latest_ckp_file): os.remove(latest_ckp_file)
if os.path.exists(latest_ckp_file): model.load_weights(latest_ckp_file)
else: print( f"Checkpoint file '{latest_ckp_file}' not found. Training from scratch." )

P=model.predict( X, batch_size=args.batch_size )

print( f" Model shapes: X{X.shape} Y{Y.shape} P{P.shape}")

