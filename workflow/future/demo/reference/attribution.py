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
parser.add_argument('-nf', '--nfeatures',     type=int, default=32)
parser.add_argument('-bs', '--batch_size',    type=int, default=512)
parser.add_argument('-l',  '--loss',          type=str, default="mae")
parser.add_argument('-ns', '--nstreams',      type=int, default=10)
parser.add_argument('-dd',  '--data_dir',     type=str, default=default_data_dir)
parser.add_argument('-dv',  '--devices',      type=intlist, default="0")
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
    model = tmodel.create_streams_model( X.shape[1], 0.0, n_streams=args.nstreams )
    model.compile( optimizer=tf.keras.optimizers.Adam( learning_rate=0.01 ), loss=args.loss )

latest_ckp_file = tmodel.get_ckp_file( args, "latest" )
assert os.path.exists(latest_ckp_file), f"Checkpint file '{latest_ckp_file}' not found."

(At,Av) = tmodel.get_masked_attribution( model, X, Y, args )

print( At.shape )
print( Av.shape )





