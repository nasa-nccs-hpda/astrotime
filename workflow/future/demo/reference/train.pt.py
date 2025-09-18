import numpy as np, os, time
from argparse import Namespace
import tmodel_pt as tmodel, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from checkpoints import CheckpointManager
from torch.utils.data import DataLoader, TensorDataset
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
parser.add_argument('-dv',  '--devices',      type=intlist, default="0")
args: Namespace = tmodel.parse_args(parser)

signal_index=args.signal
feature_type=args.feature_type
version = f"{signal_index}.{feature_type}.{args.nfeatures}"

data=tmodel.get_demo_data()
signals = data['signals']
times = data['times']
T: np.ndarray = times[signal_index].copy()
X: np.ndarray = tmodel.get_features( T, feature_type, args ).astype(np.float32)
Y: np.ndarray = signals[signal_index].astype(np.float32)

validation_split = int(0.8*X.shape[0])
Xtrain=X[:validation_split]
Xval=X[validation_split:]
Ytrain=Y[:validation_split]
Yval=Y[validation_split:]

model = tmodel.MultiStreamModel( args.nfeatures, args.dropout_frac, args.nstreams)
loss_fn = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
checkpoints: CheckpointManager = tmodel.initialize_checkpointing( version, model, optimizer, args )

X_train: torch.Tensor = torch.from_numpy(Xtrain)
y_train: torch.Tensor = torch.from_numpy(Ytrain)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

for epoch in range(args.nepochs):
    model.train()
    losses = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = torch.squeeze( model(inputs) )
        loss = loss_fn( outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append( loss.item() )
    checkpoints.save_checkpoint( epoch+1, 0 )
    print(f"Epoch {epoch+1}, Mean Loss: {np.array(losses).mean():.5f}")
