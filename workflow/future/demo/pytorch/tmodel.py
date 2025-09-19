import time, os, math, pickle, logging, numpy as np, shutil, torch
from argparse import Namespace
from checkpoints import CheckpointManager
from typing import List, Optional, Dict, Type, Tuple, Union
from torch import Tensor, device, nn
from torch.distributed import init_process_group, destroy_process_group
data_dir = os.environ.get('ASTROTIME_DATA_DIR', "/explore/nobackup/projects/ilab/data/astrotime/demo")
log_file = f"{data_dir}/astrotime.log"
current_args_path = f"{data_dir}/args.pkl"
logging.basicConfig( filename=log_file, level=logging.INFO,  format='%(asctime)s - %(levelname)s - %(message)s',  filemode='w' )

def ddp_setup(rank: int, world_size: int, args: Namespace):
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "12355"
	torch.cuda.set_device(rank)
	init_process_group(backend="nccl", rank=rank, world_size=world_size)

def args_path( signal: int, ftype: int ) -> str:
	return f"{data_dir}/args-{signal}-{ftype}.pkl"

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
	return f"{data_dir}/streamed_time_predict.s{args.signal}.f{args.feature_type}.nf{args.nfeatures}.bs{args.batch_size}.{cptype}.weights.h5"

def parse_args( parser  ) -> Namespace:
	args: Namespace = parser.parse_args()
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

def build_stream_layer( N_input_features, dropout_frac, N_hidden_features=512 ):
	stream: nn.Sequential = nn.Sequential()
	stream.append(nn.Linear(N_input_features, N_hidden_features, dtype=torch.float32 ))
	stream.append(nn.ELU())
	stream.append( nn.Dropout(p=dropout_frac) )
	stream.append( nn.BatchNorm1d(N_hidden_features) )
	return stream

def build_final_layer( N_input_features, dropout_frac, N_hidden_features=512 ):
	stream: nn.Sequential = nn.Sequential()
	stream.append(nn.BatchNorm1d(N_input_features))
	stream.append( nn.Dropout(p=dropout_frac) )
	stream.append(nn.Linear(N_input_features, N_hidden_features, dtype=torch.float32 ))
	stream.append(nn.ELU())
	stream.append(nn.BatchNorm1d(N_hidden_features))
	stream.append(nn.Linear(N_hidden_features, 1, dtype=torch.float32 ))
	return stream

def build_network_stream( N_input_features, dropout_frac, N_hidden_features=512, nlayers: int = 5 ) -> nn.Sequential:
	streams: nn.Sequential = nn.Sequential()
	nfeatures = N_input_features

	for iF in range(nlayers):
		streams.append( build_stream_layer(nfeatures, dropout_frac, N_hidden_features) )
		nfeatures = N_hidden_features

	streams.append(nn.Linear(N_hidden_features, N_hidden_features, dtype=torch.float32 ))
	streams.append(nn.ELU())
	return streams


class MultiStreamModel(nn.Module):
	def __init__(self, N_input_features, dropout_frac, n_streams, N_hidden_features ):
		super().__init__()
		self.streams: List[nn.Module] = [ build_network_stream(N_input_features, dropout_frac, N_hidden_features) for i in range(n_streams) ]
		self.final_layer = build_final_layer( N_hidden_features*n_streams, dropout_frac, N_hidden_features )

	def forward(self, x):
		outputs = [stream(x) for stream in self.streams]
		combined_output = torch.cat(outputs, dim=-1)
		return self.final_layer(combined_output)

def float_to_binary(fval: float, places) -> str:
	return bin(int(fval * pow(2, places)))[2:].rjust(places, '0')

def float_to_binary_array(x: float, places: int) -> np.array:
	binary_str: str = float_to_binary( x, places )
	return np.array( [int(bit) for bit in binary_str], dtype=np.float64 )

def get_features( T: np.ndarray, feature_type: int, args: Namespace ) -> np.ndarray:
	features = []
	tm = T[-1]*(1+(1.0/T.size))
	ts: np.ndarray = T/tm
	if feature_type == 0:
		return np.stack( [ float_to_binary_array(x,args.nfeatures) for x in ts.tolist() ], axis=0 )
	elif feature_type in (1,2):
		omega = 2*math.pi
		for ip in range(args.nfeatures):
			features.append(np.sin(omega*ts))
			omega = omega*2
		sf = np.stack(features, axis=1)
		return sf if (feature_type==1) else np.where(sf>0, 1, 0)
	elif feature_type in (3,4):
		omega = 2*math.pi
		for ip in range(args.nfeatures//2):
			features.append(np.sin(omega*ts))
			features.append(np.cos(omega*ts))
			omega = omega*2
		sf = np.stack(features, axis=1)
		return sf if (feature_type==3) else np.where(sf>0, 1, 0)
	else:
		raise ValueError(f"Invalid feature_type: {feature_type}")

def mse( Y: np.ndarray, P: np.ndarray ) -> float:
	E = P - Y
	return np.mean( E*E )

def mae( Y: np.ndarray, P: np.ndarray ) -> float:
	return np.mean( np.abs( P - Y ))

def get_masked_attribution( model, X, Y, args ) -> np.ndarray:
	P = model.predict(X, training=False)
	L = mse(Y, P)
	A = []
	for iF in range(X.shape[1]):
		print( f"Computing masked attribution for feature {iF} ... ", flush=True )
		Xm = mask_feature(X, iF)
		Pm = model.predict(Xm, training=False)
		A.append( mse(Y, Pm) - L )

	return np.array(A)

def initialize_checkpointing( version: str, model, optimizer, args: Namespace) -> CheckpointManager:
	checkpoint_manager = CheckpointManager(version, model, optimizer, args)
	if args.refresh: checkpoint_manager.clear_checkpoints()
	checkpoint_manager.load_checkpoint(update_model=True)
	return checkpoint_manager
