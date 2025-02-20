import argparse, torch, numpy as np
from torch import Tensor
from typing import Any, Dict, List, Tuple, Mapping, Union
from argparse import Namespace
defaults = dict( world_size=torch.cuda.device_count(), gpu=0, refresh_state=False, port=23467 )
CPU = -1
Array = Union[np.ndarray,Tensor]

def to_torch( x: Array, device: torch.device, **kwargs ) -> torch.Tensor:
	if type(x) is np.ndarray: x = torch.Tensor(x, **kwargs)
	return x.to( device )

def _get_args() -> Namespace:
	argparser = argparse.ArgumentParser(description=f'Execute workflow')
	argparser.add_argument('-r',  '--refresh_state', action='store_true', help="Refresh workflow by deleting existing checkpoints and learning stats")
	argparser.add_argument('-ws', '--world_size', nargs='?', default=defaults['world_size'], type=int, help="Number of gpus to use in training")
	argparser.add_argument('-p',  '--port', nargs='?', default=defaults['port'], type=int, help="Port to use in DDP")
	argparser.add_argument('-gpu', '--gpu', nargs='?', default=defaults['gpu'], type=int, help="GPU ID to use")
	argparser.add_argument('-cpu', '--cpu', action='store_true', help="Run on CPU")
	return argparser.parse_args()

def parse_clargs(ccustom: Dict[str,Any]) -> Namespace:
	args:Namespace = _get_args()
	if args.cpu:
		args.gpu = CPU
		args.world_size = 0
	elif args.gpu >= 0:
		args.world_size = 1
	print( f" Running program with args: {args}")
	return args

def default_clargs( **updates ) -> Namespace:
	clargs = Namespace( **defaults )
	for k, v in updates.items(): setattr(clargs,k,v)
	return clargs

def default_cpu_clargs( **updates ):
	return default_clargs( world_size=1, gpu=-1, **updates )

def get_device(args:Namespace) -> str:
	return f"/device:GPU:{args.gpu}" if args.gpu >= 0 else "/CPU:0"