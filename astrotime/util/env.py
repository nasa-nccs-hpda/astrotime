import argparse, torch, numpy as np
from argparse import Action
from torch import Tensor
from typing import Any, Dict, List, Tuple, Mapping, Union
from argparse import Namespace
defaults = dict( world_size=torch.cuda.device_count(), gpu=0, refresh_state=False, port=23467 )
CPU = -1
Array = Union[np.ndarray,Tensor]

def to_torch( x: Array, device: torch.device, **kwargs ) -> torch.Tensor:
	if type(x) is np.ndarray: x = torch.Tensor(x, **kwargs)
	return x.to( device )

class ConfigAction(Action):
	def __init__(self, option_strings, configuration: Dict[str,str],  *args, **kwargs):
		self._configuration = configuration
		super(ConfigAction, self).__init__(option_strings=option_strings, *args, **kwargs)

	def __call__(self, parser, namespace, values, option_string=None):
		print( f" ConfigAction: values={values}, dest={self.dest}, configuration={self._configuration}")
		setattr(namespace, self.dest, values)

def _get_args(configuration: Dict[str,str]) -> Namespace:
	argparser = argparse.ArgumentParser(description=f'Execute workflow')
	argparser.add_argument('-r',  '--refresh_state', action='store_true', help="Refresh workflow by deleting existing checkpoints and learning stats")
	argparser.add_argument('-gpu', '--gpu', nargs='?', default=defaults['gpu'], type=int, help="GPU ID to use")
	argparser.add_argument('-cpu', '--cpu', action='store_true', help="Run on CPU")
	for cid, cvalue in configuration.items():
		argparser.add_argument( '-'+cid, '--'+cid, help=f"Reset {cid} configuration",  action=ConfigAction, configuration=configuration )
	return argparser.parse_args()

def parse_clargs( configuration: Dict[str,str] ) -> Namespace:
	args:Namespace = _get_args(configuration)
	if args.cpu: args.gpu = CPU
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