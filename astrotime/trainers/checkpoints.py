import torch, time, traceback, pickle, shutil
from typing import Any, Dict, List, Optional
from astrotime.config.context import cfg
from astrotime.util.logging import lgm
from torch.optim.optimizer import Optimizer
from astrotime.util.config import TSet
from torch import nn
import os


class CheckpointManager(object):

	def __init__(self, model: nn.Module, optimizer: Optimizer, rank: int = 0 ):
		self._cpaths: Dict[str,str] = {}
		self.model: nn.Module = model
		self.rank = rank
		self.optimizer = optimizer

	def save_checkpoint(self, tset: TSet, acc_losses: Dict[str,float], mdata: Dict  ) -> str:
		if self.rank ==  0:
			t0 = time.time()
			train_mdata = dict( **acc_losses, **mdata )
			checkpoint = dict(  model_state_dict=self.model.state_dict(), optimizer_state_dict=self.optimizer.state_dict(), **train_mdata )
			cpath = self.checkpoint_path(tset)
			if os.path.isfile(cpath):
				shutil.copyfile( cpath, self.checkpoint_path(tset,backup=True) )
			torch.save( checkpoint, cpath )
			print( acc_losses )
			slosses = {  k: f'{v:.3f}' for k,v in acc_losses.items() if v is not None}
			print(f" *** SAVE {tset.name} checkpoint to {cpath}, dt={time.time()-t0:.4f} sec, losses={slosses}" )
			return cpath

	def _load_state(self, tset: TSet ) -> Dict[str,Any]:
		# sdevice = f'cuda:{cfg().training.gpu}' if torch.cuda.is_available() else 'cpu'
		cpath = self.checkpoint_path(tset)
		checkpoint = torch.load( cpath, map_location='cpu' ) # torch.device(sdevice) )
		return checkpoint

	def load_checkpoint( self, tset: TSet = TSet.Train, **kwargs ) -> Optional[Dict[str,Any]]:
		update_model = kwargs.get('update_model', False)
		cppath = self.checkpoint_path( tset )
		train_state, cp_exists = {}, os.path.exists( cppath )
		if cp_exists:
			try:
				train_state = self._load_state( tset )
				lgm().log(f"Loaded model checkpoint from {cppath}, update_model = {update_model}", display=True)
				if update_model:
					self.model.load_state_dict( train_state.pop('model_state_dict') )
					self.optimizer.load_state_dict( train_state.pop('optimizer_state_dict') )
			except Exception as e:
				lgm().log(f"Unable to load model from {cppath}: {e}", display=True)
				traceback.print_exc()
				return None
		else:
			lgm().log( f"No checkpoint file found at '{cppath}': starting from scratch.")
		lgm().log( f" ------ Saving checkpoints to '{cppath}' ------ " )
		return train_state

	def clear_checkpoints( self ):
		for tset in [ TSet.Train, TSet.Validation ]:
			cppath = self.checkpoint_path(tset)
			try:
				os.remove(cppath)
				lgm().log(f" >> Clearing state: {cppath}")
			except FileNotFoundError: pass


	@classmethod
	def checkpoint_path( cls, tset: TSet, ext: str = "pt", backup=False ) -> str:
		vtset: TSet = TSet.Validation if (tset == TSet.Test) else tset
		cpath = f"{cfg().platform.results}/checkpoints/{cfg().task.training_version}.{vtset.value}"
		if backup: cpath = f"{cpath}.backup"
		os.makedirs(os.path.dirname(cpath), 0o777, exist_ok=True)
		return cpath + '.' + ext

