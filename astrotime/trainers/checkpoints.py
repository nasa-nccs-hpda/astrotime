import torch, time, traceback, shutil
from typing import Any, Dict, Optional
from torch.optim.optimizer import Optimizer
from astrotime.util.config import TSet
from omegaconf import DictConfig
from torch import nn
import os
import logging
log = logging.getLogger("astrotime")


class CheckpointManager(object):

	def __init__(self, model: nn.Module, optimizer: Optimizer, cfg: DictConfig ):
		self._cpaths: Dict[str,str] = {}
		self.model: nn.Module = model
		self.cfg = cfg
		self.optimizer = optimizer

	def save_checkpoint(self, tset: TSet, epoch: int, batch: int  ) -> str:
		checkpoint = dict(  model_state_dict=self.model.state_dict(), optimizer_state_dict=self.optimizer.state_dict(), epoch=epoch, batch=batch )
		cpath = self.checkpoint_path(tset)
		if os.path.isfile(cpath):
			shutil.copyfile( cpath, self.checkpoint_path(tset,backup=True) )
		torch.save( checkpoint, cpath )
		return cpath

	def _load_state(self, tset: TSet ) -> Dict[str,Any]:
		cpath = self.checkpoint_path(tset)
		checkpoint = torch.load( cpath, map_location='cpu' )
		return checkpoint

	def load_checkpoint( self, tset: TSet = TSet.Train, **kwargs ) -> Optional[Dict[str,Any]]:
		update_model = kwargs.get('update_model', False)
		cppath = self.checkpoint_path( tset )
		train_state, cp_exists = {}, os.path.exists( cppath )
		if cp_exists:
			try:
				train_state = self._load_state( tset )
				log.info(f"Loaded model checkpoint from {cppath}, update_model = {update_model}", )
				if update_model:
					self.model.load_state_dict( train_state.pop('model_state_dict') )
					self.optimizer.load_state_dict( train_state.pop('optimizer_state_dict') )
			except Exception as e:
				log.info(f"Unable to load model from {cppath}: {e}", )
				traceback.print_exc()
				return None
		else:
			log.info( f"No checkpoint file found at '{cppath}': starting from scratch.")
		log.info( f" ------ Saving checkpoints to '{cppath}' ------ " )
		return train_state

	def clear_checkpoints( self ):
		for tset in [ TSet.Train, TSet.Validation ]:
			cppath = self.checkpoint_path(tset)
			try:
				os.remove(cppath)
				log.info(f" >> Clearing state: {cppath}")
			except FileNotFoundError: pass


	def checkpoint_path( self, tset: TSet, ext: str = "pt", backup=False ) -> str:
		vtset: TSet = TSet.Validation if (tset == TSet.Test) else tset
		cpath = f"{self.cfg.results_path}/checkpoints/{self.cfg.version}.{vtset.value}"
		if backup: cpath = f"{cpath}.backup"
		os.makedirs(os.path.dirname(cpath), 0o777, exist_ok=True)
		return cpath + '.' + ext

