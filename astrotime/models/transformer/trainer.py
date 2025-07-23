from typing import List, Optional, Dict, Tuple, Union
from omegaconf import DictConfig
from astrotime.trainers.checkpoints import CheckpointManager
from astrotime.util.tensor_ops import check_nan
from astrotime.util.math import shp
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg
from astrotime.loaders.base import Loader, RDict
from astrotime.encoders.spectral import SpectralProjection, embedding_space
import time, sys, torch, logging, numpy as np
from torch import nn, optim, Tensor
from astrotime.util.series import TSet
from astrotime.util.logging import elapsed
TRDict = Dict[str,Union[List[str],int,torch.Tensor]]

def tocpu( c, idx=0 ):
	if isinstance( c, Tensor ):
		ct = c.detach().cpu()
		if ct.ndim == 1: ct = ct[idx]
		return ct.item()
	else:
		return c

def tnorm(x: Tensor, dim: int=0) -> Tensor:
	m: Tensor = x.mean( dim=dim, keepdim=True)
	s: Tensor = torch.std( x, dim=dim, keepdim=True)
	return (x - m) / s

class IterativeTrainer(object):

	def __init__(self, cfg: DictConfig, device: torch.device, loader: Loader, **kwargs ):
		self.cfg: DictConfig = cfg.train
		self.device: torch.device = device
		self.verbose = kwargs.get('verbose',False)
		self.mtype: str = cfg.model.mtype
		self.noctaves = cfg.data.noctaves
		self.f0 = cfg.data.base_freq
		self.embedding_space_array, self.embedding_space_tensor = embedding_space(cfg.transform, device)
		self.loader: Loader = loader
		self.embedding = SpectralProjection(cfg.transform, self.embedding_space_tensor, device )
		self.model: nn.Module = get_model_from_cfg( cfg.model, self.embedding, **kwargs ).to(device)
		self.optimizer: optim.Optimizer =  self.get_optimizer()
		self.log = logging.getLogger()
		self.loss: nn.Module = kwargs.get( 'loss', self.get_loss() )
		self._checkpoint_manager: CheckpointManager = None
		self.start_batch: int = 0
		self.start_epoch: int = 0
		self.epoch_loss: float = 0.0
		self.epoch0: int = 0
		self.train_state = None
		self.global_time = None
		self.exec_stats = []

	def get_loss(self) -> nn.Module:
		if   "regression" in self.mtype:     return nn.L1Loss()
		elif "classification" in self.mtype: return nn.CrossEntropyLoss()
		else: raise RuntimeError( f"Unknown model type: {self.mtype}")

	def get_optimizer(self) -> optim.Optimizer:
		lr = self.cfg.lr
		if   self.cfg.optim == "rms":  return optim.RMSprop( self.model.parameters(), lr=lr )
		elif self.cfg.optim == "adam": return optim.Adam(    self.model.parameters(), lr=lr, weight_decay=self.cfg.weight_decay )
		else: raise RuntimeError( f"Unknown optimizer: {self.cfg.optim}")

	def initialize_checkpointing( self, version: str, init_version:Optional[str]=None ):
		self._checkpoint_manager = CheckpointManager( version, self.model, self.optimizer, self.cfg )
		if self.cfg.refresh_state:
			self._checkpoint_manager.clear_checkpoints()
			print("\n *** No checkpoint loaded: training from scratch *** \n")
		else:
			self.train_state = self._checkpoint_manager.load_checkpoint( init_version=init_version, update_model=True )
			self.epoch0      = self.train_state.get('epoch', 0)
			self.start_batch = self.train_state.get('batch', 0)
			self.start_epoch = int(self.epoch0)
			print(f"\n      Loading checkpoint from {self._checkpoint_manager.checkpoint_path()}: epoch={self.start_epoch}, batch={self.start_batch}\n")

	def load_checkpoint( self, version: str ):
		if version is not None:
			self._checkpoint_manager = CheckpointManager( version, self.model, self.optimizer, self.cfg )
			self.train_state = self._checkpoint_manager.load_checkpoint( update_model=True )
			self.epoch0      = self.train_state.get('epoch', 0)
			self.start_batch = self.train_state.get('batch', 0)
			self.start_epoch = int(self.epoch0)
			print(f"\n      Loading checkpoint from {self._checkpoint_manager.checkpoint_path()}: epoch={self.start_epoch}, batch={self.start_batch}\n")

	def conditionally_update_weights(self, loss: Tensor):
		if self.mode == TSet.Train:
			# print( f"  ---> Update weights({self.cfg.optim }:lr={self.cfg.lr:.4f}): loss = {loss.cpu().item():.3f} ")
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

	def encode_batch(self, batch: RDict) -> TRDict:
		self.log.debug( f"encode_batch: {list(batch.keys())}")
		t,y = batch.pop('t'), batch.pop('y')
		p: Tensor = torch.from_numpy(batch.pop('period')).to(self.device)
		z: Tensor = self.to_tensor(t,y)
		return dict( z=z, target=self.get_target(1/p), **batch )

	def get_octave(self, f: Tensor ) -> Tensor:
		octave = torch.floor( torch.log2(f/self.f0) ).to( torch.long )
		return octave

	def fold_by_octave(self, f: Tensor ) -> Tensor:
		octave = torch.floor(torch.log2(f / self.f0))
		octave_base_freq = self.f0 * torch.pow( 2, octave )
		return  f/octave_base_freq

	def get_target(self, f: Tensor ) -> Tensor:
		if "regression" in self.mtype:
			return self.fold_by_octave(f) if self.mtype.endswith("octave") else f
		elif "classification" in self.mtype:     
			return self.get_octave(f)
		else: raise RuntimeError( f"Unknown model type: {self.cfg.model_type}")

	def to_tensor(self, x: np.ndarray, y: np.ndarray) -> Tensor:
		with (self.device):
			Y: Tensor = torch.FloatTensor(y).to(self.device)
			X: Tensor = torch.FloatTensor(x).to(self.device)
			return torch.stack((X,Y), dim=1)

	def get_next_batch(self) -> Optional[TRDict]:
		while True:
			dset: RDict = self.loader.get_next_batch()
			if dset is not None:
				return self.encode_batch(dset)

	@property
	def mode(self) -> TSet:
		return TSet.Validation if self.cfg.mode.startswith("val") else TSet.Train

	@property
	def nepochs(self) -> int:
		return self.cfg.nepochs if self.training else 1

	@property
	def epoch_range(self) -> Tuple[int,int]:
		e0: int = self.start_epoch if (self.mode == TSet.Train) else 0
		return e0, e0+self.nepochs

	def set_train_status(self):
		self.loader.initialize()
		if self.mode == TSet.Train:
			self.model.train(True)

	@property
	def training(self) -> bool:
		return not self.cfg.mode.startswith("val")

	def test_model(self):
		print(f"SignalTrainer[{self.mode}]: , {self.nepochs} epochs, device={self.device}")
		with self.device:
			self.set_train_status()
			self.loader.init_epoch()
			batch: Optional[TRDict] = self.get_next_batch()
			result: Tensor = self.model(batch['z'])
			print( f" ** (batch{list(batch['z'].shape)}, target{list(batch['target'].shape)}) ->  result{list(result.shape)}")

	def compute(self,version,ckp_version=None):
		print(f"SignalTrainer[{self.mode}]: , {self.nepochs} epochs, device={self.device}")
		self.initialize_checkpointing(version,ckp_version)
		with self.device:
			for epoch in range(*self.epoch_range):
				te = time.time()
				self.set_train_status()
				self.loader.init_epoch()
				losses, log_interval, t0 = [], 50, time.time()
				try:
					for ibatch in range(0,sys.maxsize):
						t0 = time.time()
						batch = self.get_next_batch()
						target: Tensor = batch['target']
						if self.verbose: print(f"E-{epoch} B-{ibatch}: batch{shp(batch['z'])} target{shp(target)}")
						if batch['z'].shape[0] > 0:
							self.global_time = time.time()
							result: Tensor = self.model(batch['z']).squeeze()
							if result.squeeze().ndim > 0:
								rrange = [result.min().cpu().item(), result.max().cpu().item()]
								trange = [target.min().cpu().item(), target.max().cpu().item()]
								if self.verbose: check_nan('result',result)
								loss: Tensor =  self.loss( result, target )
								self.conditionally_update_weights(loss)
								lval = loss.cpu().item()
								losses.append(lval)
								if self.verbose:
									check_nan('loss', loss)
								print(f"E-{epoch} B-{ibatch} result{list(result.shape)}: loss={lval:.3f},  result-range: [{rrange[0]:.3f} -> {rrange[1]:.3f}], target-range: [{trange[0]:.3f} -> {trange[1]:.3f}]", flush=True)
								if ibatch % log_interval == 0:
									aloss = np.array(losses[-log_interval:])
									print(f"E-{epoch} B-{ibatch} loss={aloss.mean():.3f}, range=({aloss.min():.3f} -> {aloss.max():.3f}), dt/batch={elapsed(t0):.5f} sec")
									self._checkpoint_manager.save_checkpoint( epoch, ibatch )

				except StopIteration:
					print( f"Completed epoch {epoch} in {elapsed(te)/60:.5f} min, mean-loss= {np.array(losses).mean():.3f}")

				epoch_losses = np.array(losses)
				print(f" ------ Epoch Loss: mean={epoch_losses.mean():.3f}, median={np.median(epoch_losses):.3f}, range=({epoch_losses.min():.3f} -> {epoch_losses.max():.3f})")

	def test_learning(self,version,ckp_version=None):
		print(f"test_learning: mtype={self.mtype}")
		self.initialize_checkpointing(version,ckp_version)
		with self.device:
			self.set_train_status()
			self.loader.init_epoch()
			batch = self.get_next_batch()
			target: Tensor = batch['target']
			bdata: Tensor = batch['z']
			trange = [target.min().cpu().item(), target.max().cpu().item()]
			for iteration in range(50):
				result: Tensor = self.model(bdata,target).squeeze()
				rrange = [result.min().cpu().item(), result.max().cpu().item()]
				if self.verbose: check_nan('result',result)
				loss: Tensor =  self.loss( result, target )
				self.conditionally_update_weights(loss)
				if self.verbose: check_nan('loss', loss)
				print(f"I-{iteration}  result{list(result.shape)}: loss = {loss.cpu().item():.3f}, result-range: [{rrange[0]:.3f} -> {rrange[1]:.3f}], target-range: [{trange[0]:.3f} -> {trange[1]:.3f}]", flush=True)


	def evaluate(self, version: str = None):
		self.load_checkpoint(version)
		with self.device:
			self.loader.init_epoch()
			losses, log_interval = [], 50
			try:
				for ibatch in range(0, sys.maxsize):
					batch = self.get_next_batch()
					if batch['z'].shape[0] > 0:
						self.global_time = time.time()
						result: Tensor = self.model(batch['z'])
						if result.squeeze().ndim > 0:
							loss: Tensor = self.loss(result.squeeze(), batch['target'].squeeze())
							losses.append(loss.cpu().item())

			except StopIteration:
				val_losses = np.array(losses)
				print(f"       *** Validation Loss ({val_losses.size} batches): mean={val_losses.mean():.4f}, median={np.median(val_losses):.4f}, range=({val_losses.min():.4f} -> {val_losses.max():.4f})")

	def preprocess(self):
		with self.device:
			te = time.time()
			self.loader.initialize(TSet.Validation)
			self.loader.init_epoch()
			try:
				for ibatch in range(0,sys.maxsize):
					batch = self.get_next_batch()
			except StopIteration:
				print(f"Completed preprocess in {elapsed(te) / 60:.5f} min.")






