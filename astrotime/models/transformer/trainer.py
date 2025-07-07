from typing import List, Optional, Dict, Type, Tuple, Union
from omegaconf import DictConfig
from astrotime.trainers.checkpoints import CheckpointManager
from astrotime.util.tensor_ops import check_nan
from astrotime.util.math import shp
from astrotime.encoders.wavelet import embedding_space
from astrotime.trainers.loss import ExpLoss, ExpU
from astrotime.loaders.base import Loader, RDict
from .spectral import SpectralProjection
from .attention import MultiHeadAttention
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

	def __init__(self, cfg: DictConfig, device: torch.device, loader: Loader ):
		self.cfg: DictConfig = cfg.train
		self.device: torch.device = device
		self.mtype: str = cfg.model.task
		self.noctaves = cfg.data.noctaves
		self.f0 = cfg.data.base_freq
		self.embedding_space_array, self.embedding_space_tensor = embedding_space(cfg.transform, device)
		self.loader: Loader = loader
		self.embedding = SpectralProjection('spectral_projection', cfg.transform, self.embedding_space_tensor, device )
		self.model: nn.Module = self.get_model(cfg.model)
		self.optimizer: optim.Optimizer = None
		self.log = logging.getLogger()
		self.loss: nn.Module = self.get_loss()
		self._checkpoint_manager: CheckpointManager = None
		self.start_batch: int = 0
		self.start_epoch: int = 0
		self.epoch_loss: float = 0.0
		self.epoch0: int = 0
		self.train_state = None
		self.global_time = None
		self.exec_stats = []

	def get_loss(self) -> nn.Module:
		if   self.mtype.startswith("regression"): return nn.L1Loss()
		elif self.mtype.startswith("classification"): return nn.CrossEntropyLoss()
		else: raise RuntimeError( f"Unknown model type: {self.mtype}")

	def get_model(self, cfg: DictConfig, activation: nn.Module = None ) -> nn.Module:
		modules: List[nn.Module] = [ self.embedding ]
		if   self.mtype.startswith("regression"): result_dim = 1
		elif self.mtype.startswith("classification"): result_dim = self.noctaves
		else: raise RuntimeError( f"Unknown model type: {self.mtype}" )
		for iL in range(1, cfg.nlayers+1):
			input_size = self.embedding.nfreq_oct if (iL == 1) else cfg.E_internal
			output_size = result_dim if (iL == cfg.nlayers) else cfg.E_internal
			modules.append( MultiHeadAttention( cfg, self.device, input_size, output_size) )
		if activation is not None:
			modules.append( activation.to(self.device) )
		return nn.Sequential(*modules)

	def get_optimizer(self) -> optim.Optimizer:
		if   self.cfg.optim == "rms":  return optim.RMSprop( self.model.parameters(), lr=self.cfg.lr )
		elif self.cfg.optim == "adam": return optim.Adam(    self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay )
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
			self.optimizer = self.get_optimizer()
			self._checkpoint_manager = CheckpointManager( version, self.model, self.optimizer, self.cfg )
			self.train_state = self._checkpoint_manager.load_checkpoint( update_model=True )
			self.epoch0      = self.train_state.get('epoch', 0)
			self.start_batch = self.train_state.get('batch', 0)
			self.start_epoch = int(self.epoch0)
			print(f"\n      Loading checkpoint from {self._checkpoint_manager.checkpoint_path()}: epoch={self.start_epoch}, batch={self.start_batch}\n")

	def conditionally_update_weights(self, loss: Tensor):
		if self.mode == TSet.Train:
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
		octave = torch.floor( torch.log2(f/self.f0) )
		return octave

	def fold_by_octave(self, f: Tensor ) -> Tensor:
		octave = torch.floor(torch.log2(f / self.f0))
		octave_base_freq = self.f0 * torch.pow( 2, octave )
		return  f/octave_base_freq

	def get_target(self, f: Tensor ) -> Tensor:
		if self.mtype.startswith("regression"):
			return self.fold_by_octave(f) if self.mtype.endswith("octave") else f
		elif self.mtype.startswith("classification"):
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
		self.loader.initialize(self.mode)
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
		self.optimizer = self.get_optimizer()
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
						self.log.debug(f"E-{epoch} B-{ibatch}: batch{shp(batch['z'])} target{shp(batch['target'])}")
						if batch['z'].shape[0] > 0:
							self.global_time = time.time()
							result: Tensor = self.model(batch['z'])
							if result.squeeze().ndim > 0:
								rrange = [ result.min().cpu().item(), result.max().cpu().item() ]
								self.log.debug(f"result{list(result.shape)} range: [{rrange[0]:.3f} -> {rrange[1]:.3f}]")
								loss: Tensor =  self.loss( result, batch['target'] )
								self.conditionally_update_weights(loss)
								lval = loss.cpu().item()
								losses.append(lval)
								print(f"E-{epoch} B-{ibatch} loss={lval:.3f}, range: [{rrange[0]:.6f} -> {rrange[1]:.3f}]", flush=True)
								if ibatch % log_interval == 0:
									aloss = np.array(losses[-log_interval:])
									print(f"E-{epoch} B-{ibatch} loss={aloss.mean():.3f}, range=({aloss.min():.3f} -> {aloss.max():.3f}), dt/batch={elapsed(t0):.5f} sec")
									self._checkpoint_manager.save_checkpoint( epoch, ibatch )

				except StopIteration:
					print( f"Completed epoch {epoch} in {elapsed(te)/60:.5f} min, mean-loss= {np.array(losses).mean():.3f}")

				epoch_losses = np.array(losses)
				print(f" ------ Epoch Loss: mean={epoch_losses.mean():.3f}, median={np.median(epoch_losses):.3f}, range=({epoch_losses.min():.3f} -> {epoch_losses.max():.3f})")

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






