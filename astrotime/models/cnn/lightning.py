import hydra, torch, math, os
import torchmetrics
from omegaconf import DictConfig
from torch import nn, Tensor
from astrotime.trainers.loss import ExpLoss, ExpU
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.encoders.lightning import SpectralProjection, embedding_space
import pytorch_lightning as PL

class PLSpectralCNN(PL.LightningModule):

	def __init__(self, cfg: DictConfig):
		super().__init__()
		self.cfg = cfg
		self.batch_size: int = self.cfg.data.batch_size
		self.embedding_space = embedding_space(cfg.transform)[1]
		self.embedding = SpectralProjection( cfg.transform, self.embedding_space )
		self.cnn: nn.Module = self.get_model_from_cfg()
		self.loss = ExpLoss(cfg.data)
		self.train_loss_avg = torchmetrics.MeanMetric()
		self.val_loss_avg   = torchmetrics.MeanMetric()
		self.save_hyperparameters('cfg')

	def ckpt_path(self, version: str ) -> Optional[str]:
		cpath = self.checkpoint_path( version, self.cfg.train)
		return cpath if os.path.exists(cpath) else None

	@classmethod
	def checkpoint_path( cls, version: str, cfg: DictConfig  ) -> str:
		cpath = f"{cfg.results_path}/checkpoints/{version}"
		os.makedirs(os.path.dirname(cpath), 0o777, exist_ok=True)
		return cpath + '.ckpt'

	@classmethod
	def load_saved_model( cls, version: str, cfg: DictConfig ) -> Optional['PLSpectralCNN']:
		ckpt_path = cls.checkpoint_path( version, cfg.train )
		return cls.load_from_checkpoint( ckpt_path ) if os.path.exists(ckpt_path) else None

	def forward(self, x: Tensor) -> Tensor:
		self.embedding.set_device( self.device )
		return self.cnn( self.embedding(x) )

	def add_cnn_block( self, model: nn.Sequential, nchannels: int, num_input_features: int) -> int:
		mcfg = self.cfg.model
		block_input_channels = num_input_features if (num_input_features > 0) else nchannels
		in_channels = block_input_channels
		out_channels = nchannels
		for iL in range(mcfg.num_cnn_layers):
			out_channels = out_channels + mcfg.cnn_expansion_factor
			model.append(nn.Conv1d(in_channels, out_channels, kernel_size=mcfg.kernel_size, stride=mcfg.stride, padding='same'))
			model.append(nn.ELU())
			in_channels = out_channels
		model.append(nn.ELU())
		model.append(nn.BatchNorm1d(out_channels))
		model.append(nn.MaxPool1d(mcfg.pool_size))
		return out_channels

	@classmethod
	def add_dense_block( cls, model: nn.Sequential, in_channels: int, hidden_channels: int, out_channels: int):
		model.append(nn.Flatten())
		model.append(nn.Linear(in_channels, hidden_channels))  # 64
		model.append(nn.ELU())
		model.append(nn.Linear(hidden_channels, out_channels))

	def get_model_from_cfg( self ) -> nn.Module:
		mtype, cfg, dcfg = self.cfg.model.mtype, self.cfg.model, self.cfg.data
		model: nn.Sequential = nn.Sequential()
		num_input_channels = self.embedding.output_channels
		cnn_channels = cfg.cnn_channels
		for iblock in range(cfg.num_blocks):
			cnn_channels = self.add_cnn_block( model, cnn_channels, num_input_channels)
			num_input_channels = -1
		reduced_series_len = self.embedding.output_series_length // int(math.pow(cfg.pool_size, cfg.num_blocks))
		self.add_dense_block( model, cnn_channels * reduced_series_len, cfg.dense_channels, cfg.out_channels)
		model.append(ExpU(dcfg))
		return model

	def configure_optimizers(self) -> torch.optim.Optimizer:
		tcfg = self.cfg.train
		if   tcfg.optim == "rms":  return torch.optim.RMSprop( self.parameters(), lr=tcfg.lr)
		elif tcfg.optim == "adam": return torch.optim.Adam( self.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
		else: raise RuntimeError(f"Unknown optimizer: {tcfg.optim}")

	def training_step(self, batch, batch_idx):
		binput: Tensor =  batch['input'].to( self.device, non_blocking=True )
		btarget: Tensor = batch['target'].to( self.device, non_blocking=True )
		boutput: Tensor = self.forward( binput )
		loss = self.loss( boutput, btarget )
		self.train_loss_avg.update(loss)
		self.log('train_loss', self.train_loss_avg.compute(), prog_bar=True, sync_dist=True, batch_size=self.batch_size)
		return loss

	def validation_step(self, batch, batch_idx):
		binput: Tensor =  batch['input'].to( self.device, non_blocking=True )
		btarget: Tensor = batch['target'].to( self.device, non_blocking=True )
		boutput: Tensor = self.forward( binput )
		loss = self.loss( boutput, btarget )
		self.val_loss_avg.update(loss)
		self.log("val_loss", self.val_loss_avg.compute(), prog_bar=True, sync_dist=True, batch_size=self.batch_size)
		return loss

	def on_train_epoch_start(self):
		self.train_loss_avg.reset()

	def on_validation_epoch_start(self):
		self.val_loss_avg.reset()

