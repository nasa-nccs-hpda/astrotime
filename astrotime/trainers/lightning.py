import os
from omegaconf import DictConfig, OmegaConf
import logging
import torch
from torch import nn
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg
from astrotime.encoders.lightning import SpectralProjection, embedding_space
from astrotime.trainers.checkpoints import CheckpointManager
import torchmetrics
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class LightningTrainer(pl.LightningModule):
	def __init__(self,  cfg: DictConfig):
		super(LightningTrainer, self).__init__()
		self.cfg = cfg

		self.embedding_space_tensor = embedding_space(cfg.transform)[1]
		self.embedding = SpectralProjection( cfg.transform, self.embedding_space_tensor )
		self.model: nn.Module = get_model_from_cfg(cfg, self.embedding )
		# if self.cfg.MODEL.PRETRAINED:
		#    self.load_checkpoint()

		self.batch_size = cfg.data.batch_size
		self.num_workers = cfg.data.num_workers
		self.pin_memory = cfg.data.pin_memory

		# Training Metrics
		self.train_loss_avg = torchmetrics.MeanMetric()

		# Validation Metrics
		self.val_loss_avg = torchmetrics.MeanMetric()

	def initialize_checkpointing(self, version: str):
		self._checkpoint_manager = CheckpointManager(version, self.model, self.optimizer, self.cfg)
		if self.cfg.refresh_state:
			self._checkpoint_manager.clear_checkpoints()
			print("\n *** No checkpoint loaded: training from scratch *** \n")
		else:
			init_ckp_version = self.cfg.get('ckp_version', None)
			self.train_state = self._checkpoint_manager.load_checkpoint(init_version=init_ckp_version, update_model=True)
			self.epoch0 = self.train_state.get('epoch', 0)
			self.start_batch = self.train_state.get('batch', 0)
			self.start_epoch = int(self.epoch0)

	def load_checkpoint(self, version: str):
		if version is not None:
			self.optimizer = self.get_optimizer()
			self._checkpoint_manager = CheckpointManager(version, self.model, self.optimizer, self.cfg)
			self.train_state = self._checkpoint_manager.load_checkpoint(update_model=True)
			self.epoch0 = self.train_state.get('epoch', 0)
			self.start_batch = self.train_state.get('batch', 0)
			self.start_epoch = int(self.epoch0)
			print(f"\n      Loading checkpoint from {self._checkpoint_manager.checkpoint_path()}: epoch={self.start_epoch}, batch={self.start_batch}\n")


	@classmethod
	def load_checkpoint(cls, ckpt_path, config):
		"""
		Load model from either a Lightning .ckpt file or a DeepSpeed .pt checkpoint.
		"""
		if ckpt_path.endswith(".pt") or "mp_rank" in ckpt_path:
			logging.info(f"Loading DeepSpeed checkpoint: {ckpt_path}")
			model = cls(config)

			checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
			state_dict = checkpoint.get("module", checkpoint)

			# Strip "model." prefix if present
			cleaned_state_dict = {
				k.replace("model.", "", 1) if k.startswith("model.") else k: v
				for k, v in state_dict.items()
			}

			missing_keys, unexpected_keys = model.model.load_state_dict(
				cleaned_state_dict, strict=False
			)

			logging.info(f"Loaded DeepSpeed weights with {len(missing_keys)} missing and {len(unexpected_keys)} unexpected keys.")
			return model

		else:
			logging.info(f"Loading Lightning checkpoint: {ckpt_path}")
			return cls.load_from_checkpoint(ckpt_path, config=config)


	def forward(self, samples, timestamps):
		return self.model(samples, timestamps, mask_ratio=self.config.DATA.MASK_RATIO)

	def training_step(self, batch, batch_idx):
		samples, timestamps = batch
		input = samples.to(self.device, non_blocking=True)
		timestamps = timestamps.to(self.device, non_blocking=True)

		loss, pred, mask = self.forward(samples, timestamps)
		self.train_loss_avg.update(loss)

		# Compute reconstruction metrics
		B, T, C, H, W = samples.shape
		pred_imgs = self.model.unpatchify(pred, T, H, W)
		pred_imgs = torch.clamp(pred_imgs, 0, 1)
		target_imgs = samples

		psnr_val = self.train_psnr(pred_imgs[:, 0], target_imgs[:, 0])
		ssim_val = self.train_ssim(pred_imgs[:, 0], target_imgs[:, 0])

		# Log metrics
		self.log("train_loss", self.train_loss_avg.compute(), prog_bar=True, batch_size=self.batch_size)
		self.log("train_psnr", psnr_val, prog_bar=True, batch_size=self.batch_size)
		self.log("train_ssim", ssim_val, prog_bar=True, batch_size=self.batch_size)

		return loss

	def validation_step(self, batch, batch_idx):
		samples, timestamps = batch
		samples = samples.to(self.device, non_blocking=True)
		timestamps = timestamps.to(self.device, non_blocking=True)

		loss, pred, mask = self.forward(samples, timestamps)
		self.val_loss_avg.update(loss)

		# Compute reconstruction metrics
		B, T, C, H, W = samples.shape
		pred_imgs = self.model.unpatchify(pred, T, H, W)
		pred_imgs = torch.clamp(pred_imgs, 0, 1)
		target_imgs = samples

		psnr_val = self.val_psnr(pred_imgs[:, 0], target_imgs[:, 0])
		ssim_val = self.val_ssim(pred_imgs[:, 0], target_imgs[:, 0])

		# Log metrics
		self.log("val_loss", self.val_loss_avg.compute(), prog_bar=True, sync_dist=True, batch_size=self.batch_size)
		self.log("val_psnr", psnr_val, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
		self.log("val_ssim", ssim_val, prog_bar=True, sync_dist=True, batch_size=self.batch_size)

		return loss

	def configure_optimizers(self):
		optimizer = build_optimizer(self.config, self.model, is_pretrain=True)

		# Compute total steps
		total_steps = self.trainer.estimated_stepping_batches
		warmup_steps = int(self.config.TRAIN.WARMUP_EPOCHS * (total_steps / self.config.TRAIN.EPOCHS))
		cosine_steps = total_steps - warmup_steps

		# Warmup scheduler
		warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
			optimizer,
			start_factor=1e-6,
			end_factor=1.0,
			total_iters=warmup_steps
		)

		# Cosine scheduler
		cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer,
			T_max=cosine_steps,
			eta_min=self.config.TRAIN.MIN_LR
		)

		# Combine schedulers
		scheduler = torch.optim.lr_scheduler.SequentialLR(
			optimizer,
			schedulers=[warmup_scheduler, cosine_scheduler],
			milestones=[warmup_steps]
		)

		return {
			"optimizer": optimizer,
			"lr_scheduler": {
				"scheduler": scheduler,
				"interval": "step",
				"frequency": 1,
			}
		}

	def on_train_epoch_start(self):
		self.train_loss_avg.reset()
		self.train_psnr.reset()
		self.train_ssim.reset()

	def on_validation_epoch_start(self):
		self.val_loss_avg.reset()
		self.val_psnr.reset()
		self.val_ssim.reset()
