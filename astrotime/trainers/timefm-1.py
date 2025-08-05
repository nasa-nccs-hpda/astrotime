import numpy as np
from sklearn.metrics import mean_absolute_error
from timesfm import TimesFmHparams, TimesFmCheckpoint, TimesFm
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import hydra, torch
from omegaconf import DictConfig
from  timesfm.pytorch_patched_decoder import ResidualBlock, StackedDecoder
from torch import nn, Tensor
from typing import List, Optional, Dict, Type, Union, Tuple
RDict = Dict[str,Union[List[str],int,np.ndarray]]
TRDict = Dict[str,Union[List[str],int,torch.Tensor]]

class PeriodRegressor(nn.Module):
	def __init__(self, embed_dim):
		super().__init__()
		self.regressor = nn.Sequential(
			nn.Linear(embed_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 1)
		)

	def forward(self, x):
		return self.regressor(x).squeeze()


class TimeFMTrainer(object):

	def __init__( self, cfg: DictConfig, train_loader: DataLoader, val_loader: DataLoader ):
		self.cfg = cfg

		self.input_ff_layer = ResidualBlock(
			input_dims=2 * self.cfg.patch_len,
			output_dims=self.cfg.hidden_size,
			hidden_dims=self.cfg.intermediate_size,
		)
		self.output_ff_layer = ResidualBlock(
			input_dims=self.cfg.hidden_size,
			output_dims=self.cfg.horizon_len * (1 + len(self.cfg.quantiles)),
			hidden_dims=self.cfg.intermediate_size,
		)
		self.stacked_transformer = StackedDecoder(
			hidden_size=self.cfg.hidden_size,
			intermediate_size=self.cfg.intermediate_size,
			num_heads=self.cfg.num_heads,
			num_kv_heads=self.cfg.num_kv_heads,
			head_dim=self.cfg.head_dim,
			num_layers=self.cfg.num_layers,
			rms_norm_eps=self.cfg.rms_norm_eps,
		)

		self.train_loader: DataLoader = train_loader
		self.val_loader: DataLoader = val_loader
		self.model: nn.Module = PeriodRegressor(embed_dim=1280).cuda()
		self.optimizer: Optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
		self.loss_fn: nn.Module = nn.L1Loss()  # MAE

	def get_tfm(self ) -> TimesFm:
		checkpoint = TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch")
		hparams = TimesFmHparams(
			backend="gpu",
			per_core_batch_size=32,
			horizon_len=128,
			num_layers=50,
			use_positional_embedding=False,
			context_len=2048,
		)
		return TimesFm( hparams=hparams, checkpoint=checkpoint )

	def get_embedding( self, series_batch):
		model_input = self.input_ff_layer(series_batch)
		transformed = self.stacked_transformer(model_input)
		embedding = self.output_ff_layer(transformed)
		return embedding

	def train(self, version: str ):
		for epoch in range(10):
			self.model.train()
			train_losses = []
			for batch in self.train_loader:
				x: Tensor = batch['input']
				y: Tensor = batch['target']
				with torch.no_grad():
					emb = self.get_embedding(x)
				pred = self.model(emb)
				loss = self.loss_fn(pred, y)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				train_losses.append(loss.item())

			# Validation
			self.model.eval()
			val_preds, val_targets = [], []
			with torch.no_grad():
				for batch in self.val_loader:
					x: Tensor = batch['input']
					y: Tensor = batch['target']
					emb = self.get_embedding(x)
					pred = self.model(emb)
					val_preds.extend(pred.cpu().numpy())
					val_targets.extend(y.cpu().numpy())

			mae = mean_absolute_error(val_targets, val_preds)
			print(f"[Epoch {epoch + 1}] Train Loss: {np.mean(train_losses):.4f} | Val MAE: {mae:.4f}")

