import numpy as np
from sklearn.metrics import mean_absolute_error
import timesfm
import hydra, torch
from omegaconf import DictConfig
from torch import nn
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

	def __init__( self, cfg: DictConfig, train_loader, val_loader ):
		self.cfg = cfg
		self.checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch")
		self.hparams = timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=16)
		self.tfm = timesfm.TimesFm(hparams=self.hparams, checkpoint=self.checkpoint)
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.model = PeriodRegressor(embed_dim=1280).cuda()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
		self.loss_fn = nn.L1Loss()  # MAE

	def get_embedding( self, series_batch):
		# Convert list of numpy arrays into batched tensor
		padded = torch.nn.utils.rnn.pad_sequence(series_batch, batch_first=True)
		padded = padded.cuda()
		# Forecast is not used — we only use embeddings
		_, embeddings = self.tfm.forecast(padded, freq=[0] * len(series_batch), return_embedding=True)
		return embeddings

	def train(self, version: str ):
		for epoch in range(10):
			self.model.train()
			train_losses = []
			for x, y in self.train_loader:
				x = [s.cuda() for s in x]
				y = y.cuda()
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
				for x, y in self.val_loader:
					x = [s.cuda() for s in x]
					y = y.cuda()
					emb = self.get_embedding(x)
					pred = self.model(emb)
					val_preds.extend(pred.cpu().numpy())
					val_targets.extend(y.cpu().numpy())

			mae = mean_absolute_error(val_targets, val_preds)
			print(f"[Epoch {epoch + 1}] Train Loss: {np.mean(train_losses):.4f} | Val MAE: {mae:.4f}")

