import numpy as np
from sklearn.metrics import mean_absolute_error
from timesfm import TimesFmHparams, TimesFmCheckpoint, TimesFm
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import hydra, torch
from omegaconf import DictConfig
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
		self.tfm: TimesFm = self.get_tfm()
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

	# def encode_batch(self, batch: RDict) -> TRDict:
	# 	t, y = batch.pop('t'), batch.pop('y')
	# 	p: Tensor = torch.from_numpy(batch.pop('period')).cuda()
	# 	o = batch.pop('octave', None)
	# 	if o is not None: o = torch.from_numpy(o).cuda()
	# 	z: Tensor = self.to_tensor(t, y)
	# 	return dict(z=z, target=1 / p, octave=o, **batch)
	#
	# def to_tensor(self, x: np.ndarray, y: np.ndarray) -> Tensor:
	# 	Y: Tensor = torch.FloatTensor(y).cuda()
	# 	X: Tensor = torch.FloatTensor(x).cuda()
	# 	return torch.stack((X, Y), dim=1)

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
			for batch in self.train_loader:
				x: Tensor = batch['z']
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
					x: Tensor = batch['z']
					y: Tensor = batch['target']
					emb = self.get_embedding(x)
					pred = self.model(emb)
					val_preds.extend(pred.cpu().numpy())
					val_targets.extend(y.cpu().numpy())

			mae = mean_absolute_error(val_targets, val_preds)
			print(f"[Epoch {epoch + 1}] Train Loss: {np.mean(train_losses):.4f} | Val MAE: {mae:.4f}")

