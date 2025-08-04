import numpy as np
from sklearn.metrics import mean_absolute_error
import timesfm
import hydra, torch
from omegaconf import DictConfig
from torch import nn
from astrotime.util.series import TSet
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.datasets.sinusoids import SinusoidDataLoader
from astrotime.config.context import astrotime_initialize
RDict = Dict[str,Union[List[str],int,np.ndarray]]
TRDict = Dict[str,Union[List[str],int,torch.Tensor]]

version = "sinusoid_period"

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

def get_embedding(tfm, series_batch):
    # Convert list of numpy arrays into batched tensor
    padded = torch.nn.utils.rnn.pad_sequence(series_batch, batch_first=True)
    padded = padded.cuda()
    # Forecast is not used — we only use embeddings
    _, embeddings = tfm.forecast(padded, freq=[0] * len(series_batch), return_embedding=True)
    return embeddings

@hydra.main(version_base=None, config_path="../../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
    astrotime_initialize( cfg, version )

    checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch")
    hparams = timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=16)
    tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

    train_loader = SinusoidDataLoader( cfg, TSet.Train )
    val_loader = SinusoidDataLoader( cfg, TSet.Validation )

    model = PeriodRegressor(embed_dim=1280).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.L1Loss()  # MAE

    for epoch in range(10):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = [s.cuda() for s in x]
            y = y.cuda()
            with torch.no_grad():
                emb = get_embedding(tfm, x)
            pred = model(emb)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = [s.cuda() for s in x]
                y = y.cuda()
                emb = get_embedding(tfm, x)
                pred = model(emb)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        mae = mean_absolute_error(val_targets, val_preds)
        print(f"[Epoch {epoch + 1}] Train Loss: {np.mean(train_losses):.4f} | Val MAE: {mae:.4f}")

