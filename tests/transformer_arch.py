import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
import timesfm

# ----- 1. Dataset -----
class LightCurveDataset(Dataset):
    def __init__(self, df):
        self.series = df["series"].tolist()
        self.periods = df["period"].values.astype(np.float32)

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        x = torch.tensor(self.series[idx], dtype=torch.float32)
        y = torch.tensor(self.periods[idx], dtype=torch.float32)
        return x, y

# ----- 2. Load Model -----
checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch")
hparams = timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=16)
tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

# ----- 3. Load Data -----
# Assume your DataFrame has columns: series (list of floats), period (float)
df = pd.read_pickle("lightcurves.pkl")  # Preprocessed: each row is a padded sequence
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

train_dataset = LightCurveDataset(train_df)
val_dataset = LightCurveDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ----- 4. Define a simple regressor on TimesFM embeddings -----
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

model = PeriodRegressor(embed_dim=1280).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.L1Loss()  # MAE

# ----- 5. Training Loop -----
def get_embedding(tfm, series_batch):
    # Convert list of numpy arrays into batched tensor
    padded = torch.nn.utils.rnn.pad_sequence(series_batch, batch_first=True)
    padded = padded.cuda()
    # Forecast is not used — we only use embeddings
    _, embeddings = tfm.forecast(padded, freq=[0]*len(series_batch), return_embedding=True)
    return embeddings

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
    print(f"[Epoch {epoch+1}] Train Loss: {np.mean(train_losses):.4f} | Val MAE: {mae:.4f}")
