import hydra, torch
from omegaconf import DictConfig
from torch import nn
from astrotime.util.series import TSet
from typing import List, Optional, Dict, Type, Union, Tuple
import pytorch_lightning as PL
from astrotime.models.cnn.lightning import PLSpectralCNN
from astrotime.config.context import astrotime_initialize
from astrotime.datasets.sinusoids import SinusoidDataLoader
version = "sinusoid_period"

@hydra.main(version_base=None, config_path="../../../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
	astrotime_initialize( cfg, version )
	accelerator = "gpu" if torch.cuda.is_available() else "cpu"
	nnodes = cfg.train.ngpu if accelerator=="gpu" else 1

	train_loader = SinusoidDataLoader(cfg.data, TSet.Train)
	val_loader   = SinusoidDataLoader(cfg.data, TSet.Validation)
	model = PLSpectralCNN(cfg)
	trainer = PL.Trainer(accelerator=accelerator, max_epochs=10, num_nodes=nnodes )

	print( f"Training {version}...")
	trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=model.ckpt_path(version) )
	print(f"DONE")

if __name__ == "__main__":
	my_app()
