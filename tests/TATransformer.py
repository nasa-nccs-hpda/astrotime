import numpy as np
import timesfm
import hydra, torch
from omegaconf import DictConfig
from astrotime.util.series import TSet
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.datasets.sinusoids import SinusoidDataLoader
from astrotime.config.context import astrotime_initialize
from astrotime.trainers.timefm import TimeFMTrainer

RDict = Dict[str,Union[List[str],int,np.ndarray]]
TRDict = Dict[str,Union[List[str],int,torch.Tensor]]

version = "sinusoid_period"

@hydra.main(version_base=None, config_path="../config", config_name=version)
def my_app(cfg: DictConfig) -> None:
    astrotime_initialize( cfg, version )

    train_loader = SinusoidDataLoader( cfg, TSet.Train )
    val_loader = SinusoidDataLoader( cfg, TSet.Validation )

    trainer = TimeFMTrainer( cfg, train_loader, val_loader )
    trainer.train( version)

if __name__ == "__main__":
    my_app()
