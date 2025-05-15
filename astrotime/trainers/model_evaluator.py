from typing import List, Optional, Dict, Type, Tuple, Union
from omegaconf import DictConfig
import xarray as xa
from astrotime.encoders.embedding import EmbeddingLayer
from astrotime.loaders.base import IterativeDataLoader, RDict
import time, sys, torch, logging, numpy as np
from torch import nn, optim, Tensor
from astrotime.trainers.checkpoints import CheckpointManager
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg
from astrotime.loaders.MIT import MITLoader
from astrotime.encoders.baseline import ValueEncoder
TRDict = Dict[str,Union[List[str],int,torch.Tensor]]

class ModelEvaluator(object):

    def __init__(self, cfg: DictConfig, version: str, loader: MITLoader, embedding: EmbeddingLayer, device ):
        self.encoder: ValueEncoder = ValueEncoder( cfg.transform, device )
        self.embedding: EmbeddingLayer = embedding
        self.loader: MITLoader = loader
        self.cfg: DictConfig = cfg
        self.model: nn.Module = get_model_from_cfg( cfg.model, device, embedding )
        self.device = device
        self._target_freq = None
        self._model_freq = None
        self._checkpoint_manager = CheckpointManager( version, self.model, None, self.cfg.train )
        self.train_state = self._checkpoint_manager.load_checkpoint( update_model=True )

    @property
    def tname(self):
        return self.embedding.name

    def TICS(self, sector_index: int) -> List[str]:
        return self.loader.TICS(sector_index)

    @property
    def xdata(self) -> np.ndarray:
        return self.embedding.xdata.squeeze()

    def to_tensor(self, x: float|np.ndarray) -> Tensor:
        t: Tensor = Tensor(x) if type(x)==float else torch.from_numpy(x)
        return t.to(self.device)

    def encode_element(self, element: RDict) -> TRDict:
        t, y = self.encoder.encode_batch(element.pop('t'), element.pop('y'))
        z: Tensor = torch.concat((t[:, None, :], y), dim=1)
        return dict( z=z, **element )

    def get_element(self, sector: int, element: int) -> Optional[TRDict]:
        element: RDict = self.loader.get_element(sector, element)
        return self.encode_element(element)

    def get_model_result(self, element: TRDict) -> Tensor:
        return self.cfg.base_freq * torch.pow( 2.0, self.model( element['z'] ) )

    def evaluate(self, sector: int, element: int) -> np.ndarray:
        element: TRDict = self.get_element(sector, element)
        embedding: np.ndarray = self.embedding.get_result()
        self._target_freq = 1/element['p']
        self._model_freq  = self.get_model_result(element).cpu().item()
        return embedding

    @property
    def target_frequency(self) -> float:
        return self._target_freq

    @property
    def model_frequency(self) -> float:
        return self._model_freq