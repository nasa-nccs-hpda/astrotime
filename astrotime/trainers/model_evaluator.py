from typing import List, Optional, Dict, Type, Tuple, Union
from omegaconf import DictConfig
import xarray as xa
from astrotime.encoders.embedding import EmbeddingLayer
from astrotime.loaders.base import IterativeDataLoader, RDict
import time, sys, torch, logging, numpy as np
from torch import nn, optim, Tensor
from astrotime.trainers.checkpoints import CheckpointManager
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg
from astrotime.encoders.baseline import ValueEncoder
TRDict = Dict[str,Union[List[str],int,torch.Tensor]]

def tnorm(x: Tensor, dim: int=0) -> Tensor:
    m: Tensor = x.mean( dim=dim, keepdim=True)
    s: Tensor = torch.std( x, dim=dim, keepdim=True)
    return (x - m) / s

class ModelEvaluator(object):

    def __init__(self, cfg: DictConfig, version: str, loader: IterativeDataLoader, embedding: EmbeddingLayer, device ):
        self.embedding: EmbeddingLayer = embedding
        self.loader: IterativeDataLoader = loader
        self.cfg: DictConfig = cfg
        self.log = logging.getLogger()
        self.model: nn.Module = get_model_from_cfg( cfg.model, device, embedding )
        self.device = device
        self._target_freq = None
        self._model_freq = None
        self._checkpoint_manager = CheckpointManager( version, self.model, None, self.cfg.train )
        self.train_state = self._checkpoint_manager.load_checkpoint( update_model=True )

    @property
    def nelements(self) -> int:
        return self.loader.nelements

    @property
    def tname(self):
        return self.embedding.name

    @property
    def xdata(self) -> np.ndarray:
        return self.embedding.xdata.squeeze()

    def get_element(self, dset_idx: int, element: int) -> Optional[TRDict]:
        dset: Optional[RDict] = self.loader.get_element(dset_idx,element)
        return None if (dset is None) else self.encode_element(dset)

    def encode_element(self, batch: RDict) -> TRDict:
        p: Tensor = torch.FloatTensor(batch.pop('period')).to(self.device)
        z: Tensor = self.to_tensor(batch.pop('t'), batch.pop('y'))
        return dict( z=z, target=1/p, **batch )

    def to_tensor(self, x: np.ndarray, y: np.ndarray) -> Tensor:
        with (self.device):
            Y: Tensor = torch.FloatTensor(y).to(self.device)
            X: Tensor = torch.FloatTensor(x).to(self.device)
            Y = tnorm(Y, dim=1)
            return torch.stack((X,Y), dim=1)

    def evaluate(self, sector: int, element: int) -> np.ndarray:
        element: TRDict = self.get_element(sector, element)
        result: Tensor = self.model( element['z'] )
        self._target_freq = element['target'].cpu().item()
        self._model_freq  = result.cpu().item()
        return self.embedding.get_result()

    @property
    def target_frequency(self) -> float:
        return self._target_freq

    @property
    def model_frequency(self) -> float:
        return self._model_freq