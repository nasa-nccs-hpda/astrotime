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

class ModelEvaluator(object):

    def __init__(self, cfg: DictConfig, version: str, loader: IterativeDataLoader, embedding: EmbeddingLayer, device ):
        self.encoder: ValueEncoder = ValueEncoder( cfg.transform, device )
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

    def to_tensor(self, x: float|np.ndarray) -> Tensor:
        t: Tensor = Tensor(x) if type(x)==float else torch.from_numpy(x)
        return t.to(self.device)

    def encode_batch(self, batch: RDict) -> TRDict:
        self.log.debug( f"encode_batch: {list(batch.keys())}")
        p: Tensor = torch.from_numpy(batch.pop('period')).to(self.device)
        t, y = self.encoder.encode_batch(batch.pop('t'), batch.pop('y'))
        z: Tensor = torch.concat((t[:, None, :], y), dim=1)
        return dict( z=z, target=1/p, **batch )

    def get_batch(self, dset_idx: int, ibatch: int) -> Optional[TRDict]:
        dset: Optional[RDict] = self.loader.get_batch(dset_idx,ibatch)
        return None if (dset is None) else self.encode_batch(dset)

    def get_model_result(self, element: TRDict) -> Tensor:
        return self.cfg.base_freq * torch.pow( 2.0, self.model( element['z'] ) )

    def evaluate(self, sector: int, element: int) -> np.ndarray:
        element: TRDict = self.get_batch(sector, element)
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