from typing import List, Optional, Dict, Type, Tuple, Union
from omegaconf import DictConfig
from astrotime.encoders.wavelet import WaveletAnalysisLayer, embedding_space
from astrotime.encoders.embedding import EmbeddingLayer
from astrotime.loaders.base import RDict, ElementLoader
import time, sys, torch, logging, numpy as np
from torch import nn, optim, Tensor
from astrotime.models.spectral.peak_finder import SpectralPeakSelector
from astrotime.trainers.checkpoints import CheckpointManager
from astrotime.models.cnn.cnn_baseline import get_model_from_cfg, ExpU
from astrotime.encoders.correlation import AutoCorrelationLayer

TRDict = Dict[str,Union[List[str],int,torch.Tensor]]

def tnorm(x: Tensor, dim: int=0) -> Tensor:
    m: Tensor = x.mean( dim=dim, keepdim=True)
    s: Tensor = torch.std( x, dim=dim, keepdim=True)
    return (x - m) / s

class ModelEvaluator(object):

    def __init__(self, cfg: DictConfig, version: str, loader: ElementLoader, device, **kwargs ):
        espace = embedding_space(cfg.transform, device)
        self.freqspace: np.ndarray = espace[0]
        self.loader: ElementLoader = loader
        self.cfg: DictConfig = cfg
        self.log = logging.getLogger()
        self.mtype = kwargs.get( 'mtype', "cnn" )
        self.peak_selector: SpectralPeakSelector = None
        if self.mtype == "peakfinder":
            self.embedding: EmbeddingLayer = WaveletAnalysisLayer('analysis', cfg.transform, espace[1], device)
            self.peak_selector = SpectralPeakSelector( cfg.transform, device, self.embedding.xdata )
            self.model: nn.Module =nn.Sequential( self.embedding, self.peak_selector ).to(device)
        elif self.mtype == "cnn":
            self.embedding: EmbeddingLayer = WaveletAnalysisLayer('analysis', cfg.transform, espace[1], device)
            self.model: nn.Module = get_model_from_cfg( cfg.model, device, self.embedding, ExpU(cfg.data) )
        elif self.mtype == "autocor":
            self.embedding: EmbeddingLayer = AutoCorrelationLayer('autocor', cfg.transform, espace[1], device)
            self.model: nn.Module = get_model_from_cfg( cfg.model, device, self.embedding, ExpU(cfg.data) )
        self.device = device
        self._target_freq = None
        self._model_freq = None
        if self.mtype == "cnn":
            self.optimizer = self.get_optimizer()
            self._checkpoint_manager = CheckpointManager( version, self.model, self.optimizer, self.cfg.train )
            self.train_state = self._checkpoint_manager.load_checkpoint( update_model=True )

    def get_optimizer(self) -> optim.Optimizer:
        cfg = self.cfg.train
        if   cfg.optim == "rms":  return optim.RMSprop( self.model.parameters(), lr=cfg.lr )
        elif cfg.optim == "adam": return optim.Adam(    self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay )
        else: raise RuntimeError( f"Unknown optimizer: {cfg.optim}")

    def process_event( self, id: str, key: str, ax=None, **kwargs ):
        if id == "KeyEvent":
            if self.peak_selector is not None:
                self.peak_selector.process_key_event( key )
                return self.peak_selector.feature
        return None

    @property
    def nelements(self) -> int:
        return self.loader.nelements

    @property
    def tname(self):
        return self.embedding.name

    @property
    def xdata(self) -> Tensor:
        return self.embedding.xdata.squeeze()

    def get_element(self, element: int) -> Optional[TRDict]:
        dset: Optional[RDict] = self.loader.get_element(element)
        return None if (dset is None) else self.encode_element(dset)

    def encode_element(self, batch: RDict) -> TRDict:
        p: float = batch.pop('p') if ('p' in batch) else batch.pop('period')
        z: Tensor = self.to_tensor(batch.pop('t'), batch.pop('y'))
        return dict( z=z, target=1/p, **batch )

    def to_tensor(self, x: np.ndarray, y: np.ndarray) -> Tensor:
        with (self.device):
            Y: Tensor = torch.FloatTensor(y).to(self.device)
            X: Tensor = torch.FloatTensor(x).to(self.device)
            return torch.stack((X,tnorm(Y)), dim=0).unsqueeze(0)

    def evaluate(self, element: int) -> np.ndarray:
        element: TRDict = self.get_element( element)
        result: Tensor = self.model( element['z'] )
        self._target_freq = element['target']
        self._model_freq  = result.cpu().item()
        return self.embedding.get_result()

    @property
    def target_frequency(self) -> float:
        return self._target_freq

    @property
    def model_frequency(self) -> float:
        return self._model_freq