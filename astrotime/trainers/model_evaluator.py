from typing import List, Optional, Dict, Type, Tuple, Union
from omegaconf import DictConfig
import xarray as xa
from astrotime.encoders.embedding import EmbeddingLayer
from astrotime.loaders.base import IterativeDataLoader, RDict
import time, sys, torch, logging, numpy as np
from torch import nn, optim, Tensor
from astrotime.encoders.wavelet import WaveletAnalysisLayer, embedding_space
from astrotime.models.cnn.cnn_baseline import get_nn_model_from_cfg
from astrotime.encoders.baseline import ValueEncoder
TRDict = Dict[str,Union[List[str],int,torch.Tensor]]

# embedding_space_array, embedding_space_tensor = embedding_space(cfg.transform, device)
# self.wavelet: WaveletAnalysisLayer = WaveletAnalysisLayer('analysis', cfg.transform, embedding_space_tensor, device)

class ModelEvaluator(object):

    def __init__(self, cfg: DictConfig, loader: IterativeDataLoader, embedding: EmbeddingLayer, device ):
        self.encoder: ValueEncoder = ValueEncoder( cfg.transform, device )
        self.embedding: EmbeddingLayer = embedding
        self.loader: IterativeDataLoader = loader
        self.cfg: DictConfig = cfg
        self.model: nn.Module = get_nn_model_from_cfg( cfg.model, device, embedding.nfeatures, embedding.output_series_length )
        self.device = device

    def encode_element(self, element: RDict) -> TRDict:
        p: Tensor = torch.from_numpy(element.pop('p')).to(self.device)
        t, y = self.encoder.encode_batch(element.pop('t'), element.pop('y'))
        z: Tensor = torch.concat((t[:, None, :], y), dim=1)
        return dict( z=z, target=p, **element )

    def get_element(self, sector: int, element: int) -> Optional[TRDict]:
        element: RDict = self.loader.get_element(sector, element)
        return self.encode_element(element)

    def evaluate(self, sector: int, element: int) -> Tuple[np.ndarray,float]:
        element: TRDict = self.get_element(sector, element)
        embedding: Tensor = self.embedding.forward(element['z'])
        result: Tensor = self.model(embedding)
        return embedding.detach().cpu().numpy(), result.detach().cpu().item()