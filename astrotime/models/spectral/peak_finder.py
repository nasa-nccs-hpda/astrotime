from torch import Tensor, device
from torch.nn.modules import Module
from astrotime.util.math import tnorm
import logging, torch
import time, sys, numpy as np
from omegaconf import DictConfig
from torch import nn
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.loaders.base import ElementLoader, RDict
TRDict = Dict[str,Union[List[str],int,torch.Tensor]]

def harmonic( y: float, t: float) -> float:
    if y > t: return round(y/t)
    else:     return 1 / round(t/y)

class SpectralPeakSelector(Module):

    def __init__(self, cfg: DictConfig, device: device, fspace: Tensor, feature: int = 0 ) -> None:
        super().__init__()
        self.requires_grad_(False)
        self.device: device = device
        self.cfg: DictConfig = cfg
        self.log = logging.getLogger()
        self.fspace = fspace
        self.feature: int = feature

    def forward(self, input: Tensor) -> Tensor:
        spectrum: Tensor = input[:,self.feature,:].squeeze()
        speak: Tensor = spectrum.argmax(dim=-1).squeeze()
        result = self.fspace[speak]
        return result

class Evaluator:

    def __init__(self, cfg: DictConfig, device: device, loader: ElementLoader, model: nn.Module, loss: nn.Module ) -> None:
        super().__init__()
        self.device: device = device
        self.cfg: DictConfig = cfg
        self.log = logging.getLogger()
        self.model = model
        self.loader = loader
        self.loss = loss

    def encode_element(self, element: RDict) -> TRDict:
        t,y,p = element.pop('t'), element.pop('y'), element.pop('p')
        z: Tensor = self.to_tensor(t,y)
        return dict( z=z, target=1/p, **element )

    def to_tensor(self, x: np.ndarray, y: np.ndarray) -> Tensor:
        with (self.device):
            Y: Tensor = torch.FloatTensor(y).unsqueeze(0).to(self.device)
            X: Tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)
            Y = tnorm(Y, dim=1)
            return torch.stack((X,Y), dim=1)

    def get_element(self,ibatch) -> Optional[TRDict]:
        element = self.loader.get_element(ibatch)
        return None if element is None else self.encode_element(element)

    def evaluate(self):
        self.cfg["mode"] = "val"
        with self.device:
            losses = []
            for ibatch in range(0, self.loader.nelements):
                element: Optional[TRDict] =  self.get_element(ibatch)
                if element is not None:
                    result: Tensor = self.model(element['z'])
                    y,t = result.item(), element['target']
                    h = harmonic(y,t)
                    loss: float = self.loss(y,t*h)
                    losses.append(loss)
                    print(f" * Batch-{ibatch}: H={h:.2f}, Loss = {loss:.3f}")
            L: np.array = np.array(losses)
            print(f"Loss mean = {L.mean():.3f}, range=[{L.min():.3f} -> {L.max():.3f}]")


