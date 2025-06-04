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
    else:     return 1.0 / round(t/y)

def sH(h:float) -> str:
    if h > 1: return str(round(h))
    else:
        sh = round(1/h)
        return f"1/{sh}" if sh > 1 else str(sh)

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
            for ifile in range(0,self.loader.nfiles):
                self.loader.set_file(ifile)
                for ielem in range(0,self.loader.file_size):
                    element: Optional[TRDict] =  self.get_element(ielem)
                    if element is not None:
                        result: Tensor = self.model(element['z'])
                        y,t = result.item(), element['target']
                        h = harmonic(y,t)
                        loss: float = self.loss(y,t*h)
                        losses.append(loss)
                        if loss > 0.1:
                            print(f" * F-{ifile} Elem-{ielem}: yt=({y:.3f},{t*h:.3f},{t:.3f}), H= {sH(h)}, yLoss= {loss:.5f}")
            L: np.array = np.array(losses)
            print(f"Loss mean = {L.mean():.3f}, range=[{L.min():.3f} -> {L.max():.3f}]")


