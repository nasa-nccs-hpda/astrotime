from torch import Tensor, device
from torch.nn.modules import Module
from astrotime.util.math import tnorm
import logging, torch
import time, sys, numpy as np
from omegaconf import DictConfig
from torch import nn
from astrotime.util.math import shp
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.loaders.base import ElementLoader, RDict
from astrotime.trainers.loss import HLoss
TRDict = Dict[str,Union[List[str],int,torch.Tensor]]

def harmonic( y: float, t: float) -> float:
    if y > t: return round(y/t)
    else:     return 1.0 / round(t/y)

def sH(h:float) -> str:
    if abs(h) > 1: return str(round(h))
    else:
        sh = round(1/h)
        return f"1/{sh}" if sh > 1 else str(sh)

class SpectralPeakSelector(Module):

    def __init__(self, cfg: DictConfig, device: device, fspace: Tensor ) -> None:
        super().__init__()
        self.requires_grad_(False)
        self.device: device = device
        self.cfg: DictConfig = cfg
        self.log = logging.getLogger()
        self.fspace = fspace
        self.hsr: Tensor = None

    def process_key_event(self, key: str):
        if key == 'ctrl+t':
            pass

    def forward(self, hsmag: Tensor) -> Tensor:
        self.hsr = hsmag[:, 0, :].squeeze()
        hspeak: Tensor = self.hsr.argmax(dim=-1).squeeze()
        result: Tensor = self.fspace[hspeak]
        self.log.info(f" SpectralPeakSelector.forward: result{shp(result)}, hspeak{shp(hspeak)}, hsr{shp(self.hsr)}, hsmag{shp(hsmag)}, fspace{shp(self.fspace)}")
        return result

class Evaluator:

    def __init__(self, cfg: DictConfig, device: device, loader: ElementLoader, model: nn.Module, loss: HLoss ) -> None:
        super().__init__()
        self.device: device = device
        self.cfg: DictConfig = cfg
        self.log = logging.getLogger()
        self.model = model
        self.loader = loader
        self.loss: HLoss =   loss

    def encode_element(self, element: RDict) -> TRDict:
        t,y,p = element.pop('t'), element.pop('y'), element.pop('period')
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
            losses, hs= [], []
            for ifile in range(0,10):
                self.loader.set_file(ifile)
                elem_idx = 0
                for ielem in range(0,self.loader.file_size):
                    element: Optional[TRDict] =  self.get_element(ielem)
                    if element is not None:
                        result: Tensor = self.model(element['z'])
                        y,t = result.item(), element['target']
                        loss: float = self.loss(y,t)
                        h, rh = self.loss.h, self.loss.rh
                        losses.append(loss)
                        hs.append(h)
                        if loss > 0.1:
                            print(f" * F-{ifile} Elem-{elem_idx}: yt=({y:.3f},{t*h:.3f},{t:.3f}), H={sH(h)}({sH(rh)}), yLoss= {loss:.5f}")
                        elem_idx+=1
            L: np.array = np.array(losses)
            H: np.array = np.array(hs)
            print(f"Loss mean = {L.mean():.3f}, range=[{L.min():.3f} -> {L.max():.3f}], H range=[{H.min():.3f} -> {H.max():.3f}]")


