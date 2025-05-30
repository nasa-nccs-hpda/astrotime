from torch import Tensor, device
from torch.nn.modules import Module
from omegaconf import DictConfig, OmegaConf
import logging

class SpectralPeakSelector(Module):

    def __init__(self, cfg: DictConfig, device: device, fspace: Tensor, dim: int = 0 ) -> None:
        super().__init__()
        self.requires_grad_(False)
        self.device: device = device
        self.cfg: DictConfig = cfg
        self.log = logging.getLogger()
        self.fspace = fspace
        self.dim: int = dim

    def forward(self, input: Tensor) -> Tensor:
        spectrum: Tensor = input[:,self.dim,:].squeeze()
        speak: Tensor = spectrum.argmax(dim=1).squeeze()
        return self.fspace[speak]


