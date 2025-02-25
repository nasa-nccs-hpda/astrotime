from omegaconf import DictConfig, OmegaConf
import logging
log = logging.getLogger(__name__)
import torch

def astrotime_initialize(config: DictConfig, **kwargs):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    OmegaConf.resolve(config)
    device: torch.device = torch.device(f"cuda:{config.platform.gpu}" if (torch.cuda.is_available() and (config.platform.gpu >= 0)) else "cpu")
    return device