from omegaconf import DictConfig, OmegaConf
from astrotime.util.logging import lgm, exception_handled
import logging
import torch

def astrotime_initialize(config: DictConfig, **kwargs):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    OmegaConf.resolve(config)
    lgm().init_logging( config.platform.logs,  config.train.version, config.train.overwrite_log )
    lgm().set_level( kwargs.get('log_level',logging.INFO) )
    device: torch.device = torch.device(f"cuda:{config.platform.gpu}" if (torch.cuda.is_available() and (config.platform.gpu >= 0)) else "cpu")
    return device