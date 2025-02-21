from omegaconf import DictConfig, OmegaConf
from astrotime.util.logging import lgm, exception_handled
import logging
import torch

@exception_handled
def astrotime_initialize(config: DictConfig, **kwargs):
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    OmegaConf.resolve(config)
    lgm().init_logging( config.platform.logs,  config.train.version, config.train.overwrite_log )
    lgm().set_level( kwargs.get('log_level',logging.INFO) )
    device: torch.device = torch.device(f"cuda:{config.platform.gpu}" if (torch.cuda.is_available() and (config.platform.gpu >= 0)) else "cpu")
    return device