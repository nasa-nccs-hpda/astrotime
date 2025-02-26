from omegaconf import DictConfig, OmegaConf
import logging
from hydra.core.hydra_config import HydraConfig
log = logging.getLogger("astrotime")
import torch

def astrotime_initialize(config: DictConfig, **kwargs):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    OmegaConf.resolve(config)
    device: torch.device = torch.device(f"cuda:{config.platform.gpu}" if (torch.cuda.is_available() and (config.platform.gpu >= 0)) else "cpu")
    # print(f" Hydra Output directory  : {HydraConfig.get().runtime.output_dir}")
    print( f"\n Logging to {config.platform.project_root}/logs/{config.train.version}.csv \n")
    return device