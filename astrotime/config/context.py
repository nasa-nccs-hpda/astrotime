from omegaconf import DictConfig, OmegaConf
import logging
import torch

def astrotime_initialize(config: DictConfig, **kwargs):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    OmegaConf.resolve(config)
    device: torch.device = torch.device(f"cuda:{config.platform.gpu}" if (torch.cuda.is_available() and (config.platform.gpu >= 0)) else "cpu")
    log_file = f"{config.platform.project_root}/logs/astrotime.{config.train.version}.log"
    logging.basicConfig( filename=log_file, encoding='utf-8', level=logging.INFO )
    print( f"\n Logging to {log_file} \n")
    return device