from omegaconf import DictConfig, OmegaConf
import logging, os
import torch

def astrotime_initialize(config: DictConfig, version: str, **kwargs):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    OmegaConf.resolve(config)
    device: torch.device = torch.device(f"cuda:{config.platform.gpu}" if (torch.cuda.is_available() and (config.platform.gpu >= 0)) else "cpu")
    log_file = f"{config.platform.project_root}/logs/astrotime.{version}.log"
    if os.path.exists(log_file): os.remove(log_file)
    logging.basicConfig( filename=log_file, encoding='utf-8' )
    logger, level = logging.getLogger("astrotime"), config.platform.log_level.upper()
    logger.setLevel( level )
    print( f"\n      Logging to {log_file}, level = {logging.getLevelName(logger.level)} ({level})")
    logger.info("INIT")
    return device