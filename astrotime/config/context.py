from omegaconf import DictConfig, OmegaConf
import logging, os, sys
from logging import Logger
import torch

def astrotime_initialize(config: DictConfig, version: str):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    OmegaConf.resolve(config)
    device: torch.device = torch.device(f"cuda:{config.platform.gpu}" if (torch.cuda.is_available() and (config.platform.gpu >= 0)) else "cpu")
    configure_logging(config.platform, version)
    logging.getLogger().info("INIT")
    return device

def configure_logging(cfg: DictConfig, version) -> Logger:
    log_file = f"{cfg.project_root}/logs/astrotime.{version}.log"
    os.makedirs(f"{cfg.project_root}/logs", exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(cfg.log_level.upper())
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.ERROR)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler( log_file, mode='w' )
    file_handler.setLevel( cfg.log_level.upper() )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    print(f"\n      Logging to {log_file}, level = {logging.getLevelName(logger.level)}")
    return logger