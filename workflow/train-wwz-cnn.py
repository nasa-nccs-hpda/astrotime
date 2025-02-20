import os, glob, sys, torch
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Hashable
import xarray as xr, numpy as np
from torch import nn
from astrotime.util.env import parse_clargs, get_device
from astrotime.loaders.sinusoid import ncSinusoidLoader
from astrotime.encoders.wavelet import WaveletEncoder
from argparse import Namespace
from astrotime.config.context import ConfigContext, cfg
from astrotime.trainers.signal_trainer import SignalTrainer
from astrotime.models.cnn_baseline import get_model_from_cfg


cname = "sinusoid_period"
configuration = dict( platform="explore", train="signal1", data="sinusoids.nc", transform="wwz", model="baseline_cnn" )
ccustom: Dict[str,Any] = { }
clargs, configuration = parse_clargs(configuration)
device: torch.device = torch.device(f"cuda:{clargs.gpu}" if (torch.cuda.is_available() and (clargs.gpu >= 0)) else "cpu")
cc = ConfigContext.initialize(cname, configuration, ccustom)


sinusoid_loader = ncSinusoidLoader( dataset_root, dataset_files, file_size, batch_size )
encoder = WaveletEncoder( device, series_length, nfreq, fbounds, fscale, nfeatures, int(max_series_length*(1-sparsity)) )



model: nn.Module = get_model_from_cfg( cc.cfg.model )

trainer = SignalTrainer( signal, encoder, model, clargs, cc.cfg.training )

trainer.train()