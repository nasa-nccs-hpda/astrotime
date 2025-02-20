from typing import Any, Dict, List, Tuple, Type, Optional, Union
from astrotime.util.config import TSet
from omegaconf import DictConfig
from .checkpoints import CheckpointManager
from .accumulators import ResultsAccumulator, LossAccumulator
from astrotime.encoders.base import Encoder
import xarray, math, random
from astrotime.loaders.base import DataLoader
from astrotime.config.context import cfg
import time, torch, numpy as np
from torch import nn, optim, Tensor
from argparse import Namespace

def tocpu( c, idx=0 ):
    if isinstance( c, Tensor ):
        ct = c.detach().cpu()
        if ct.ndim == 1: ct = ct[idx]
        return ct.item()
    else:
        return c

class SignalTrainer(object):

    def __init__(self, loader: DataLoader, encoder: Encoder, args: Namespace, cfg: DictConfig):
        self.loader: DataLoader = loader
        self.encoder: Encoder = encoder
        self.nbatches: int = loader.nbatches
        self.batch_size = self.loader.batch_size
        self.args = args
        self.cfg = cfg
        self.loss_function: nn.Module = nn.L1Loss()
        self._checkpoint_manager: CheckpointManager = None
        self.results_accum: ResultsAccumulator = ResultsAccumulator()
        self.model.initialize( self.get_batch() )
        self.optimizer: optim.Optimizer = self.get_optimizer()
        self.start_batch: int = 0
        self.start_epoch: int = 0
        self.epoch_loss: float = 0.0
        self.epoch0: int = 0
        self.train_state: Optional[Dict[str,Any]] = None
        self._losses: Dict[TSet, LossAccumulator] = {}

    def __setattr__(self, key: str, value: Any) -> None:
        if ('parms' in self.__dict__.keys()) and (key in self.parms.keys()):
            self.parms[key] = value
        else:
            super(SignalTrainer, self).__setattr__(key, value)

    def __getattr__(self, key: str) -> Any:
        if 'parms' in self.__dict__.keys() and (key in self.parms.keys()):
            return self.parms[key]
        return super(SignalTrainer, self).__getattribute__(key)

    def get_optimizer(self) -> optim.Optimizer:
         if   self.optim == "rms":  return optim.RMSprop(self.model.parameters(), lr=cfg().training.lr)
         elif self.optim == "adam": return optim.Adam(self.model.parameters(), lr=cfg().training.lr)
         else: raise RuntimeError( f"Unknown optimizer: {self.optim}")

    @property
    def device(self) -> torch.device:
        return self.model.device

    def accumulate_losses(self, tset: TSet, epoch: int, mdata: Dict) -> Dict[str, float]:
        losses: LossAccumulator = self.get_losses(TSet.Train)
        acc_losses: Dict[str, float] = losses.accumulate_losses()
        if len(acc_losses) > 0:
            print(f"accumulate_losses: {acc_losses}")
            self.results_accum.record_losses(tset, epoch, acc_losses)
            self._checkpoint_manager.save_checkpoint(tset, acc_losses, mdata)
        return acc_losses

    def get_losses(self, tset: TSet) -> LossAccumulator:
        return self._losses.setdefault(tset, LossAccumulator())

    def initialize_checkpointing(self):
        self._checkpoint_manager = CheckpointManager(self.model, self.optimizer )
        if self.args.refresh_state:
            self._checkpoint_manager.clear_checkpoints()
            print(" *** No checkpoint loaded: training from scratch *** ")
        else:
            self.train_state = self._checkpoint_manager.load_checkpoint(TSet.Train, update_model=True)
            self.epoch0      = tocpu(self.train_state.get('epoch', 0))
            self.start_batch = tocpu(self.train_state.get('batch', 0))
            self.epoch_loss  = tocpu(self.train_state.get('loss', float('inf')))
            self.start_epoch = int(self.epoch0)
            self.nepochs += self.start_epoch

    def update_weights(self, loss: Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        losses: LossAccumulator = self.get_losses(TSet.Train)
        losses.register_loss('result', loss)

    def get_batch(self, batch_index: int = 0 ) -> Dict[str,torch.Tensor]:
        batch_data: xarray.Dataset = self.signal.get_batch(batch_index)
        batch_tensors = self.preprocessor.create_batch(batch_data)
        return batch_tensors

#    @torch.compile
    def train(self,**kwargs):
        nbatches: int = round(self.signal.nbatches * cfg().training.val_split)
        print(f"SignalTrainer: {nbatches} train_batches, {self.nepochs} epochs, nelements = {nbatches*self.model.batch_size}")
        self.initialize_checkpointing()
        losses,  log_interval = [], 100
        for epoch in range(self.start_epoch,self.nepochs):
            self.model.train()
            tset = TSet.Train
            batch0 = self.start_batch if (epoch == self.start_epoch) else 0
            train_batchs = range(batch0, nbatches)
            for ibatch in train_batchs:
                batch_data: xarray.Dataset = self.signal.get_batch(ibatch)
                batch_tensors = self.preprocessor.create_batch(batch_data)
                batch: Tensor = batch_tensors.pop("batch").to(self.device)
                target: Tensor = batch_tensors.pop("target").to(self.device)
                result = self.model( batch, **batch_tensors )
                loss: Tensor = self.loss_function( result.squeeze(), target.squeeze() )
                self.update_weights(loss)
                losses.append(loss.item())
                if (ibatch % log_interval == 0) or ((ibatch < 10) and (epoch==0)):
                    aloss = np.array(losses)
                    print(f"E-{epoch} B-{ibatch} loss={aloss.mean():.3f} ({aloss.min():.3f} -> {aloss.max():.3f})")
                    losses = []

            mdata = dict()
            acc_losses = self.accumulate_losses(tset, epoch, mdata )
            print(f"E-{epoch} acc_losses: {acc_losses}")
