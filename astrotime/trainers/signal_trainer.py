from typing import Any, Dict, List, Tuple, Type, Optional, Union
from astrotime.util.config import TSet
from omegaconf import DictConfig
from .checkpoints import CheckpointManager
from .accumulators import LossAccumulator
from astrotime.encoders.base import Encoder
import xarray as xa, math, random
from astrotime.util.math import logspace, shp
from astrotime.loaders.base import DataLoader
from astrotime.util.logging import lgm, exception_handled
import time, torch, numpy as np
from torch import nn, optim, Tensor

def tocpu( c, idx=0 ):
    if isinstance( c, Tensor ):
        ct = c.detach().cpu()
        if ct.ndim == 1: ct = ct[idx]
        return ct.item()
    else:
        return c

class SignalTrainer(object):

    def __init__(self, loader: DataLoader, model: nn.Module, cfg: DictConfig ):
        self.loader: DataLoader = loader
        self.cfg: DictConfig = cfg
        self.loss_function: nn.Module = nn.L1Loss()
        self.model: nn.Module = model
        self.optimizer: optim.Optimizer = self.get_optimizer()
        self._checkpoint_manager = CheckpointManager( model, self.optimizer, cfg )
        self.start_batch: int = 0
        self.start_epoch: int = 0
        self.epoch_loss: float = 0.0
        self.epoch0: int = 0
        self.nepochs = self.cfg.nepochs
        self._losses: Dict[TSet, LossAccumulator] = {}
        self.train_state = None

    def log_sizes(self, input_tensor: Tensor):
        output = input_tensor
        lgm().log( f" Model data sizes: input{list(input_tensor.shape)}")
        for m in self.model.children():
            output = m(output)
            lgm().log(f" ** Layer-{m.__class__.__name__}: output{list(output.shape)}")

    def get_optimizer(self) -> optim.Optimizer:
         if   self.cfg.optim == "rms":  return optim.RMSprop( self.model.parameters(), lr=self.cfg.lr )
         elif self.cfg.optim == "adam": return optim.Adam(    self.model.parameters(), lr=self.cfg.lr )
         else: raise RuntimeError( f"Unknown optimizer: {self.cfg.optim}")

    @property
    def device(self) -> torch.device:
        return self.model.device

    def accumulate_losses(self, tset: TSet, epoch: int, mdata: Dict) -> Dict[str, float]:
        losses: LossAccumulator = self.get_losses(TSet.Train)
        acc_losses: Dict[str, float] = losses.accumulate_losses()
        if len(acc_losses) > 0:
            self._checkpoint_manager.save_checkpoint(tset, acc_losses, mdata)
        return acc_losses

    def get_losses(self, tset: TSet) -> LossAccumulator:
        return self._losses.setdefault(tset, LossAccumulator())

    def initialize_checkpointing(self):
        if self.cfg.refresh_state:
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

    def get_batch(self, batch_index) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        dset: xa.Dataset = self.loader.get_batch(batch_index)
        target: Tensor = torch.from_numpy(dset['p'].values[:, None]).to(self.device)
        y: Tensor = torch.from_numpy(dset['y'].values[:, None]).to(self.device)
        t: Tensor = torch.from_numpy(dset['t'].values).to(self.device)
        return y, t, target

    def train(self):
        print(f"SignalTrainer: {self.loader.nbatches} train_batches, {self.nepochs} epochs, nelements = {self.loader.nelements}, device={self.encoder.device}")
        self.initialize_checkpointing()
        losses,  log_interval = [], 100
        with self.device:
            for epoch in range(self.start_epoch,self.nepochs):
                self.model.train()
                batch0 = self.start_batch if (epoch == self.start_epoch) else 0
                train_batchs = range(batch0, self.loader.nbatches)
                for ibatch in train_batchs:
                    t0 = time.time()
                    y, t, target = self.get_batch(ibatch)
                    t1 = time.time()
                    result: Tensor = self.model( y, t )
                    loss: Tensor = self.loss_function( result.squeeze(), target.squeeze() )
                    self.update_weights(loss)
                    losses.append(loss.item())
                    lgm().log(f"E-{epoch} B-{ibatch}  process-time={time.time() - t0:.4f}s")
                    if (ibatch % log_interval == 0) or ((ibatch < 10) and (epoch==0)):
                        aloss = np.array(losses)
                        print(f"E-{epoch} B-{ibatch} loss={aloss.mean():.3f} ({aloss.min():.3f} -> {aloss.max():.3f}), tinmes: encode={t1-t0:.3f}s network={time.time()-t1:.3f}s")
                        t0, losses = t1, []

                mdata = dict()
                acc_losses = self.accumulate_losses(TSet.Train, epoch, mdata )
                print(f"E-{epoch} acc_losses: {acc_losses}")
