from typing import Dict, Tuple
from omegaconf import DictConfig
from .checkpoints import CheckpointManager
from astrotime.encoders.base import Encoder
import xarray as xa
from astrotime.util.math import shp
from astrotime.loaders.base import DataLoader
import time, torch, logging, numpy as np
from torch import nn, optim, Tensor
from astrotime.util.series import TSet
from astrotime.util.logging import elapsed

def tocpu( c, idx=0 ):
    if isinstance( c, Tensor ):
        ct = c.detach().cpu()
        if ct.ndim == 1: ct = ct[idx]
        return ct.item()
    else:
        return c

class SignalTrainer(object):

    def __init__(self, cfg: DictConfig, loader: DataLoader, encoder: Encoder, model: nn.Module ):
        self.device = encoder.device
        self.loader: DataLoader = loader
        self.cfg: DictConfig = cfg
        self.loss_function: nn.Module = nn.L1Loss()
        self.model: nn.Module = model
        self.encoder: Encoder = encoder
        self.time_scale = encoder.time_scale
        self.optimizer: optim.Optimizer = self.get_optimizer()
        self.log = logging.getLogger()
        self._checkpoint_manager = None
        self.start_batch: int = 0
        self.start_epoch: int = 0
        self.epoch_loss: float = 0.0
        self.epoch0: int = 0
        self.train_state = None
        self.global_time = None
        self.exec_stats = []
        for module in model.modules(): self.add_callbacks(module)

    def add_callbacks(self, module):
        pass
        #module.register_forward_hook(self.store_time)

    def store_time(self, module, input, output ):
        self.exec_stats.append( (module.__class__.__name__, time.time()-self.global_time) )
        self.global_time = time.time()

    def log_layer_stats(self):
        self.log.info( f" Model layer stats:")
        for stats in  self.exec_stats:
            self.log.info(f"{stats[0]}: dt={stats[1]}s")

    def get_optimizer(self) -> optim.Optimizer:
         if   self.cfg.optim == "rms":  return optim.RMSprop( self.model.parameters(), lr=self.cfg.lr )
         elif self.cfg.optim == "adam": return optim.Adam(    self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay )
         else: raise RuntimeError( f"Unknown optimizer: {self.cfg.optim}")

    def initialize_checkpointing(self, version: str):
        self._checkpoint_manager = CheckpointManager( version, self.model, self.optimizer, self.cfg )
        if self.cfg.refresh_state:
            self._checkpoint_manager.clear_checkpoints()
            print("\n *** No checkpoint loaded: training from scratch *** \n")
        else:
            self.train_state = self._checkpoint_manager.load_checkpoint( update_model=True )
            self.epoch0      = self.train_state.get('epoch', 0)
            self.start_batch = self.train_state.get('batch', 0)
            self.start_epoch = int(self.epoch0)
            print(f"\n      Loading checkpoint from {self._checkpoint_manager.checkpoint_path()}: epoch={self.start_epoch}, batch={self.start_batch}\n")

    def conditionally_update_weights(self, loss: Tensor):
        if self.mode == TSet.Train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_batch(self, tset: TSet, batch_index) -> Tuple[torch.Tensor,torch.Tensor]:
        dset: xa.Dataset = self.loader.get_batch(tset,batch_index)
        target: Tensor = torch.from_numpy(dset['p'].values[:, None]).to(self.device)
        t, y = self.encoder.encode_batch( dset['t'].values, dset['y'].values )
        z: Tensor = torch.concat((t[:, None, :]*self.time_scale, y), dim=1)
        return z, target*self.time_scale

    @property
    def mode(self) -> TSet:
        return TSet.Validation if self.cfg.mode.startswith("val") else TSet.Train

    @property
    def nbatches(self) -> int:
        return self.loader.nbatches(self.mode)

    @property
    def nepochs(self) -> int:
        return self.cfg.nepochs if self.training else 1

    @property
    def epoch_range(self) -> Tuple[int,int]:
        e0: int = self.start_epoch if (self.mode == TSet.Train) else 0
        return e0, e0+self.nepochs

    def set_train_status(self):
        if self.mode == TSet.Train:
            self.model.train(True)

    @property
    def training(self) -> bool:
        return not self.cfg.mode.startswith("val")

    def compute(self):
        print(f"SignalTrainer[{self.mode}]: {self.nbatches} batches, {self.nepochs} epochs, nelements = {self.loader.nelements(self.mode)}, device={self.device}")
        with self.device:
            self.set_train_status()
            for epoch in range(*self.epoch_range):
                losses, log_interval = [], 200
                batch0 = self.start_batch if ((epoch == self.start_epoch) and (self.mode == TSet.Train)) else 0
                for ibatch in range(batch0,self.nbatches):
                    t0 = time.time()
                    batch_input, target = self.get_batch(self.mode,ibatch)
                    self.global_time = time.time()
                    result: Tensor = self.model( batch_input )
                    loss: Tensor = self.loss_function( result.squeeze(), target.squeeze() )
                    self.conditionally_update_weights(loss)
                    losses.append(loss.item())
                    if (self.mode == TSet.Train) and ((ibatch % log_interval == 0) or ((ibatch < 5) and (epoch==0))):
                        aloss = np.array(losses)
                        mean_loss = aloss.mean()
                        print(f"E-{epoch} B-{ibatch} loss={mean_loss:.3f} (unscaled: {mean_loss/self.time_scale:.3f}), range=({aloss.min():.3f} -> {aloss.max():.3f}), dt-epoch={elapsed(t0)*self.nbatches/60:.5f} min")
                        losses = []

                if self.mode == TSet.Train:
                    self._checkpoint_manager.save_checkpoint( epoch, 0 )
                else:
                    val_losses = np.array(losses)
                    print( f" Validation Loss: mean={val_losses.mean():.3f}, median={np.median(val_losses):.3f}, range=({val_losses.min():.3f} -> {val_losses.max():.3f})")







