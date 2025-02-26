from typing import Dict, Tuple
from astrotime.util.config import TSet
from omegaconf import DictConfig
from .checkpoints import CheckpointManager
from astrotime.encoders.base import Encoder
import xarray as xa
from astrotime.util.math import shp
from astrotime.loaders.base import DataLoader
import time, torch, numpy as np
from torch import nn, optim, Tensor
import logging
log = logging.getLogger("astrotime")

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
        self.optimizer: optim.Optimizer = self.get_optimizer()
        self._checkpoint_manager = CheckpointManager( model, self.optimizer, cfg )
        self.start_batch: int = 0
        self.start_epoch: int = 0
        self.epoch_loss: float = 0.0
        self.epoch0: int = 0
        self.nepochs = self.cfg.nepochs
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
        log.info( f" Model layer stats:")
        for stats in  self.exec_stats:
            log.info(f"{stats[0]}: dt={stats[1]}s")

    def get_optimizer(self) -> optim.Optimizer:
         if   self.cfg.optim == "rms":  return optim.RMSprop( self.model.parameters(), lr=self.cfg.lr )
         elif self.cfg.optim == "adam": return optim.Adam(    self.model.parameters(), lr=self.cfg.lr )
         else: raise RuntimeError( f"Unknown optimizer: {self.cfg.optim}")

    def initialize_checkpointing(self):
        if self.cfg.refresh_state:
            self._checkpoint_manager.clear_checkpoints()
            print("\n *** No checkpoint loaded: training from scratch *** \n")
        else:
            self.train_state = self._checkpoint_manager.load_checkpoint( TSet.Train, update_model=True )
            self.epoch0      = self.train_state.get('epoch', 0)
            self.start_batch = self.train_state.get('batch', 0)
            self.start_epoch = int(self.epoch0)
            self.nepochs    += self.start_epoch
            print(f"\n Loading checkpoint from {self._checkpoint_manager.checkpoint_path(TSet.Train)}: epoch={self.start_epoch}, batch={self.start_batch}\n")

    def update_weights(self, loss: Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_batch(self, batch_index) -> Tuple[torch.Tensor,torch.Tensor]:
        dset: xa.Dataset = self.loader.get_batch(batch_index)
        target: Tensor = torch.from_numpy(dset['p'].values[:, None]).to(self.device)
        t, y = self.encoder.encode_batch( dset['t'].values, dset['y'].values )
        input: Tensor = torch.concat((t[:, None, :], y), dim=1)
        return input, target

    def exec_validation(self, threshold = None):
        self.model.train(False)
        losses = []
        print( f"Exec validation: {self.loader.nbatches_validation} batches")
        for ibatch in range(0, self.loader.nbatches_validation):
            input, target = self.get_batch(self.loader.nbatches + ibatch)
            result: Tensor = self.model(input)
            loss: float = self.loss_function(result.squeeze(), target.squeeze()).item()
            if (threshold is not None) and (loss > threshold):
                print(f" B-{ibatch} loss = {loss:.3f}")
            losses.append(loss)
        return np.array(losses)

    def train(self):
        print(f"SignalTrainer: {self.loader.nbatches} train batches, {self.loader.nbatches_validation} validation batches, {self.nepochs} epochs, nelements = {self.loader.nelements}, device={self.device}")
        self.initialize_checkpointing()
        with self.device:
            for epoch in range(self.start_epoch,self.nepochs):
                self.model.train(True)
                losses, log_interval = [], 200
                batch0 = self.start_batch if (epoch == self.start_epoch) else 0
                for ibatch in range(batch0, self.loader.nbatches):
                    t0 = time.time()
                    input, target = self.get_batch(ibatch)
                    self.global_time = time.time()
                    log.info( f"TRAIN BATCH-{ibatch}: input={shp(input)}, target={shp(target)}")
                    result: Tensor = self.model( input )
                    loss: Tensor = self.loss_function( result.squeeze(), target.squeeze() )
                    self.update_weights(loss)
                    losses.append(loss.item())
                    if (ibatch % log_interval == 0) or ((ibatch < 5) and (epoch==0)):
                        aloss = np.array(losses)
                        print(f"E-{epoch} B-{ibatch} loss={aloss.mean():.3f} ({aloss.min():.3f} -> {aloss.max():.3f}), dt={time.time()-t0:.4f} sec")
                        losses = []

                self._checkpoint_manager.save_checkpoint( TSet.Train, epoch, 0 )






