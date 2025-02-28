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
        self.log = logging.getLogger()
        self._checkpoint_manager = None
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
            self.nepochs    += self.start_epoch
            print(f"\n      Loading checkpoint from {self._checkpoint_manager.checkpoint_path()}: epoch={self.start_epoch}, batch={self.start_batch}\n")

    def update_weights(self, loss: Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_batch(self, tset: TSet, batch_index) -> Tuple[torch.Tensor,torch.Tensor]:
        dset: xa.Dataset = self.loader.get_batch(tset,batch_index)
        target: Tensor = torch.from_numpy(dset['p'].values[:, None]).to(self.device)
        t, y = self.encoder.encode_batch( dset['t'].values, dset['y'].values )
        input: Tensor = torch.concat((t[:, None, :], y), dim=1)
        return input, target

    def exec_validation(self, threshold = None):
        self.model.train(False)
        losses, nb = [], self.loader.nbatches(TSet.Validation)
        print(f"      Exec validation: {nb} batches, nelements = {self.loader.nelements(TSet.Validation)}, device={self.device}\n")
        for ibatch in range(0, nb):
            batch_input, batch_target = self.get_batch(TSet.Validation, ibatch)
            batch_losses = []
            for ielem in range(batch_input.shape[0]):
                elem_input, elem_target = batch_input[ielem:ielem+1], batch_target[ielem:ielem+1]
                #print(f" >>>> batch_input:  {batch_input.shape} -> {elem_input.shape}")
                #print(f" >>>> batch_target: {batch_target.shape} -> {elem_target.shape}")
                result: Tensor = self.model(elem_input)
                #print(f" >>>> result:       {result.shape}")
                fr, ft, fi = result.squeeze().item(), elem_target.squeeze().item(), elem_input[0,1,:]
                loss: float = abs( fr - ft )
                if (threshold is not None) and (loss > threshold):
                    print(f" B-{ibatch}:{ielem} loss = {loss:.3f}, fr={fr:.2f}, ft={ft:.2f}, target_range=({batch_target.min().item():.2f}, {batch_target.max().item():.2f}), "
                          f"input{list(fi.shape)}: stats=({fi.min().item():.2f},{fi.max().item():.2f},{fi.mean().item():.2f},{fi.std().item():.2f})")
                else:
                    batch_losses.append(loss)
            print( f"                  B-{ibatch}: loss = {np.array(batch_losses).mean():.3f}")
            losses.extend(batch_losses)
        return np.array(losses)

    def train(self):
        nb = self.loader.nbatches(TSet.Train)
        print(f"SignalTrainer: {nb} batches, {self.nepochs} epochs, nelements = {self.loader.nelements(TSet.Train)}, device={self.device}")
        with self.device:
            for epoch in range(self.start_epoch,self.nepochs):
                self.model.train(True)
                losses, log_interval = [], 200
                batch0 = self.start_batch if (epoch == self.start_epoch) else 0
                for ibatch in range(batch0,nb):
                    t0 = time.time()
                    input, target = self.get_batch(TSet.Train,ibatch)
                    self.global_time = time.time()
                    self.log.info( f"TRAIN BATCH-{ibatch}: input={shp(input)}, target={shp(target)}")
                    result: Tensor = self.model( input )
                    loss: Tensor = self.loss_function( result.squeeze(), target.squeeze() )
                    self.update_weights(loss)
                    losses.append(loss.item())
                    if (ibatch % log_interval == 0) or ((ibatch < 5) and (epoch==0)):
                        aloss = np.array(losses)
                        print(f"E-{epoch} B-{ibatch} loss={aloss.mean():.3f} ({aloss.min():.3f} -> {aloss.max():.3f}), dt={time.time()-t0:.4f} sec")
                        losses = []

                self._checkpoint_manager.save_checkpoint( epoch, 0 )






