from typing import List, Optional, Dict, Type, Tuple, Union
from omegaconf import DictConfig
from .checkpoints import CheckpointManager
from astrotime.util.tensor_ops import check_nan
from astrotime.loaders.base import RDict, ElementLoader
from astrotime.trainers.loss import ExpLoss, OctaveRegressionLoss
from astrotime.encoders.embedding import EmbeddingLayer
import time, sys, torch, logging, numpy as np
from torch import nn, optim, Tensor
from astrotime.util.series import TSet
from astrotime.util.logging import elapsed
TRDict = Dict[str,Union[List[str],int,torch.Tensor]]

def tocpu( c, idx=0 ):
    if isinstance( c, Tensor ):
        ct = c.detach().cpu()
        if ct.ndim == 1: ct = ct[idx]
        return ct.item()
    else:
        return c


class IterativeTrainer(object):

    def __init__(self, cfg: DictConfig, device: torch.device, loader: ElementLoader, model: nn.Module, embedding: EmbeddingLayer ):
        self.device: torch.device = device
        self.loader: ElementLoader = loader
        self.embedding = embedding
        self.cfg: DictConfig = cfg.train
        self.model: nn.Module = model
        self.mtype: str = cfg.model.mtype
        self.noctaves = cfg.data.noctaves
        self.f0 = cfg.data.base_freq
        self.optimizer: optim.Optimizer = None
        self.log = logging.getLogger()
        self.loss: nn.Module = self.get_loss(cfg.data)
        self._checkpoint_manager: CheckpointManager = None
        self.start_batch: int = 0
        self.start_epoch: int = 0
        self.epoch_loss: float = 0.0
        self.epoch0: int = 0
        self.train_state = None
        self.global_time = None
        self.exec_stats = []
        if model is not None:
            for module in model.modules(): self.add_callbacks(module)

    def get_loss(self, cfg: DictConfig) -> nn.Module:
        if   "octave_regression" in self.mtype: return OctaveRegressionLoss(cfg, self.embedding)
        elif "regression"        in self.mtype: return ExpLoss(cfg)
        elif "peakfinder"        in self.mtype: return ExpLoss(cfg)
        elif "classification"    in self.mtype: return nn.CrossEntropyLoss()
        else: raise RuntimeError(f"Unknown model type: {self.mtype}")

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

    def get_octave(self, f: Tensor) -> Tensor:
        octave = torch.floor( torch.log2(f/self.f0) ).to(torch.long)
        return octave

    def fold_by_octave(self, f: Tensor) -> Tensor:
        octave = torch.floor(torch.log2(f / self.f0))
        octave_base_freq = self.f0 * torch.pow(2, octave)
        return f / octave_base_freq

    def get_input(self, batch: TRDict) -> Tensor:
        return batch['z']

    def get_input_octave(self, batch: TRDict) -> Optional[Tensor]:
        return batch.get('octave')

    def get_target(self, batch: TRDict ) -> Tensor:
        f: Tensor = batch['target']
        if "regression" in self.mtype:       return f
        elif "classification" in self.mtype: return self.get_octave(f)
        else: raise RuntimeError(f"Unknown model type: {self.cfg.model_type}")

    def get_optimizer(self) -> optim.Optimizer:
        if   self.cfg.optim == "rms":  return optim.RMSprop( self.model.parameters(), lr=self.cfg.lr )
        elif self.cfg.optim == "adam": return optim.Adam(    self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay )
        else: raise RuntimeError( f"Unknown optimizer: {self.cfg.optim}")

    def initialize_checkpointing( self, version: str, init_version:Optional[str]=None ):
        self._checkpoint_manager = CheckpointManager( version, self.model, self.optimizer, self.cfg )
        if self.cfg.refresh_state:
            self._checkpoint_manager.clear_checkpoints()
            print("\n *** No checkpoint loaded: training from scratch *** \n")
        else:
            self.train_state = self._checkpoint_manager.load_checkpoint( init_version=init_version, update_model=True )
            self.epoch0      = self.train_state.get('epoch', 0)
            self.start_batch = self.train_state.get('batch', 0)
            self.start_epoch = int(self.epoch0)
            print(f"\n      Loading checkpoint from {self._checkpoint_manager.checkpoint_path()}: epoch={self.start_epoch}, batch={self.start_batch}\n")

    def load_checkpoint( self, version: str ):
        if version is not None:
            self.optimizer = self.get_optimizer()
            self._checkpoint_manager = CheckpointManager( version, self.model, self.optimizer, self.cfg )
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

    def encode_batch(self, batch: RDict) -> TRDict:
        self.log.debug( f"encode_batch: {list(batch.keys())}")
        t,y = batch.pop('t'), batch.pop('y')
        p: Tensor = torch.from_numpy(batch.pop('period')).to(self.device)
        o = batch.pop('octave', None)
        if o is not None: o = torch.from_numpy(o).to(self.device)
        z: Tensor = self.to_tensor( t, y )
        return dict( z=z, target=1/p, octave=o, **batch )

    def to_tensor(self, x: np.ndarray, y: np.ndarray) -> Tensor:
        with (self.device):
            Y: Tensor = torch.FloatTensor(y).to(self.device)
            X: Tensor = torch.FloatTensor(x).to(self.device)
            return torch.stack((X,Y), dim=1)

    def get_next_batch(self) -> Optional[TRDict]:
        while True:
            dset: RDict = self.loader.get_next_batch()
            if dset is not None:
                return self.encode_batch(dset)

    @property
    def mode(self) -> TSet:
        return self.loader.tset

    @property
    def nepochs(self) -> int:
        return self.cfg.nepochs if self.training else 1

    @property
    def epoch_range(self) -> Tuple[int,int]:
        e0: int = self.start_epoch if (self.mode == TSet.Train) else 0
        return e0, e0+self.nepochs

    def set_train_status(self):
        self.loader.initialize()
        if self.mode == TSet.Train:
            self.model.train(True)

    @property
    def training(self) -> bool:
        return not self.cfg.mode.startswith("val")

    def test_model(self):
        print(f"SignalTrainer[{self.mode}]: , {self.nepochs} epochs, device={self.device}")
        with self.device:
            self.set_train_status()
            self.loader.init_epoch()
            batch: Optional[TRDict] = self.get_next_batch()
            result: Tensor = self.model( batch['z'] )
            print( f" ** (batch{list(batch['z'].shape)}, target{list(batch['target'].shape)}) ->  result{list(result.shape)}")

    def train(self,version,ckp_version=None):
        print(f"SignalTrainer[{self.mode}]: , {self.nepochs} epochs, device={self.device}")
        self.optimizer = self.get_optimizer()
        self.initialize_checkpointing(version,ckp_version)
        with self.device:
            print(f" ---- Running Training cycles ---- ")
            for epoch in range(*self.epoch_range):
                te = time.time()
                self.set_train_status()
                self.loader.init_epoch(TSet.Train)
                losses, log_interval, t0 = [], 50, time.time()
                try:
                    for ibatch in range(0,sys.maxsize):
                        t0 = time.time()
                        batch = self.get_next_batch()
                        binput: Tensor = self.get_input(batch)
                        target: Tensor = self.get_target(batch)
                        octave: Tensor = self.get_octave(target)
                        if binput.shape[0] > 0:
                            self.global_time = time.time()
                            self.embedding.set_octave_data(octave)
                            result: Tensor = self.model( binput )
                            if result.squeeze().ndim > 0:
                                # print(f"result{list(result.shape)} range: [{result.min().cpu().item()} -> {result.max().cpu().item()}]")
                                loss: Tensor =  self.loss( result.squeeze(), target )
                                self.conditionally_update_weights(loss)
                                losses.append(loss.cpu().item())
                                if ibatch % log_interval == 0:
                                    aloss = np.array(losses[-log_interval:])
                                    print(f"E-{epoch} F-{self.loader.ifile}:{self.loader.file_index} B-{ibatch} loss={aloss.mean():.3f}, range=({aloss.min():.3f} -> {aloss.max():.3f}), dt/batch={elapsed(t0):.5f} sec")
                                    self._checkpoint_manager.save_checkpoint(epoch, ibatch)

                except StopIteration:
                    loss_data = np.array(losses)
                    print( f"Completed epoch {epoch} in {elapsed(te)/60:.5f} min, mean-loss= {loss_data.mean():.3f}, median= {np.median(loss_data):.3f}")


                epoch_losses = np.array(losses)
                print(f" ------ Epoch Loss: mean={epoch_losses.mean():.3f}, median={np.median(epoch_losses):.3f}, range=({epoch_losses.min():.3f} -> {epoch_losses.max():.3f})")

    def evaluate(self,version,ckp_version=None):
        print(f"SignalTrainer[{self.mode}]: , {self.nepochs} epochs, device={self.device}")
        self.optimizer = self.get_optimizer()
        self.initialize_checkpointing(version,ckp_version)
        with self.device:
            self.loader.initialize()
            print(f" ---- Running Test cycles ---- ")
            self.loader.init_epoch(TSet.Validation)
            losses, log_interval, t0 = [], 50, time.time()
            try:
                for ibatch in range(0,sys.maxsize):
                    t0 = time.time()
                    batch = self.get_next_batch()
                    binput: Tensor = self.get_input(batch)
                    target: Tensor = self.get_target(batch)
                    octave: Tensor = self.get_octave(target)
                    if binput.shape[0] > 0:
                        self.global_time = time.time()
                        self.embedding.set_octave_data(octave)
                        result: Tensor = self.model( binput )
                        if result.squeeze().ndim > 0:
                            # print(f"result{list(result.shape)} range: [{result.min().cpu().item()} -> {result.max().cpu().item()}]")
                            loss: Tensor =  self.loss( result.squeeze(), target )
                            losses.append(loss.cpu().item())
                            if ibatch % log_interval == 0:
                                aloss = np.array(losses[-log_interval:])
                                print(f"F-{self.loader.ifile}:{self.loader.file_index} B-{ibatch} loss={aloss.mean():.3f}, range=({aloss.min():.3f} -> {aloss.max():.3f}), dt/batch={elapsed(t0):.5f} sec")

            except StopIteration:
                epoch_losses = np.array(losses)
                print(f" ------ EVAL Loss: mean={epoch_losses.mean():.3f}, median={np.median(epoch_losses):.3f}, range=({epoch_losses.min():.3f} -> {epoch_losses.max():.3f})")

    def evaluate_classification(self, version: str = None) -> Tensor:
        self.load_checkpoint(version)
        with self.device:
            self.loader.init_epoch()
            losses, log_interval, results = [], 50, []
            try:
                for ibatch in range(0, sys.maxsize):
                    batch = self.get_next_batch()
                    binput: Tensor = self.get_input(batch)
                    target: Tensor = self.get_target(batch)
                    result: Tensor = self.model(binput)
                    max_idx: Tensor = torch.argmax(result,dim=1,keepdim=False)
                    results.append( max_idx )
                    ncorrect = torch.eq(max_idx,target).sum()
                    losses.append( (ncorrect,result.shape[0]) )
            except StopIteration:
                    ncorrect, ntotal = 0, 0
                    for (nc,nt) in losses:
                        ncorrect += nc
                        ntotal += nt
                    print(f"       *** Classification: {ncorrect*100.0/ntotal:.1f}% correct with {ntotal} elements.")
            return torch.concatenate(results)

    def update(self, version: str = None):
        self.load_checkpoint(version)
        with self.device:
            self.loader.init_epoch()
            losses, octave_data = [], []
            try:
                for ibatch in range(0, sys.maxsize):
                    batch = self.get_next_batch()
                    if batch is not None:
                        binput: Tensor = self.get_input(batch)
                        target: Tensor = self.get_target(batch)
                        result: Tensor = self.model(binput)
                        max_idx: Tensor = torch.argmax(result,dim=1,keepdim=False)
                        ncorrect = torch.eq(max_idx, target).sum()
                        losses.append((ncorrect, result.shape[0]))
                        batch_start = batch['offset']
                        file_index = batch['file']
                        octaves = max_idx.cpu().tolist()
                        for ib in range(len(octaves) ):
                            octave_data.append( (file_index, batch_start+ib, octaves[ib]) )
            except StopIteration:
                self.loader.add_octave_data(octave_data)
                ncorrect, ntotal = 0, 0
                for (nc, nt) in losses:
                    ncorrect += nc
                    ntotal += nt
                print(f" Updated dataset files with octave data ( {ncorrect * 100.0 / ntotal:.1f}% correct with {ntotal} elements )")


    def preprocess(self):
        with self.device:
            te = time.time()
            self.loader.initialize(TSet.Validation)
            self.loader.init_epoch()
            try:
                for ibatch in range(0,sys.maxsize):
                    batch = self.get_next_batch()
            except StopIteration:
                print(f"Completed preprocess in {elapsed(te) / 60:.5f} min.")






