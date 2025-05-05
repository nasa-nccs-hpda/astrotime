import random, time, numpy as np
import torch, math
from typing import List, Tuple, Mapping
from torch import Tensor, device, nn
from .embedding import EmbeddingLayer
from astrotime.util.math import log2space, tnorm
from astrotime.util.logging import elapsed
from .wavelet import WaveletAnalysisLayer

def clamp( idx: int ) -> int: return max( 0, idx )

def embedding_space( cfg, device: device ) -> Tuple[np.ndarray,Tensor]:
	lspace = log2space( cfg.base_freq, cfg.base_freq*pow(2,cfg.noctaves), cfg.nfreq_oct*cfg.noctaves )
	tspace = torch.FloatTensor( lspace ).to(device)
	return lspace, tspace

class FoldingAnalysisLayer(WaveletAnalysisLayer):

	def __init__(self, name, cfg, embedding_space: Tensor, device: device):
		WaveletAnalysisLayer.__init__(self, name, cfg, embedding_space, device)
		self.nfreq_oct: int   = cfg.nfreq_oct
		self.base_freq: float = cfg.base_freq
		self.noctaves: int    = cfg.noctaves
		self.nfreq: int       = self.nfreq_oct * self.noctaves

	def magnitude(self, embedding: Tensor, **kwargs) -> np.ndarray:
		t0 = time.time()
		mag: np.ndarray = torch.sqrt( torch.sum( embedding**2, dim=1 ) ).to('cpu').numpy()
		norm: np.ndarray = np.ones(mag.shape[1])
		for i in range(1,self.noctaves):
			idx_fold = i*self.nfreq_oct
			if idx_fold < mag.shape[1]:
				octave = mag[:,idx_fold:]
				mag[:,:octave.shape[1]] += octave
				norm[:octave.shape[1]] += 1
		mag = mag/norm[None,:]
		self.log.info(f"Completed folding magnitude in {elapsed(t0):.5f} sec: mag{list(mag.shape)}")
		return mag

	def get_target_freq( self, target_period: float ) -> float:
		f0 = 1/target_period
		return f0