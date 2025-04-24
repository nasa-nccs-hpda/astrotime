
import random, time, numpy as np
import torch, math
from typing import List, Tuple, Mapping
from torch import Tensor, device, nn, matmul
from omegaconf import DictConfig, OmegaConf
from astrotime.util.math import log2space, tnorm
from astrotime.util.logging import elapsed
from astrotime.encoders.octaves import OctaveAnalysisLayer

def clamp( idx: int ) -> int: return max( 0, idx )

def spectral_space( cfg, device: device ) -> Tuple[np.ndarray,Tensor]:
	lspace = log2space( cfg.base_freq, cfg.base_freq*pow(2,cfg.noctaves), cfg.nfreq_oct*cfg.noctaves )
	tspace = torch.FloatTensor( lspace ).to(device)
	return lspace, tspace

def harmonics_space( cfg, device: device ) -> Tuple[np.ndarray,Tensor]:
	lspace = log2space( cfg.base_freq, cfg.base_freq*pow(2,cfg.nharmonics), cfg.nfreq_oct*cfg.nharmonics )
	tspace = torch.FloatTensor( lspace ).to(device)
	return lspace, tspace

def harmonics_filter( ts: np.ndarray, ys: np.ndarray, cfg: DictConfig, device ) -> np.ndarray:
	t: Tensor = torch.from_numpy( ts[None,:] if ts.ndim == 1 else ts )
	y: Tensor = torch.from_numpy( ys[None,:] if ys.ndim == 1 else ys )
	proj = HarmonicsFilterLayer( cfg, device )
	embedding = proj.embed( t, y )
	return proj.magnitude( embedding )

class HarmonicsFilterLayer(OctaveAnalysisLayer):

	def __init__(self, cfg, device: device):
		OctaveAnalysisLayer.__init__(self, cfg, harmonics_space(cfg, device)[1], device)
		self.fspace: np.ndarray = None
		self.ftspace: torch.Tensor = None
		self.f0 = None
		self.alpha = 200.0
		self.nharmonics = 6

	def embed1(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs ) -> Tensor:
		alpha = 500.0
		nharmonics = 6
		if self.f0 is None: self.f0 = ts[0].item()
		print(f"SpectralAutocorrelationLayer:")
		spectral_features: torch.Tensor = super(HarmonicsFilterLayer, self).embed( ts, ys, **kwargs)
		spectral_projection: torch.Tensor = torch.sqrt(torch.sum(spectral_features ** 2, dim=1))
		f: torch.Tensor = self._embedding_space
		harmonics: torch.Tensor = torch.stack( [f * ih for ih in range(1, nharmonics+1)], dim=1)
		df = (self._embedding_space[:,None,None] - harmonics[None,:,:])/self._embedding_space[:,None]
		W: torch.Tensor = torch.exp(-(df*alpha)**2).sum(dim=2)
		self.fspace, sspace = spectral_space(self.cfg, self.device)
		hfilter: torch.Tensor = matmul(spectral_projection, W) / self._embedding_space.shape[0]

		# print(f" ----- embedding_space{list(self._embedding_space.shape)}: {self._embedding_space.min():.3f} -> {self._embedding_space.max():.3f}")
		# print(f" ----- df{list(df.shape)}: {df.min():.3f} -> {df.max():.3f}")
		# print(f" ----- W{list(W.shape)}: {W.min():.5f} -> {W.max():.5f}")
		# print(f" ----- spectral_projection{list(spectral_projection.shape)}: {spectral_projection.min():.5f} -> {spectral_projection.max():.5f}")
		# print(f" ----- f{list(f.shape)}: {f.min():.5f} -> {f.max():.5f}")
		# print(f" ----- sspace{list(sspace.shape)}: {sspace.min():.5f} -> {sspace.max():.5f}")
		# print(f" ----- hfilter{list(hfilter.shape)}: {hfilter.min():.5f} -> {hfilter.max():.5f}")

		return hfilter[:,:sspace.shape[0]]

	def embed11(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs ) -> Tensor:
		alpha = 200.0
		nharmonics = 6
		self.log.info(f"SpectralAutocorrelationLayer:")
		spectral_features: torch.Tensor = super(HarmonicsFilterLayer, self).embed( ts, ys, **kwargs)
		spectral_projection: torch.Tensor = torch.sqrt(torch.sum(spectral_features ** 2, dim=1))
		f: torch.Tensor = self._embedding_space
		fh = [f * ih for ih in range(1, nharmonics+1)]
		harmonics: torch.Tensor = torch.stack(fh,dim=1)
		df = (self._embedding_space[None,:,None] - harmonics[:,None,:])
		W: torch.Tensor = torch.exp(-(df*alpha)**2/self._embedding_space[None,:,None]).sum(dim=2)
		self.fspace, sspace = spectral_space(self.cfg, self.device)
		hfilter: torch.Tensor = matmul(spectral_projection,W) / self._embedding_space.shape[0]

		self.log.info(f" ----- embedding_space{list(self._embedding_space.shape)}: {self._embedding_space.min():.3f} -> {self._embedding_space.max():.3f}")
		self.log.info(f" ----- fh: {fh}")
		self.log.info(f" ----- df{list(df.shape)}: {df.min():.3f} -> {df.max():.3f}")
		self.log.info(f" ----- W{list(W.shape)}: {W.min():.5f} -> {W.max():.5f}")
		self.log.info(f" ----- spectral_projection{list(spectral_projection.shape)}: {spectral_projection.min():.5f} -> {spectral_projection.max():.5f}")

		return hfilter[:,:sspace.shape[0]]

	def embed2(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs ) -> Tensor:
		alpha = 200.0
		nharmonics = 6
		self.log.info(f"SpectralAutocorrelationLayer:")
		spectral_features: torch.Tensor = super(HarmonicsFilterLayer, self).embed( ts, ys, **kwargs)
		spectral_projection: torch.Tensor = torch.sqrt(torch.sum(spectral_features ** 2, dim=1))
		f: torch.Tensor = self._embedding_space
		fh = [f * ih for ih in range(1, nharmonics+1)]
		harmonics: torch.Tensor = torch.stack(fh,dim=1)
		df = (self._embedding_space[None,:,None] - harmonics[:,None,:])
		W: torch.Tensor = torch.exp(-(df*alpha)**2).sum(dim=2)
		self.fspace, sspace = spectral_space(self.cfg, self.device)
		hfilter: torch.Tensor = matmul(spectral_projection,W) / self._embedding_space.shape[0]

		self.log.info(f" ----- embedding_space{list(self._embedding_space.shape)}: {self._embedding_space.min():.3f} -> {self._embedding_space.max():.3f}")
		self.log.info(f" ----- fh: {fh}")
		self.log.info(f" ----- df{list(df.shape)}: {df.min():.3f} -> {df.max():.3f}")
		self.log.info(f" ----- W{list(W.shape)}: {W.min():.5f} -> {W.max():.5f}")
		self.log.info(f" ----- spectral_projection{list(spectral_projection.shape)}: {spectral_projection.min():.5f} -> {spectral_projection.max():.5f}")

		return hfilter[:,:sspace.shape[0]]

	def get_harmonic_weights1(self, f ) -> torch.Tensor:
		hw = []
		for ih in range(1, self.nharmonics + 1):
			df: torch.Tensor = self.alpha*(self._embedding_space-f*ih)/f
			hw.append( torch.exp(-df**2 ) )
		W = torch.stack(hw,dim=1).sum(dim=1)
		return W

	def gaussian_harmonics(self, espace: torch.Tensor, harmonics: torch.Tensor ) -> torch.Tensor:
		df = self.alpha*(espace-harmonics[None,:])/espace
		W: torch.Tensor = torch.exp(-df**2).sum(dim=1)
		return W

	def get_harmonic_weights(self, f ):
		espace: torch.Tensor = self._embedding_space[:,None]
		forward_harmonics =  torch.FloatTensor( [f * ih for ih in range(1, self.nharmonics+1)] ).to(self.device)
		W1: torch.Tensor = self.gaussian_harmonics(espace, forward_harmonics )
		backward_harmonics =  torch.FloatTensor( [f/ih for ih in range(1, self.nharmonics+1)] ).to(self.device)
		W2: torch.Tensor = self.gaussian_harmonics(espace, backward_harmonics)
		return W1 # - W2

	def embed(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs) -> Tensor:
		self.log.info(f"SpectralAutocorrelationLayer:")
		spectral_features: torch.Tensor = super(HarmonicsFilterLayer, self).embed( ts, ys, **kwargs)
		spectral_projection: torch.Tensor = torch.sqrt(torch.sum(spectral_features ** 2, dim=1)).squeeze()
		espace: torch.Tensor = self._embedding_space
		self.fspace, sspace = spectral_space(self.cfg, self.device)
		hfilter = []
		for f in self.fspace:
			W = self.get_harmonic_weights1(f)
			hfilter.append( torch.dot(W,espace) )
		return torch.FloatTensor(hfilter).to(self.device)

	def embed0(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs ) -> Tensor:
		self.log.info(f"SpectralAutocorrelationLayer:")
		spectral_features: torch.Tensor = super(HarmonicsFilterLayer, self).embed( ts, ys, **kwargs)
		spectral_projection: torch.Tensor = torch.sqrt(torch.sum(spectral_features ** 2, dim=1)).squeeze()
		espace: torch.Tensor = self._embedding_space
		self.fspace, sspace = spectral_space(self.cfg, self.device)
		hfilter = []
		f = self.fspace[self.fspace.shape[0]//2]
		W = self.get_harmonic_weights(f)
		return W[:sspace.shape[0]]

#	"crtl-mouse-press", x = event.xdata, y = event.ydata, ax = event.inaxes
	def process_event(self, **kwargs ):
		self.log.info(f"           *** ---- HarmonicsFilterLayer.process_event: {kwargs} ")
		if kwargs["id"] == "crtl-mouse-press":
			self.f0 = kwargs["x"]
			self.log.info(f"           *** ---- set f0 = {self.f0} ")

	def magnitude(self, embedding: Tensor, **kwargs) -> np.ndarray:
		mag: np.ndarray = embedding.cpu().numpy()
		return mag.squeeze()

	@property
	def xdata(self) -> np.ndarray:
		return self.fspace


