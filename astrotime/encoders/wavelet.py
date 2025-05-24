import random, time, numpy as np
import torch, math
from astrotime.util.math import shp
from typing import List, Tuple, Mapping
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, device, nn
from .embedding import EmbeddingLayer
from astrotime.util.math import log2space, tnorm
from astrotime.util.logging import elapsed
from astrotime.util.interpolation import interp1d

def clamp( idx: int ) -> int: return max( 0, idx )

def tclamp( x: Tensor ) -> Tensor:
	return torch.where( x < 0.0, torch.zeros_like(x), x )

def pnorm1( x: Tensor ) -> Tensor:
	x = torch.where( x < 0.0, torch.zeros_like(x), x )
	return x / torch.sum( x, dim=-1, keepdim=True )

def pnorm( x: Tensor ) -> Tensor:
	x0: Tensor = x.min( dim=-1, keepdim=True )[0]
	x1: Tensor = x.max( dim=-1, keepdim=True )[0]
	return (x-x0)/(x1-x0)

def embedding_space( cfg: DictConfig, device: device ) -> Tuple[np.ndarray,Tensor]:
	base_freq =  cfg.base_freq
	fold_harmonic = cfg.get('fold_harmonic', True)
	nharmonic_octaves = 1 if fold_harmonic else 0
	noctaves =  cfg.noctaves + nharmonic_octaves
	top_freq = base_freq + base_freq*2**noctaves
	nfreq = cfg.nfreq_oct * noctaves
	nfspace = log2space( base_freq, top_freq, nfreq )
	tfspace = torch.FloatTensor( nfspace ).to(device)
	return nfspace, tfspace

def wavelet_analysis_projection( ts: np.ndarray, ys: np.ndarray, fspace: np.ndarray, cfg: DictConfig, device ) -> np.ndarray:
	t: Tensor = torch.from_numpy( ts[None,:] if ts.ndim == 1 else ts )
	y: Tensor = torch.from_numpy( ys[None,:] if ys.ndim == 1 else ys )
	embedding_space: Tensor = torch.from_numpy(fspace)
	proj = WaveletAnalysisLayer( cfg, embedding_space, device )
	embedding = proj.embed( t, y )
	return proj.magnitude( embedding )

class WaveletSynthesisLayer(EmbeddingLayer):

	def __init__(self, cfg: DictConfig, embedding_space: Tensor, device: device):
		EmbeddingLayer.__init__(self, cfg, embedding_space, device)
		self.C = cfg.decay_factor / (8 * math.pi ** 2)
		self.init_log(f"WaveletSynthesisLayer: nfreq={self.nfreq} ")

	def embed(self, ts: torch.Tensor, ys: torch.Tensor) -> Tensor:
		t0 = time.time()
		self.init_log(f"WaveletSynthesisLayer shapes:")
		slen: int = self.series_length if (self.series_length > 0) else ys.shape[1]
		ones: Tensor = torch.ones(ys.shape[0], self.nfreq, slen, device=self.device)
		tau = 0.5 * (ts[:, self.series_length // 2] + ts[:, self.series_length // 2 + 1])
		self.init_log(f" ys{list(ys.shape)} ts{list(ts.shape)} tau{list(tau.shape)}")
		tau: Tensor = tau[:, None, None]
		omega = self._embedding_space * 2.0 * math.pi
		omega_: Tensor = omega[None, :, None]  # broadcast-to(self.batch_size,self.nfreq,self.series_length)
		ts: Tensor = ts[:, None, :]  # broadcast-to(self.batch_size,self.nfreq,self.series_length)
		dt: Tensor = (ts - tau)
		dz: Tensor = omega_ * dt
		weights: Tensor = torch.exp(-self.C * dz ** 2) if (self.cfg.decay_factor > 0.0) else 1.0
		sum_w: Tensor = torch.sum(weights, dim=-1) if (self.cfg.decay_factor > 0.0) else 1.0

		def w_prod(xs: Tensor, ys: Tensor) -> Tensor:
			return torch.sum(weights * xs * ys, dim=-1) / sum_w

		theta: Tensor = omega_ * ts
		sin_basis: Tensor = torch.sin(theta)
		cos_basis: Tensor = torch.cos(theta)

		sin_one: Tensor = w_prod(sin_basis, ones)
		cos_one: Tensor = w_prod(cos_basis, ones)
		sin_cos: Tensor = w_prod(sin_basis, cos_basis)
		sin_sin: Tensor = w_prod(sin_basis, sin_basis)
		cos_cos: Tensor = w_prod(cos_basis, cos_basis)
		self.init_log(f" --> sin_one{list(sin_one.shape)} cos_one{list(cos_one.shape)} sin_cos{list(sin_cos.shape)} sin_sin{list(sin_sin.shape)}")

		numerator: Tensor = 2 * (sin_cos - sin_one * cos_one)
		denominator: Tensor = (cos_cos - cos_one ** 2) - (sin_sin - sin_one ** 2)
		time_shift: Tensor = torch.atan2(numerator, denominator) / (2 * omega)  # Eq. (S5)
		time_shift_: Tensor = time_shift[:, :, None]  # broadcast-to(self.batch_size,self.nfreq,self.series_length)
		self.init_log(f" --> omega{list(omega.shape)} time_shift{list(time_shift.shape)} tau{list(tau.shape)}")

		sin_shift: Tensor = torch.sin(omega_ * (ts - time_shift_))
		cos_shift: Tensor = torch.cos(omega_ * (ts - time_shift_))
		sin_tau_center: Tensor = torch.sin(omega * (time_shift - tau[:, :, 0]))
		cos_tau_center: Tensor = torch.cos(omega * (time_shift - tau[:, :, 0]))
		self.init_log(f" --> cos_tau_center{list(cos_tau_center.shape)} sin_tau_center{list(sin_tau_center.shape)}")

		ys_cos_shift: Tensor = w_prod(ys, cos_shift)
		ys_sin_shift: Tensor = w_prod(ys, sin_shift)
		ys_one: Tensor = w_prod(ys, ones)
		self.init_log(f" --> ys_one{list(ys_one.shape)} ys{list(ys.shape)} ones{list(ones.shape)}")

		cos_shift_one: Tensor = w_prod(cos_shift, ones)
		sin_shift_one: Tensor = w_prod(sin_shift, ones)
		self.init_log(f" --> sin_shift_one{list(sin_shift_one.shape)} cos_shift_one{list(cos_shift_one.shape)}")

		A: Tensor = 2 * (ys_cos_shift - ys_one * cos_shift_one)
		B: Tensor = 2 * (ys_sin_shift - ys_one * sin_shift_one)
		self.init_log(f" --> A{list(A.shape)} B{list(B.shape)} ")

		a0: Tensor = ys_one
		a1: Tensor = cos_tau_center * A - sin_tau_center * B  # Eq. (S6)
		a2: Tensor = sin_tau_center * A + cos_tau_center * B  # Eq. (S7)
		self.init_log(f" --> a0{list(a0.shape)} a1{list(a1.shape)} a2{list(a2.shape)}")

		wwp: Tensor = a1 ** 2 + a2 ** 2
		phase: Tensor = torch.atan2(a2, a1)
		self.init_log(f"WaveletSynthesisLayer: wwp{list(wwp.shape)}({torch.mean(wwp):.2f},{torch.std(wwp):.2f}), phase{list(phase.shape)}({torch.mean(phase):.2f},{torch.std(phase):.2f})")
		rv = torch.concat( (wwp[:, None, :] , phase[:, None, :]), dim=1)
		self.init_log(f" Completed embedding in {elapsed(t0):.5f} sec: result{list(rv.shape)}")
		self.init_state = False
		return rv

	def magnitude(self, embedding: Tensor) -> Tensor:
		return embedding[:,0,:]

class WaveletAnalysisLayer(EmbeddingLayer):

	def __init__(self, name: str,  cfg, embedding_space: Tensor, device: device):
		EmbeddingLayer.__init__(self, name, cfg, embedding_space, device)
		self.C: float = cfg.decay_factor / (8 * math.pi ** 2)
		self.init_log(f"WaveletAnalysisLayer: nfreq={self.nfreq} ")
		self.subbatch_size: int = cfg.get('subbatch_size',-1)
		self.fold_harmonic =  self.cfg.get('fold_harmonic', True)
		self.nharmonics: int = 1 if self.fold_harmonic else 0
		self.noctaves: int = self.cfg.noctaves
		self.nfreq_oct: int = self.cfg.nfreq_oct

	@property
	def xdata(self) -> Tensor:
		return self._embedding_space[:self.output_series_length]

	@property
	def output_series_length(self) -> int:
		return self.nf

	def sbatch(self, ts: torch.Tensor, ys: torch.Tensor, subbatch: int) -> tuple[Tensor,Tensor]:
		sbr = [ subbatch*self.subbatch_size, (subbatch+1)*self.subbatch_size ]
		return ts[sbr[0]:sbr[1]], ys[sbr[0]:sbr[1]]

	def embed(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs) -> Tensor:
		if ys.ndim == 1:
			return self.embed_subbatch( ts[None,:], ys[None,:] )
		elif self.subbatch_size <= 0:
			return self.embed_subbatch( ts, ys )
		else:
			nsubbatches = math.ceil(ys.shape[0]/self.subbatch_size)
			subbatches = [ self.embed_subbatch( *self.sbatch(ts,ys,i), **kwargs ) for i in range(nsubbatches) ]
			return torch.concat( subbatches, dim=0 )

	def embed_subbatch(self, ts: torch.Tensor, ys: torch.Tensor, **kwargs ) -> Tensor:
		t0 = time.time()
		self.init_log(f"WaveletAnalysisLayer shapes: ts{list(ts.shape)} ys{list(ys.shape)}")
		slen: int = ys.shape[1]
		tau = 0.5 * (ts[:, slen // 2] + ts[:, slen // 2 + 1])
		tau: Tensor = tau[:, None, None]
		omega = self._embedding_space * 2.0 * math.pi
		omega_: Tensor = omega[None, :, None]  # broadcast-to(self.batch_size,self.nfreq,slen)
		ts: Tensor = ts[:, None, :]  # broadcast-to(self.batch_size,self.nfreq,slen)
		dt: Tensor = (ts - tau)
		self.init_log(f" tau{list(tau.shape)} dt{list(dt.shape)}")
		dz: Tensor = omega_ * dt
		weights: Tensor = torch.exp(-self.C * dz ** 2) if (self.cfg.decay_factor > 0.0) else 1.0
		sum_w: Tensor = torch.sum(weights, dim=-1) if (self.cfg.decay_factor > 0.0) else 1.0

		def w_prod( x0: Tensor, x1: Tensor) -> Tensor:
			return torch.sum(weights * x0 * x1, dim=-1) / sum_w

		pw1: Tensor = torch.sin(dz)
		pw2: Tensor = torch.cos(dz)
		p1: Tensor = w_prod(ys, pw1)
		p2: Tensor = w_prod(ys, pw2)
		mag: Tensor =  torch.sqrt( p1**2 + p2**2 )
	#	phase: Tensor = torch.atan2(p1, p2)
#		features =  [ f[:,:self.nf] for f in [p1,p2,mag,phase]]
		features = [ mag[:, :self.nf] ]
		if self.fold_harmonic: features.append( self.fold_harmonic_layer(mag) )
		embedding: Tensor = torch.stack( features, dim=1)
		self.init_log(f" Completed embedding{list(embedding.shape)} in {elapsed(t0):.5f} sec: nfeatures={embedding.shape[1]}")
		self.init_state = False
		return tnorm(embedding,dim=2)

	def fold_harmonic_layer(self, mag: Tensor) -> Tensor:      # [Batch,NF]
		l0: Tensor = mag[:,:self.nf]
		dfH: int = self.nfreq_oct
		harmonic: Tensor = mag[:,dfH:self.nf+dfH]
		return tnorm(harmonic*l0,dim=1)

	@property
	def nf(self):
		return self.noctaves * self.nfreq_oct

	# def fold_harmonic_layers(self, embedding: Tensor, **kwargs) -> Tensor:      # [Batch,NF]
	# 	if self.nharmonics <= 0:
	# 		return embedding
	# 	else:
	# 		threshold = kwargs.get('threshold', self.fold_threshold)
	# 		mag = pnorm( torch.sqrt(torch.sum(embedding ** 2, dim=1)) )
	# 		nf0 = self.noctaves * self.nfreq_oct
	# 		full_freq: Tensor = self._embedding_space[None,:].expand(mag.shape)
	# 		base_freq = full_freq[:,:nf0]
	# 		l0: Tensor = mag[:,:nf0]
	# 		flayers =  l0  if self.sum_features else [ l0 ]
	# 		for iH in range(2,self.nharmonics+2):
	# 			octave: float = math.log2(iH)
	# 			if octave.is_integer():
	# 				dfH: int = self.nfreq_oct*int(octave)
	# 				harmonic: Tensor = mag[:,dfH:nf0+dfH]
	# 				print(f"harmonic-{iH}({octave}): h{shp(harmonic)}, mag{shp(mag)}, l0{shp(l0)} dfH={dfH} nf0={nf0}")
	# 			else:
	# 				harmonic: Tensor = tclamp( interp1d( full_freq, mag, iH*base_freq ) )
	#
	# 			harmonic =  torch.where( l0 < threshold, torch.zeros_like(harmonic), harmonic )
	# 			if self.sum_features:   flayers = flayers + harmonic
	# 			else:                   flayers.append( harmonic )
	# 		return flayers[:,None,:] if self.sum_features else torch.stack( flayers, dim=1 )

	@property
	def nfeatures(self):
		return 2 if self.fold_harmonic else 1

	def magnitude(self, embedding: Tensor) -> np.ndarray:
		self.init_log(f" -> Embedding magnitude{embedding.shape}")
		return embedding.cpu().numpy()

class WaveletProjConvLayer(EmbeddingLayer):

	def __init__(self, cfg, embedding_space: Tensor, device: device):
		EmbeddingLayer.__init__(self, cfg, embedding_space, device)
		self._nfeatures = cfg.nfeatures
		self.nk = cfg.nkernels
		self.K = cfg.kernel_size
		self.ktime_spacing = cfg.kernel_time_spacing
		self.C = math.log( cfg.envelope_reduction_factor )
		self.weights = nn.Parameter( Tensor( self.nfreq*3, self._nfeatures ) )
		self.init_log(f"WaveletProjConvLayer: nfreq={self.nfreq} ")

	def get_tau(self, ts: torch.Tensor ) -> tuple[Tensor,Tensor]:
		taus: torch.Tensor =   ts[:,0][:,None] + (self.ktime_spacing/2)*torch.arange(2,self.nk+2)[None,:]
		diff: torch.Tensor = torch.abs( taus[:,:,None] - ts[:,None,:] )
		time_indices: torch.Tensor = torch.argmin(diff, dim=2)
		return taus, time_indices

	def embed(self, ts: torch.Tensor, ys: torch.Tensor) -> Tensor:
		t0 = time.time()
		self.init_log(f"WaveletProjConvLayer shapes:")
		self.init_log(f" ys{list(ys.shape)} ts{list(ts.shape)}")
		tau, tidx = self.get_tau(ts)
		self.init_log(f" tau{list(tau.shape)} time_indices{list(tidx.shape)}")
		self.init_log(f" K//2={self.K//2} ")
		tys: Tensor = torch.concatenate([ts[:,None,:],ys],dim=1)
		kernel_inputs = torch.stack( [ torch.stack( [ tys[ ib, :, tidx[ib,kidx]-self.K//2 : tidx[ib,kidx]+self.K//2+1 ] for kidx in range(self.nk) ] ) for ib in range(ys.shape[0]) ] )
		dt: Tensor = kernel_inputs[:,:,0:1,:] - tau[:,:,None,None]
		yk: Tensor = kernel_inputs[:,:,0:1,:]
		omega = (self._embedding_space * 2.0 * math.pi)
		z: Tensor = omega[None,None,:,None] * dt
		self.init_log(f" dt{list(dt.shape)} yk{list(yk.shape)} omega{list(omega.shape)} z{list(z.shape)}")
		sdt: Tensor = 2*dt/self.ktime_spacing
		weights: Tensor = torch.exp( -self.C * (sdt**2) )
		sum_w: Tensor = torch.sum(weights, dim=-1)
		self.init_log(f" weights{list(weights.shape)} sdt{list(sdt.shape)} sum_w{list(sum_w.shape)}")

		def w_prod( x0: Tensor, x1: Tensor) -> Tensor:
			return torch.sum(weights * x0 * x1, dim=-1) / sum_w

		pw1: Tensor = torch.sin(z)
		pw2: Tensor = torch.cos(z)
		self.init_log(f" --> pw1{list(pw1.shape)} pw2{list(pw2.shape)}  z{list(z.shape)}  ")
		ones: Tensor = torch.ones( pw1.shape, device=self.device )

		p0: Tensor = w_prod(yk, ones)
		p1: Tensor = w_prod(yk, pw1)
		p2: Tensor = w_prod(yk, pw2)
		self.init_log(f" --> p0{list(p0.shape)} p1{list(p1.shape)} p2{list(p2.shape)}")
		projection: Tensor = torch.stack( (p0,p1,p2), dim=3).reshape( [p0.shape[0],p0.shape[1],3*p0.shape[2]] )
		self.init_log(f" Completed embedding in {elapsed(t0):.5f} sec: result{list(projection.shape)}")
		result: Tensor  = (projection[...,None] * self.weights[None,None,:,:]).sum(dim=2)
		self.init_state = False
		return result

	@property
	def projection_dims(self) -> int:
		return 3

	@property
	def output_series_length(self):
		return self.cfg.nfreq