import random, time, numpy as np
import torch, math
from sympy import Tuple
from torch import Tensor, device
from .embedding import EmbeddingLayer
from astrotime.util.math import logspace, tnorm
from astrotime.util.logging import elapsed

class WaveletEmbeddingLayer(EmbeddingLayer):

	def __init__(self, cfg, device: device):
		EmbeddingLayer.__init__(self,cfg,device)
		self.nfreq = cfg.nfreq
		self.C = 1 / (8 * math.pi ** 2)
		fspace = logspace if (self.cfg.fscale == "log") else np.linspace
		self.freq = torch.FloatTensor( fspace( self.cfg.freq_start, self.cfg.freq_end, self.cfg.nfreq ) ).to(self.device)
		self.ones: Tensor = None
		self.init_log(f"WaveletEmbeddingLayer: nfreq={self.nfreq} ")

	def embed(self, ts: torch.Tensor, ys: torch.Tensor ) -> Tensor:
		t0 = time.time()
		self.init_log(f"WaveletEmbeddingLayer shapes:")
		if self.ones is None:
			self.ones: Tensor = torch.ones( ys.shape[0], self.nfreq, self.series_length, device=self.device)
		tau = 0.5 * (ts[:, self.series_length // 2] + ts[:, self.series_length // 2 + 1])
		self.init_log(f" ys{list(ys.shape)} ts{list(ts.shape)} tau{list(tau.shape)}")
		tau: Tensor = tau[:, None, None]
		omega = self.freq * 2.0 * math.pi
		omega_: Tensor = omega[None, :, None]  # broadcast-to(self.batch_size,self.nfreq,self.series_length)
		ts: Tensor = ts[:, None, :]  # broadcast-to(self.batch_size,self.nfreq,self.series_length)
		dt: Tensor = (ts - tau)
		dz: Tensor = omega_ * dt
		weights: Tensor = torch.exp(-self.C * dz ** 2)
		sum_w: Tensor = torch.sum(weights, dim=-1)

		def w_prod(xs: Tensor, ys: Tensor) -> Tensor:
			return torch.sum(weights * xs * ys, dim=-1) / sum_w

		theta: Tensor = omega_ * ts
		sin_basis: Tensor = torch.sin(theta)
		cos_basis: Tensor = torch.cos(theta)

		sin_one: Tensor = w_prod(sin_basis, self.ones)
		cos_one: Tensor = w_prod(cos_basis, self.ones)
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
		ys_one: Tensor = w_prod(ys, self.ones)
		self.init_log(f" --> ys_one{list(ys_one.shape)} ys{list(ys.shape)} ones{list(self.ones.shape)}")

		cos_shift_one: Tensor = w_prod(cos_shift, self.ones)
		sin_shift_one: Tensor = w_prod(sin_shift, self.ones)
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
		self.init_log(f"WaveletEmbeddingLayer: wwp{list(wwp.shape)}({torch.mean(wwp):.2f},{torch.std(wwp):.2f}), phase{list(phase.shape)}({torch.mean(phase):.2f},{torch.std(phase):.2f})")
		rv = torch.concat( (wwp[:, None, :] , phase[:, None, :]), dim=1)
		self.log.info(f" Completed embedding in {elapsed(t0):.5f} sec: result{list(rv.shape)}")
		return rv

	@property
	def nfeatures(self) -> int:
		return 2

class WaveletProjectionLayer(EmbeddingLayer):

	def __init__(self, cfg, device: device):
		EmbeddingLayer.__init__(self,cfg,device)
		self.nfreq = cfg.nfreq
		self.C = 1 / (8 * math.pi ** 2)
		fspace = logspace if (self.cfg.fscale == "log") else np.linspace
		self.freq = torch.FloatTensor( fspace( self.cfg.freq_start, self.cfg.freq_end, self.cfg.nfreq ) ).to(self.device)
		self.ones: Tensor = None
		self.init_log(f"WaveletProjectionLayer: nfreq={self.nfreq} ")

	def embed(self, ts: torch.Tensor, ys: torch.Tensor ) -> Tensor:
		t0 = time.time()
		self.init_log(f"WaveletProjectionLayer shapes:")
		if self.ones is None:
			self.ones: Tensor = torch.ones( ys.shape[0], self.nfreq, self.series_length, device=self.device)
		tau = 0.5 * (ts[:, self.series_length // 2] + ts[:, self.series_length // 2 + 1])
		self.init_log(f" ys{list(ys.shape)} ts{list(ts.shape)} tau{list(tau.shape)}")
		tau: Tensor = tau[:, None, None]
		omega = self.freq * 2.0 * math.pi
		omega_: Tensor = omega[None, :, None]  # broadcast-to(self.batch_size,self.nfreq,self.series_length)
		ts: Tensor = ts[:, None, :]  # broadcast-to(self.batch_size,self.nfreq,self.series_length)
		dt: Tensor = (ts - tau)
		self.init_log(f" ys{list(ys.shape)} ts{list(ts.shape)} tau{list(tau.shape)} dt{list(dt.shape)}")
		dz: Tensor = omega_ * dt
		weights: Tensor = torch.exp(-self.C * dz ** 2)
		sum_w: Tensor = torch.sum(weights, dim=-1)

		def w_prod( x0: Tensor, x1: Tensor) -> Tensor:
			return torch.sum(weights * x0 * x1, dim=-1) / sum_w

		self.init_log(f" dz{list(dz.shape)} weights{list(weights.shape)} sum_w{list(sum_w.shape)}")
		pw1: Tensor = torch.sin(dz)
		pw2: Tensor = torch.cos(dz)
		self.init_log(f" --> pw0{list(self.ones.shape)} pw1{list(pw1.shape)} pw2{list(pw2.shape)}  ")

		p0: Tensor = w_prod(ys, self.ones)
		p1: Tensor = w_prod(ys, pw1)
		p2: Tensor = w_prod(ys, pw2)
		self.init_log(f" --> p0{list(p0.shape)} p1{list(p1.shape)} p2{list(p2.shape)}")

		rv: Tensor = torch.concat( (p0[:, None, :], p1[:, None, :], p2[:, None, :]), dim=1)
		self.log.info(f" Completed embedding in {elapsed(t0):.5f} sec: result{list(rv.shape)}")
		return rv

	@property
	def nfeatures(self) -> int:
		return 3

class WaveletProjConvLayer(EmbeddingLayer):

	def __init__(self, cfg, device: device):
		EmbeddingLayer.__init__(self,cfg,device)
		self.nfreq = cfg.nfreq
		self.K = cfg.kernel_size
		self.C = 1 / (8 * math.pi ** 2)
		fspace = logspace if (self.cfg.fscale == "log") else np.linspace
		self.freq = torch.FloatTensor( fspace( self.cfg.freq_start, self.cfg.freq_end, self.cfg.nfreq ) ).to(self.device)
		self.ones: Tensor = None
		self.init_log(f"WaveletProjConvLayer: nfreq={self.nfreq} ")

	def get_tau(self, ts: torch.Tensor ) -> tuple[Tensor,Tensor]:
		NK: int =  self.series_length // self.K
		dt: float = (ts[-1]-ts[0]).item() / NK
		tau: torch.Tensor = torch.arange( dt/2, ts[-1].item(), dt )
		diff: torch.Tensor = torch.abs(tau.unsqueeze(1) - ts)
		time_indices: torch.Tensor = torch.argmin(diff, dim=1)
		return tau, time_indices

	def embed(self, ts: torch.Tensor, ys: torch.Tensor ) -> Tensor:
		t0 = time.time()
		self.init_log(f"WaveletProjConvLayer shapes:")
		if self.ones is None:
			self.ones: Tensor = torch.ones( ys.shape[0], self.nfreq, self.series_length, device=self.device)

		self.init_log(f" ys{list(ys.shape)} ts{list(ts.shape)}")
		tau, time_indices = self.get_tau(ts)
		self.init_log(f" tau{list(tau.shape)} time_indices{list(time_indices.shape)}")
		omega = self.freq * 2.0 * math.pi
		omega_: Tensor = omega[None, :, None]  # broadcast-to(self.batch_size,self.nfreq,self.series_length)
		ts: Tensor = ts[:, None, :].unfold(2, self.K, self.K)         # [B,1,NK,K]
		tau = 0.5 * (ts[:, self.series_length // 2] + ts[:, self.series_length // 2 + 1])
		dt: Tensor = (ts - tau)
		self.init_log(f" ys{list(ys.shape)} ts{list(ts.shape)} tau{list(tau.shape)} dt{list(dt.shape)}")
		dz: Tensor = omega_ * dt
		weights: Tensor = torch.exp(-self.C * dz ** 2)
		sum_w: Tensor = torch.sum(weights, dim=-1)

		def w_prod( x0: Tensor, x1: Tensor) -> Tensor:
			return torch.sum(weights * x0 * x1, dim=-1) / sum_w

		self.init_log(f" dz{list(dz.shape)} weights{list(weights.shape)} sum_w{list(sum_w.shape)}")
		pw1: Tensor = torch.sin(dz)
		pw2: Tensor = torch.cos(dz)
		self.init_log(f" --> pw0{list(self.ones.shape)} pw1{list(pw1.shape)} pw2{list(pw2.shape)}  ")

		p0: Tensor = w_prod(ys, self.ones)
		p1: Tensor = w_prod(ys, pw1)
		p2: Tensor = w_prod(ys, pw2)
		self.init_log(f" --> p0{list(p0.shape)} p1{list(p1.shape)} p2{list(p2.shape)}")

		rv: Tensor = torch.concat( (p0[:, None, :], p1[:, None, :], p2[:, None, :]), dim=1)
		self.log.info(f" Completed embedding in {elapsed(t0):.5f} sec: result{list(rv.shape)}")
		return rv

	@property
	def nfeatures(self) -> int:
		return 3