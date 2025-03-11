import math
import torch
from .util import compute_dist_mat
from .gaussian_process import GaussianProcess
torch.set_default_tensor_type(torch.DoubleTensor)

class SpectralMixture(GaussianProcess):
	"""
	Class that inherits from the base Gp class, and defines the function init_params, compute_kernel and
	compute_spectral_density for the kernel defined in
	Wilson, A., & Adams, R. (2013, May). Gaussian process kernels for pattern discovery and extrapolation.
	Params of the kernel are:
		"mixtures_weight": weight of each of the mixture
		"mixtures_scale": std of the mixtures (inverse of lengthscales)
		"mixtures_mean": central frequency of the mixtures
	"""
	def __init__(self, x, y, sn_range_limits=(1e-3, math.inf), sn_range_init=(1e-3, 1), n_gm=10, max_f_init=None):
		"""
		Init the object
		Args:
			x (numpy.array or torch.Tensor): feature of the data that will be stored in the gp memory
			y (numpy.array or torch.Tensor): target values of the data that will be stored in the gp memory
			sn_range_init (tuple): min and max value of the signal noise used for initialization (random uniform). Shape=(2,)
			sn_range_limits (tuple): min and max value allowed for the signal noise. Shape=(2,)
			n_gm (int): number of gaussian mixtures for the model
			max_f_init (float or None): maximum frequency used for initialization of the mean gaussian mixture.
										If None, the maximum frequency possible in the data will be used
		"""
		super(SpectralMixture, self).__init__(x, y, sn_range_limits, sn_range_init)
		# the keys in the dict kernel params limits must correspond to the keys in kernel_params
		# l_range can be specified for each dimension separately or for all dimensions at the same time
		self.n_gm = n_gm

		self.max_f_init = max_f_init

		self.mixtures_weight = None
		self.mixtures_scale = None
		self.mixtures_mean = None

	def init_params(self):
		"""
		Define and return (random) initial values for the parameters of the kernel.
		Returns:
			kernel_params (dict): Contains the parameters name and value. The values must be torch.Tensor
									The keys are:
										"mixtures_weight": weight of each of the mixture
										"mixtures_scale": std of the mixtures (inverse of lengthscales)
										"mixtures_mean": central frequency of the mixtures
			sn (torch.Tensor): Std of the likelihood
		"""
		dists = compute_dist_mat(self.x, self.x).flatten(0, 1)
		dist_max = dists.max(0).values

		sn = torch.Tensor(self.sn_range_init[0] + torch.rand(1) * (self.sn_range_init[1] - self.sn_range_init[0]))
		if self.max_f_init is not None:
			mixtures_mean = torch.rand(self.n_gm, self.x.shape[1]) * self.max_f_init  # * 0.5 / dist_min
		else:
			# if the max frequency is not provided to the object, set it to 0.5/ dist_min in the data,
			# which is the highest frequency that can appear
			dist_min = torch.where(dists.eq(0.0), torch.tensor(1.0e10, dtype=self.x.dtype, device=self.x.device), dists).min(0).values
			mixtures_mean = torch.rand(self.n_gm, self.x.shape[1]) * 0.5 / dist_min
		# Inverse of lengthscales should be drawn from truncated Gaussian | N(0, max_dist^2)
		mixtures_scale = torch.randn(self.n_gm, self.x.shape[1]).mul_(dist_max).abs_().reciprocal_()
		mixtures_weight = self.y.std().div(self.n_gm)[None].repeat(self.n_gm)

		# impose the first component of mixtures_mean to be low_frequency
		mixtures_mean[0, :] = 0.5 / dist_max

		kernel_params = {
			"mixtures_weight": mixtures_weight,
			"mixtures_scale": mixtures_scale,
			"mixtures_mean": mixtures_mean
		}
		return kernel_params, sn

	@staticmethod
	def compute_kernel(dist_mat, kernel_params):
		"""
		Compute the covariance matrix given the distance matrix between points
		distance matrix shape is (N, N, p)
		where N is the number of points, p the number of features
		"""
		dist_mat = dist_mat[None, ...]
		v_matrix = kernel_params["mixtures_scale"].pow(2)[:, None, None, :]
		mu_matrix = kernel_params["mixtures_mean"][:, None, None, :]
		w_matrix = kernel_params["mixtures_weight"][:, None, None]
		temp_prod = torch.exp(-2 * math.pi ** 2 * dist_mat.pow(2) * v_matrix) * torch.cos(2 * math.pi * dist_mat * mu_matrix)
		temp_sum = temp_prod.prod(-1)
		compo_corr_matrix = w_matrix * temp_sum
		corr_matrix = compo_corr_matrix.sum(0)
		return corr_matrix

	@staticmethod
	def compute_spectral_density(max_f, idx_feat, n_pts, kernel_params):
		"""
		Compute and return the spectral density of the kernel
		Args:
			max_f (float): maximum frequency for which to compute the spectral density
			idx_feat (int): feature index for which to represent the density function.
							max=fn where fn is the number of features of x
			n_pts (int): number of points between 0 and max_f for which to compute the spectral density
			kernel_params (dict): Contains the kernel parameters.
									The keys are:
										"mixtures_weight": weight of each of the mixture
										"mixtures_scale": std of the mixtures (inverse of lengthscales)
										"mixtures_mean": central frequency of the mixtures
		Returns:
			f_axis (torch.Tensor): axis containing the value of the points at which the spectral density have been computed
			spectre (torch.Tensor): log of the spectral density, computed at the points of f_axis
		"""
		v = kernel_params["mixtures_scale"].pow(2)
		with torch.no_grad():
			try:
				max_f_feat = max_f[idx_feat]
			except:
				max_f_feat = max_f
			f_axis = torch.arange(0, max_f_feat, max_f_feat / n_pts)
			spectre_compo = 1 / torch.sqrt(2 * math.pi * v[None, :, idx_feat]) * \
							torch.exp(- torch.pow((f_axis[:, None] - kernel_params["mixtures_mean"][None, :, idx_feat]), 2) / (2 * v[None, :, idx_feat]))
			spectre = torch.log((kernel_params["mixtures_weight"][None, ...] * spectre_compo).sum(-1))
		return f_axis, spectre

