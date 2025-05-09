# Copied from pyro
import math
import numbers
from typing import Callable, List, Optional, Tuple, Union
import torch
from torch.fft import irfft, rfft
from .tensor_ops import next_fast_len

def _compute_chain_variance_stats(input):
    # compute within-chain variance and variance estimator
    # input has shape N x C x sample_shape
    N = input.size(0)
    chain_var = input.var(dim=0)
    var_within = chain_var.mean(dim=0)
    var_estimator = (N - 1) / N * var_within
    if input.size(1) > 1:
        chain_mean = input.mean(dim=0)
        var_between = chain_mean.var(dim=0)
        var_estimator = var_estimator + var_between
    else:
        # to make rho_k is the same as autocorrelation when num_chains == 1
        # in computing effective_sample_size
        var_within = var_estimator
    return var_within, var_estimator


def gelman_rubin(input, chain_dim=0, sample_dim=1):
    """
    Computes R-hat over chains of samples. It is required that
    ``input.size(sample_dim) >= 2`` and ``input.size(chain_dim) >= 2``.

    :param torch.Tensor input: the input tensor.
    :param int chain_dim: the chain dimension.
    :param int sample_dim: the sample dimension.
    :returns torch.Tensor: R-hat of ``input``.
    """
    assert input.dim() >= 2
    assert input.size(sample_dim) >= 2
    assert input.size(chain_dim) >= 2
    # change input.shape to 1 x 1 x input.shape
    # then transpose sample_dim with 0, chain_dim with 1
    sample_dim = input.dim() + sample_dim if sample_dim < 0 else sample_dim
    chain_dim = input.dim() + chain_dim if chain_dim < 0 else chain_dim
    assert chain_dim != sample_dim
    input = input.reshape((1, 1) + input.shape)
    input = input.transpose(0, sample_dim + 2).transpose(1, chain_dim + 2)

    var_within, var_estimator = _compute_chain_variance_stats(input)
    rhat = (var_estimator / var_within).sqrt()
    return rhat.squeeze(max(sample_dim, chain_dim)).squeeze(min(sample_dim, chain_dim))


def split_gelman_rubin(input, chain_dim=0, sample_dim=1):
    """
    Computes R-hat over chains of samples. It is required that
    ``input.size(sample_dim) >= 4``.

    :param torch.Tensor input: the input tensor.
    :param int chain_dim: the chain dimension.
    :param int sample_dim: the sample dimension.
    :returns torch.Tensor: split R-hat of ``input``.
    """
    assert input.dim() >= 2
    assert input.size(sample_dim) >= 4
    # change input.shape to 1 x 1 x input.shape
    # then transpose chain_dim with 0, sample_dim with 1
    sample_dim = input.dim() + sample_dim if sample_dim < 0 else sample_dim
    chain_dim = input.dim() + chain_dim if chain_dim < 0 else chain_dim
    assert chain_dim != sample_dim
    input = input.reshape((1, 1) + input.shape)
    input = input.transpose(0, chain_dim + 2).transpose(1, sample_dim + 2)

    N_half = input.size(1) // 2
    new_input = torch.stack([input[:, :N_half], input[:, -N_half:]], dim=1)
    new_input = new_input.reshape((-1, N_half) + input.shape[2:])
    split_rhat = gelman_rubin(new_input)
    return split_rhat.squeeze(max(sample_dim, chain_dim)).squeeze(
        min(sample_dim, chain_dim)
    )


def autocorrelation( input: torch.Tensor, dim=0 ):
    """
    Computes the autocorrelation of samples at dimension ``dim``.

    Reference: https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation

    :param torch.Tensor input: the input tensor.
    :param int dim: the dimension to calculate autocorrelation.
    :returns torch.Tensor: autocorrelation of ``input``.
    """
    # Adapted from Stan implementation
    # https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/autocorrelation.hpp
    N = input.size(dim)
    M = next_fast_len(N)
    M2 = 2 * M

    # transpose dim with -1 for Fourier transform
    input = input.transpose(dim, -1)

    # centering and padding x
    centered_signal = input - input.mean(dim=-1, keepdim=True)

    # Fourier transform
    freqvec = torch.view_as_real(rfft(centered_signal, n=M2))
    # take square of magnitude of freqvec (or freqvec x freqvec*)
    freqvec_gram = freqvec.pow(2).sum(-1)
    # inverse Fourier transform
    autocorr = irfft(freqvec_gram, n=M2)

    # truncate and normalize the result, setting autocorrelation to 1 for all
    # constant channels
    autocorr = autocorr[..., :N]
    autocorr = autocorr / torch.tensor(
        range(N, 0, -1), dtype=input.dtype, device=input.device
    )
    variance = autocorr[..., :1]
    constant = (variance == 0).expand_as(autocorr)
    autocorr = autocorr / variance.clamp(min=torch.finfo(variance.dtype).tiny)
    autocorr[constant] = 1

    # transpose back to original shape
    ac = autocorr.transpose(dim, -1)
    print(f"autocorr{ac.shape}")
    return ac


def autocovariance(input, dim=0):
    """
    Computes the autocovariance of samples at dimension ``dim``.

    :param torch.Tensor input: the input tensor.
    :param int dim: the dimension to calculate autocorrelation.
    :returns torch.Tensor: autocorrelation of ``input``.
    """
    return autocorrelation(input, dim) * input.var(dim, unbiased=False, keepdim=True)


def _cummin(input):
    """
    Computes cummulative minimum of input at dimension ``dim=0``.

    :param torch.Tensor input: the input tensor.
    :returns torch.Tensor: accumulate min of `input` at dimension `dim=0`.
    """
    # FIXME: is there a better trick to find accumulate min of a sequence?
    N = input.size(0)
    input_tril = input.unsqueeze(0).repeat((N,) + (1,) * input.dim())
    triu_mask = (
        torch.ones(N, N, dtype=input.dtype, device=input.device)
        .triu(diagonal=1)
        .reshape((N, N) + (1,) * (input.dim() - 1))
    )
    triu_mask = triu_mask.expand((N, N) + input.shape[1:]) > 0.5
    input_tril.masked_fill_(triu_mask, input.max())
    return input_tril.min(dim=1)[0]


def effective_sample_size(input, chain_dim=0, sample_dim=1):
    """
    Computes effective sample size of input.

    Reference:

    [1] `Introduction to Markov Chain Monte Carlo`,
        Charles J. Geyer

    [2] `Stan Reference Manual version 2.18`,
        Stan Development Team

    :param torch.Tensor input: the input tensor.
    :param int chain_dim: the chain dimension.
    :param int sample_dim: the sample dimension.
    :returns torch.Tensor: effective sample size of ``input``.
    """
    assert input.dim() >= 2
    assert input.size(sample_dim) >= 2
    # change input.shape to 1 x 1 x input.shape
    # then transpose sample_dim with 0, chain_dim with 1
    sample_dim = input.dim() + sample_dim if sample_dim < 0 else sample_dim
    chain_dim = input.dim() + chain_dim if chain_dim < 0 else chain_dim
    assert chain_dim != sample_dim
    input = input.reshape((1, 1) + input.shape)
    input = input.transpose(0, sample_dim + 2).transpose(1, chain_dim + 2)

    N, C = input.size(0), input.size(1)
    # find autocovariance for each chain at lag k
    gamma_k_c = autocovariance(input, dim=0)  # N x C x sample_shape

    # find autocorrelation at lag k (from Stan reference)
    var_within, var_estimator = _compute_chain_variance_stats(input)
    rho_k = (var_estimator - var_within + gamma_k_c.mean(dim=1)) / var_estimator
    rho_k[0] = 1  # correlation at lag 0 is always 1

    # initial positive sequence (formula 1.18 in [1]) applied for autocorrelation
    Rho_k = rho_k if N % 2 == 0 else rho_k[:-1]
    Rho_k = Rho_k.reshape((N // 2, 2) + Rho_k.shape[1:]).sum(dim=1)

    # separate the first index
    Rho_init = Rho_k[0]

    if Rho_k.size(0) > 1:
        # Theoretically, Rho_k is positive, but due to noise of correlation computation,
        # Rho_k might not be positive at some point. So we need to truncate (ignore first index).
        Rho_positive = Rho_k[1:].clamp(min=0)

        # Now we make the initial monotone (decreasing) sequence.
        Rho_monotone = _cummin(Rho_positive)

        # Formula 1.19 in [1]
        tau = -1 + 2 * Rho_init + 2 * Rho_monotone.sum(dim=0)
    else:
        tau = -1 + 2 * Rho_init

    n_eff = C * N / tau
    return n_eff.squeeze(max(sample_dim, chain_dim)).squeeze(min(sample_dim, chain_dim))


def resample(input, num_samples, dim=0, replacement=False):
    """
    Draws ``num_samples`` samples from ``input`` at dimension ``dim``.

    :param torch.Tensor input: the input tensor.
    :param int num_samples: the number of samples to draw from ``input``.
    :param int dim: dimension to draw from ``input``.
    :returns torch.Tensor: samples drawn randomly from ``input``.
    """
    weights = torch.ones(input.size(dim), dtype=input.dtype, device=input.device)
    indices = torch.multinomial(weights, num_samples, replacement)
    return input.index_select(dim, indices)


def quantile(input, probs, dim=0):
    """
    Computes quantiles of ``input`` at ``probs``. If ``probs`` is a scalar,
    the output will be squeezed at ``dim``.

    :param torch.Tensor input: the input tensor.
    :param list probs: quantile positions.
    :param int dim: dimension to take quantiles from ``input``.
    :returns torch.Tensor: quantiles of ``input`` at ``probs``.
    """
    if isinstance(probs, (numbers.Number, list, tuple)):
        probs = torch.tensor(probs, dtype=input.dtype, device=input.device)
    sorted_input = input.sort(dim)[0]
    max_index = input.size(dim) - 1
    indices = probs * max_index
    # because indices is float, we interpolate the quantiles linearly from nearby points
    indices_below = indices.long()
    indices_above = (indices_below + 1).clamp(max=max_index)
    quantiles_above = sorted_input.index_select(dim, indices_above)
    quantiles_below = sorted_input.index_select(dim, indices_below)
    shape_to_broadcast = [1] * input.dim()
    shape_to_broadcast[dim] = indices.numel()
    weights_above = indices - indices_below.type_as(indices)
    weights_above = weights_above.reshape(shape_to_broadcast)
    weights_below = 1 - weights_above
    quantiles = weights_below * quantiles_below + weights_above * quantiles_above
    return quantiles if probs.shape != torch.Size([]) else quantiles.squeeze(dim)


def weighed_quantile(
    input: torch.Tensor,
    probs: Union[List[float], Tuple[float, ...], torch.Tensor],
    log_weights: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """
    Computes quantiles of weighed ``input`` samples at ``probs``.

    :param torch.Tensor input: the input tensor.
    :param list probs: quantile positions.
    :param torch.Tensor log_weights: sample weights tensor.
    :param int dim: dimension to take quantiles from ``input``.
    :returns torch.Tensor: quantiles of ``input`` at ``probs``.

    **Example:**

    .. doctest::

        >>> input = torch.Tensor([[10, 50, 40], [20, 30, 0]])
        >>> probs = torch.Tensor([0.2, 0.8])
        >>> log_weights = torch.Tensor([0.4, 0.5, 0.1]).log()
        >>> result = weighed_quantile(input, probs, log_weights, -1)
        >>> torch.testing.assert_close(result, torch.Tensor([[40.4, 47.6], [9.0, 26.4]]))
    """
    dim = dim if dim >= 0 else (len(input.shape) + dim)
    if isinstance(probs, (list, tuple)):
        probs = torch.tensor(probs, dtype=input.dtype, device=input.device)
    assert isinstance(probs, torch.Tensor)
    # Calculate normalized weights
    weights = (log_weights - torch.logsumexp(log_weights, 0)).exp()
    # Sort input and weights
    sorted_input, sorting_indices = input.sort(dim)
    weights = weights[sorting_indices].cumsum(dim)
    # Scale weights to be between zero and one
    weights = weights - weights.min(dim, keepdim=True)[0]
    weights = weights / weights.max(dim, keepdim=True)[0]
    # Calculate indices
    indices_above = (
        (weights[..., None] <= probs)
        .sum(dim, keepdim=True)
        .swapaxes(dim, -1)
        .clamp(max=input.size(dim) - 1)[..., 0]
    )
    indices_below = (indices_above - 1).clamp(min=0)
    # Calculate below and above qunatiles
    quantiles_below = sorted_input.gather(dim, indices_below)
    quantiles_above = sorted_input.gather(dim, indices_above)
    # Calculate weights for below and above quantiles
    probs_shape = [None] * dim + [slice(None)] + [None] * (len(input.shape) - dim - 1)
    expanded_probs_shape = list(input.shape)
    expanded_probs_shape[dim] = len(probs)
    probs = probs[probs_shape].expand(*expanded_probs_shape)
    weights_below = weights.gather(dim, indices_below)
    weights_above = weights.gather(dim, indices_above)
    weights_below = (weights_above - probs) / (weights_above - weights_below)
    weights_above = 1 - weights_below
    # Return quantiles
    return weights_below * quantiles_below + weights_above * quantiles_above


def pi(input, prob, dim=0):
    """
    Computes percentile interval which assigns equal probability mass
    to each tail of the interval.

    :param torch.Tensor input: the input tensor.
    :param float prob: the probability mass of samples within the interval.
    :param int dim: dimension to calculate percentile interval from ``input``.
    :returns torch.Tensor: quantiles of ``input`` at ``probs``.
    """
    return quantile(input, [(1 - prob) / 2, (1 + prob) / 2], dim)


def hpdi(input, prob, dim=0):
    """
    Computes "highest posterior density interval" which is the narrowest
    interval with probability mass ``prob``.

    :param torch.Tensor input: the input tensor.
    :param float prob: the probability mass of samples within the interval.
    :param int dim: dimension to calculate percentile interval from ``input``.
    :returns torch.Tensor: quantiles of ``input`` at ``probs``.
    """
    sorted_input = input.sort(dim)[0]
    mass = input.size(dim)
    index_length = int(prob * mass)
    intervals_left = sorted_input.index_select(
        dim,
        torch.tensor(range(mass - index_length), dtype=torch.long, device=input.device),
    )
    intervals_right = sorted_input.index_select(
        dim,
        torch.tensor(range(index_length, mass), dtype=torch.long, device=input.device),
    )
    intervals_length = intervals_right - intervals_left
    index_start = intervals_length.argmin(dim)
    indices = torch.stack([index_start, index_start + index_length], dim)
    return torch.gather(sorted_input, dim, indices)


def _weighted_mean(input, log_weights, dim=0, keepdim=False):
    dim = input.dim() + dim if dim < 0 else dim
    log_weights = log_weights.reshape([-1] + (input.dim() - dim - 1) * [1])
    max_log_weight = log_weights.max(dim=0)[0]
    relative_probs = (log_weights - max_log_weight).exp()
    return (input * relative_probs).sum(dim=dim, keepdim=keepdim) / relative_probs.sum()


def _weighted_variance(input, log_weights, dim=0, keepdim=False, unbiased=True):
    # Ref: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Frequency_weights
    deviation_squared = (
        input - _weighted_mean(input, log_weights, dim, keepdim=True)
    ).pow(2)
    correction = log_weights.size(0) / (log_weights.size(0) - 1.0) if unbiased else 1.0
    return _weighted_mean(deviation_squared, log_weights, dim, keepdim) * correction


def waic(input, log_weights=None, pointwise=False, dim=0):
    """
    Computes "Widely Applicable/Watanabe-Akaike Information Criterion" (WAIC) and
    its corresponding effective number of parameters.

    Reference:

    [1] `WAIC and cross-validation in Stan`,
    Aki Vehtari, Andrew Gelman

    :param torch.Tensor input: the input tensor, which is log likelihood of a model.
    :param torch.Tensor log_weights: weights of samples along ``dim``.
    :param int dim: the sample dimension of ``input``.
    :returns tuple: tuple of WAIC and effective number of parameters.
    """
    if log_weights is None:
        log_weights = torch.zeros(
            input.size(dim), dtype=input.dtype, device=input.device
        )

    # computes log pointwise predictive density: formula (3) of [1]
    dim = input.dim() + dim if dim < 0 else dim
    weighted_input = input + log_weights.reshape([-1] + (input.dim() - dim - 1) * [1])
    lpd = torch.logsumexp(weighted_input, dim=dim) - torch.logsumexp(log_weights, dim=0)

    # computes the effective number of parameters: formula (6) of [1]
    p_waic = _weighted_variance(input, log_weights, dim)

    # computes expected log pointwise predictive density: formula (4) of [1]
    elpd = lpd - p_waic
    waic = -2 * elpd
    return (waic, p_waic) if pointwise else (waic.sum(), p_waic.sum())


def fit_generalized_pareto(X):
    """
    Given a dataset X assumed to be drawn from the Generalized Pareto
    Distribution, estimate the distributional parameters k, sigma using a
    variant of the technique described in reference [1], as described in
    reference [2].

    References
    [1] 'A new and efficient estimation method for the generalized Pareto distribution.'
    Zhang, J. and Stephens, M.A. (2009).
    [2] 'Pareto Smoothed Importance Sampling.'
    Aki Vehtari, Andrew Gelman, Jonah Gabry

    :param torch.Tensor: the input data X
    :returns tuple: tuple of floats (k, sigma) corresponding to the fit parameters
    """
    if not isinstance(X, torch.Tensor) or X.dim() != 1:
        raise ValueError("Input X must be a 1-dimensional torch tensor")

    X = X.double()
    X = torch.sort(X, descending=False)[0]

    N = X.size(0)
    M = 30 + int(math.sqrt(N))

    # b = k / sigma
    bs = 1.0 - math.sqrt(M) / (torch.arange(1, M + 1, dtype=torch.double) - 0.5).sqrt()
    bs /= 3.0 * X[int(N / 4 - 0.5)]
    bs += 1 / X[-1]

    ks = torch.log1p(-bs.unsqueeze(-1) * X).mean(-1)
    Ls = N * (torch.log(-bs / ks) - (ks + 1.0))

    weights = torch.exp(Ls - Ls.unsqueeze(-1))
    weights = 1.0 / weights.sum(-1)

    not_small_weights = weights > 1.0e-30
    weights = weights[not_small_weights]
    bs = bs[not_small_weights]
    weights /= weights.sum()

    b = (bs * weights).sum().item()
    k = torch.log1p(-b * X).mean().item()
    sigma = -k / b
    k = k * N / (N + 10.0) + 5.0 / (N + 10.0)

    return k, sigma


def crps_empirical(pred, truth):
    """
    Computes negative Continuous Ranked Probability Score CRPS* [1] between a
    set of samples ``pred`` and true data ``truth``. This uses an ``n log(n)``
    time algorithm to compute a quantity equal that would naively have
    complexity quadratic in the number of samples ``n``::

        CRPS* = E|pred - truth| - 1/2 E|pred - pred'|
              = (pred - truth).abs().mean(0)
              - (pred - pred.unsqueeze(1)).abs().mean([0, 1]) / 2

    Note that for a single sample this reduces to absolute error.

    **References**

    [1] Tilmann Gneiting, Adrian E. Raftery (2007)
        `Strictly Proper Scoring Rules, Prediction, and Estimation`
        https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

    :param torch.Tensor pred: A set of sample predictions batched on rightmost dim.
        This should have shape ``(num_samples,) + truth.shape``.
    :param torch.Tensor truth: A tensor of true observations.
    :return: A tensor of shape ``truth.shape``.
    :rtype: torch.Tensor
    """
    if pred.shape[1:] != (1,) * (pred.dim() - truth.dim() - 1) + truth.shape:
        raise ValueError(
            "Expected pred to have one extra sample dim on left. "
            "Actual shapes: {} versus {}".format(pred.shape, truth.shape)
        )
    opts = dict(device=pred.device, dtype=pred.dtype)
    num_samples = pred.size(0)
    if num_samples == 1:
        return (pred[0] - truth).abs()

    pred = pred.sort(dim=0).values
    diff = pred[1:] - pred[:-1]
    weight = torch.arange(1, num_samples, **opts) * torch.arange(
        num_samples - 1, 0, -1, **opts
    )
    weight = weight.reshape(weight.shape + (1,) * (diff.dim() - 1))

    return (pred - truth).abs().mean(0) - (diff * weight).sum(0) / num_samples**2


def energy_score_empirical(
    pred: torch.Tensor,
    truth: torch.Tensor,
    pred_batch_size: Optional[int] = None,
    cdist: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.cdist,
) -> torch.Tensor:
    r"""
    Computes negative Energy Score ES* (see equation 22 in [1]) between a
    set of multivariate samples ``pred`` and a true data vector ``truth``. Running time
    is quadratic in the number of samples ``n``. In case of univariate samples
    the output coincides with the CRPS::

        ES* = E|pred - truth| - 1/2 E|pred - pred'|

    Note that for a single sample this reduces to the Euclidean norm of the difference between
    the sample ``pred`` and the ``truth``.

    This is a strictly proper score so that for ``pred`` distirbuted according to a
    distribution :math:`P` and ``truth`` distributed according to a distribution :math:`Q`
    we have :math:`ES^{*}(P,Q) \ge ES^{*}(Q,Q)` with equality holding if and only if :math:`P=Q`, i.e.
    if :math:`P` and :math:`Q` have the same multivariate distribution (it is not sufficient for
    :math:`P` and :math:`Q` to have the same marginals in order for equality to hold).

    **References**

    [1] Tilmann Gneiting, Adrian E. Raftery (2007)
        `Strictly Proper Scoring Rules, Prediction, and Estimation`
        https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

    :param torch.Tensor pred: A set of sample predictions batched on the second leftmost dim.
        The leftmost dim is that of the multivariate sample.
    :param torch.Tensor truth: A tensor of true observations with same shape as ``pred`` except
        for the second leftmost dim which can have any value or be omitted.
    :param int pred_batch_size: If specified the predictions will be batched before calculation
        according to the specified batch size in order to reduce memory consumption.
    :param callable cdist: Function for calculating an euclidean distance (see
        https://github.com/pytorch/pytorch/issues/42479 for why you might need to change this in order to
        balance speed versus accuracy). Default is :any:`torch.cdist`.

    :return: A tensor of shape ``truth.shape``.
    :rtype: torch.Tensor
    """
    if pred.dim() == (truth.dim() + 1):
        remove_leftmost_dim = True
        truth = truth[..., None, :]
    elif pred.dim() == truth.dim():
        remove_leftmost_dim = False
    else:
        raise ValueError(
            "Expected pred to have at most one extra dim versus truth."
            "Actual shapes: {} versus {}".format(pred.shape, truth.shape)
        )

    if pred_batch_size is None:
        retval = (
            cdist(pred, truth).mean(dim=-2)
            - 0.5 * cdist(pred, pred).mean(dim=[-1, -2])[..., None]
        )
    else:
        # Divide predictions into batches
        pred_len = pred.shape[-2]
        pred_batches = []
        while pred.numel() > 0:
            pred_batches.append(pred[..., :pred_batch_size, :])
            pred = pred[..., pred_batch_size:, :]
        # Calculate predictions distance to truth
        retval = (
            torch.stack(
                [cdist(pred_batch, truth).sum(dim=-2) for pred_batch in pred_batches],
                dim=0,
            ).sum(dim=0)
            / pred_len
        )
        # Calculate predictions self distance
        for aux_pred_batch in pred_batches:
            retval = (
                retval
                - 0.5
                * torch.stack(
                    [
                        cdist(pred_batch, aux_pred_batch).sum(dim=[-1, -2])
                        for pred_batch in pred_batches
                    ],
                    dim=0,
                ).sum(dim=0)[..., None]
                / pred_len
                / pred_len
            )

    if remove_leftmost_dim:
        retval = retval[..., 0]

    return retval