import math

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

def read_co2_mlo():
    """
    Read and return x and y data from "monthly_in_situ_co2_mlo.csv"
    The data is processed (points with y negative are removed)
    Returns:
        x (numpy.array): feature vector of dimension (N,) where N is the number of points
        y (numpy.array): target vector of dimension (N,) where N is the number of points
    """
    data = np.array(pd.read_csv("monthly_in_situ_co2_mlo.csv", header=56))
    y = data[:, 4].astype(np.float32)
    x = data[:, 0].astype(np.float32) * 12 + data[:, 1].astype(np.float32)
    inds_keep = np.nonzero(y > 0)[0]

    y = y[inds_keep]
    x = x[inds_keep]
    return x, y


def generate_sync_pattern(x_min=-15, x_max=15, pattern_dist=10, x_step=0.25):
    """
    Generate and return the sum of sinc pattern, spaced by pattern_dist, between x_min and x_max with a step of x_step
    Returns:
        x (numpy.array): x values at which the function is evaluated. Shape=(N,)
        y (numpy.array): value of the function, used as target values. Shape=(N,)
    """
    min_pattern = (x_min // pattern_dist) * pattern_dist
    max_pattern = (x_max // pattern_dist + 1) * pattern_dist
    x = np.arange(x_min, x_max, x_step)
    x_s = [(x - pattern_dist) for pattern_dist in range(min_pattern, max_pattern + 1, pattern_dist)]
    x_s = np.array(x_s)
    y = np.sinc(x_s).sum(0)
    return x, y


def split_train_val_test(x, y, prop_train=0.8, prop_val=0.):
    """
    Split the input data in 3 sets: train, val, test.
    Args:
        x (numpy.array): feature vector of dimension (N, f) or (N,) where N is the number of data points and f the number of features
        y (numpy.array): target vector of dimension (N,) where N is the number of data points
        prop_train (float): proportion of data used for training
        prop_val (float): proportion of data used for validation
    Returns:
        x_train (numpy.array): feature vector used for training. Shape=(N * prop_train,) or (N * prop_train, f)
        y_train (numpy.array): target vector used for training. Shape=(N * prop_train,)
        x_val (numpy.array): feature vector used for validation. Shape=(N * prop_val,) or (N * prop_val, f).
        y_val (numpy.array): target vector used for validation. Shape=(N * prop_val,).
        x_test (numpy.array): feature vector used for test.
                                Shape=(N * (1 - prop_train - prop_val),) or (N * (1 - prop_train - prop_val), f)
        y_test (numpy.array): target vector used for test. Shape=(N * (1 - prop_train - prop_val),)
    """
    num_train = int(x.shape[0] * prop_train)
    num_val = int(x.shape[0] * prop_val)
    x_train = x[:num_train]
    y_train = y[:num_train]
    x_val = x[num_train: (num_train + num_val)]
    y_val = y[num_train: (num_train + num_val)]
    x_test = x[(num_train + num_val):]
    y_test = y[(num_train + num_val):]
    return x_train, y_train, x_val, y_val, x_test, y_test


def generate_array_pred(x_max, x_min, border_prop=0.1, step_prop=1e-3):
    """
    Generate and return an array containing the values between x_max and x_min, with an added border, and a step defined as
    (x_max - x_min) * step_prop
    """
    border = (x_max - x_min) * border_prop
    min_repr = x_min - border
    max_repr = x_max + border
    tick = (x_max - x_min) * step_prop
    x_pred = np.arange(min_repr, max_repr, tick)
    return x_pred


def generate_string_kernel_params(kernel_params):
    """"""
    params_str = ""
    for key in kernel_params.keys():
        if type(kernel_params[key] == torch.Tensor):
            values = kernel_params[key].detach()
        else:
            values = kernel_params[key]
        params_str += "\n" + key + ": " + str(values)
    return params_str


def compute_dist_mat(x1, x2):
    """
    Compute the l1 distance matrix between two feature vectors, where the entry (a, b) represents the distance
    between x1[a] and x2[b]
    Args:
        x1 (torch.Tensor): feature vector 1. Shape = (n1, f)
        x2 (torch.Tensor): feature vector 2. Shape = (n2, f)

    Returns:
        dist_mat (torch.Tensor): distance matrix. Shape = (n1, n2)
    """
    return (x1[None, ...].transpose(1, 0) - x2[None, ...]).abs()


def compute_posterior(k_xs_xs, k_xs_x, y, inv_k):
    """
    Compute and return the mean and covariance matrix of the prediction.
    Args:
        k_xs_xs (torch.Tensor): prior covariance matrix between the points to predict. Shape=(Np, Np)
        k_xs_x (torch.Tensor): covariance matrix between the points to predict and the points in memory. Shape=(Np, Nm)
        y (torch.Tensor): target values of the points in memory. Shape=(Nm,)
        inv_k (torch.Tensor): inverse of the covariance matrix between points in memory. Shape=(Nm, Nm)
    Returns:
        mean_pred (torch.Tensor): mean of the prediction distribution. Shape=(Np,)
        cov_pred (torch.Tensor): updated covariance matrix between the points to predict. Shape=(Np, Np)
    """
    coef_pred = k_xs_x @ inv_k
    mean_pred = coef_pred @ y
    reduc_cov = k_xs_x @ inv_k @ k_xs_x.t()
    cov_pred = k_xs_xs - reduc_cov
    return mean_pred, cov_pred


def plot_results(x_train, x_val, x_test, x_pred, y_train, y_val, y_test, y_pred, y_pred_cov):
    """
    Plot the predictions along with its confidence interval.
    Show the ground truth points with x marker, with different color for training, validation and test
    Args:
        x_train (torch.Tensor): input training x data. Shape = (Ntr,)
        x_val (torch.Tensor or None): validation x data. Shape = (Nv,). If is Null, will not be plotted
        x_test (torch.Tensor): test x data. Shape = (Nte,)
        x_pred (torch.Tensor): prediction x data. Shape=(Np,)
        y_train (torch.Tensor): target values for the training data. Shape=(Ntr,)
        y_val  (torch.Tensor or None): target values for the validation data. Shape=(Nval,).
                                        If is None, it will not be plotted
        y_test (torch.Tensor): target values for the test data. Shape=(Nte,)
        y_pred (torch.Tensor): predicted mean values. Shape=(Np,)
        y_pred_cov (torch.Tensor): covariance matrix of the predicted points. Shape=(Np, Np)
    """
    plt.figure()
    plt.fill_between(x_pred, y_pred - 3 * np.sqrt(y_pred_cov.diagonal()), y_pred + 3 * np.sqrt(y_pred_cov.diagonal()),
                     color="y")
    plt.plot(x_pred, y_pred)
    plt.scatter(x_train, y_train, color="k", marker='x', s=10, label='train')
    if len(x_val) != 0 and len(y_val) != 0:
        plt.scatter(x_val, y_val, color="g", marker='x', s=10, label='val')
    plt.scatter(x_test, y_test, color="r", marker='x', s=10, label='test')
    plt.legend()
    plt.show()