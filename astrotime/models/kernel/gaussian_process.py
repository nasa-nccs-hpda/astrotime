import math
import torch
import matplotlib.pyplot as plt
from .util import generate_string_kernel_params, compute_posterior, compute_dist_mat
torch.set_default_tensor_type(torch.DoubleTensor)

class GaussianProcess:
    """
        Base class for gaussian processes prediction.
        To be usable, another class must be created that inherits from this class and specify the
        functions: init_params, compute_kernel, compute_spectral_density
        The kernel parameters are stored in a dictionary, so that it can easily be extended for different kernels
        An example of inheritance can be found in Gp_se.py for the squared exponential kernel
        and Gp_pattern.py for the kernel defined in
        Wilson, A., & Adams, R. (2013, May). Gaussian process kernels for pattern discovery and extrapolation.
    """
    def __init__(self, x, y, sn_range_limits=(1e-3, math.inf), sn_range_init=(1e-3, 1.)):
        """
        Initialize the gp object with the x and y data.
        y data is normalized between 0 and 1 (and is unnormalized for the prediction)
        Args:
            x (numpy.array or torch.tensor or list):
                feature vector. Shape=(n, f) or (n,) where n is the number of points and f the number of features

            y (numpy.array or torch.tensor or list):
                target values. Shape=(n) where n is the number of points

            sn_range_limits (list, tuple, torch.Tensor or numpy.array):
                lower and upper values of the signal noise std

            sn_range_init (list, tuple, torch.Tensor or numpy.array):
                lower and upper values of the signal noise std random initialization values.
                It must be in between the limits above
        """
        if type(x) != torch.Tensor:
            x = torch.Tensor(x)
        if type(y) != torch.Tensor:
            y = torch.Tensor(y)
        if x.dim() == 1:
            x = x.reshape((x.shape[0], 1))

        self.sn_range_limits = torch.Tensor(sn_range_limits)
        self.sn_range_init = torch.Tensor(sn_range_init)

        self.sn = None
        self.kernel_params = {}
        self.kernel_params_limits = {}

        self.x = x

        self.y_min = y.min(0).values
        self.y_max = y.max(0).values
        self.y = self.normalize_y(y)

        self.inv_k = None

    def init_params(self):
        """
        Define and return (random) initial values for the parameters of the kernel.
        Return values of the kernel parameters in a dict and the signal noise.
        It must be implemented for the kernel of the gaussian process that inherits this class
        """
        raise NotImplementedError
        kernel_params = {}
        sn = 1e-3
        return kernel_params, sn

    @staticmethod
    def compute_kernel(dist_mat, kernel_params):
        """
        Given a distance matrix (l1) between two set of points of dimension (N1, N2),
        and the values of the kernel parameters in a dictionary,
        return the covariance matrix between the two set of points.
        It must be implemented for the gaussian process that inherits this class.
        Args:
            dist_mat (torch.Tensor): l1 distance matrix between two sets of points. Shape=(N1, N2)
            kernel_params (dict): Contains the name and value of the kernel parameters (eg.: {'l': 1e-3, 'sf': 1e-1})
        """
        raise NotImplementedError

    @staticmethod
    def compute_spectral_density(kernel_params, idx_feat, max_f=1, n_pts=1e4):
        """
        Compute and return the spectral density of the kernel.
        It must be implemented for the gaussian process that inherits this class.
        Args:
            max_f (float): maximum frequency for which to compute the spectral density
            idx_feat (int): feature index for which to represent the density function.
                            max=fn where fn is the number of features of x
            n_pts (int): number of points between 0 and max_f for which to compute the spectral density
            kernel_params (dict): Contains the kernel parameters.
                                    The keys are:
                                        "l": lengthscales
                                        "sf": signal std
        Returns:
            f_axis (torch.Tensor): axis containing the value of the points at which the spectral density have been computed
            spectre (torch.Tensor): log of the spectral density, computed at the points of f_axis
        """
        raise NotImplementedError

    def transform_constraints(self, kernel_params, sn):
        """
        Return a dict containing the kernel parameters and signal noise, transformed with logit and log
        This is used to allow bounded optimization, by optimizing the transformed parameters instead of their raw values.
        self.kernel_params_limits is used to read the limit of the parameters and self.sn_range_limits for the limits of sn
        self.kernel_params_limits must have the same keys as kernel_params,
        but the last dimension correspond to the min max values: index 0 is the min and index 1 the max.
        If a key from self.kernel_params is not in self.kernel_params_limit, it is transformed with log instead of logit
        Args:
            kernel_params (dict): the kernel parameters with structure {'name': value}, where values are torch.Tensor
            sn (torch.Tensor): signal noise. Shape=(1,)

        Returns:
            kernel_params_t (dict): the transformed kernel parameters with structure {'name': value}, where values are torch.Tensor
            sn_t (torch.Tensor): transformed signal noise. Shape=(1,)
        """
        kernel_params_t = {}
        for key in kernel_params.keys():
            if key in self.kernel_params_limits.keys():
                if self.kernel_params_limits[key][..., 1] != math.inf:
                    kernel_params_t[key] = torch.logit(
                        (kernel_params[key] - self.kernel_params_limits[key][..., 0]) /
                            (self.kernel_params_limits[key][..., 1] - self.kernel_params_limits[key][..., 0]))
                else:
                    kernel_params_t[key] = torch.log(kernel_params[key] - self.kernel_params_limits[key][..., 0])
            else:
                kernel_params_t[key] = torch.log(kernel_params[key])

        if self.sn_range_limits[1] != math.inf:
            sn_t = torch.logit((sn - self.sn_range_limits[0]) / (self.sn_range_limits[1] - self.sn_range_limits[0]))
        else:
            sn_t = torch.log(sn - self.sn_range_limits[0])
        return kernel_params_t, sn_t

    def inverse_constraints(self, kernel_params_t, sn_t):
        """
        Return a dict containing the original kernel parameters and signal noise, given their transformed values.
        self.kernel_params_limits is used to read the limit of the parameters and self.sn_range_limits for the limits of sn
        self.kernel_params_limits must have the same keys as kernel_params,
        but the last dimension correspond to the min max values: index 0 is the min and index 1 the max
        Args:
            kernel_params_t (dict): the transformed kernel parameters with structure {'name': value}, where values are torch.Tensor
            sn_t (torch.Tensor): transformed signal noise

        Returns:
            kernel_params (dict): the kernel parameters with structure {'name': value}, where values are torch.Tensor
            sn (torch.Tensor): signal noise. Shape=(1,)
        """
        kernel_params = {}
        for key in kernel_params_t.keys():
            if key in self.kernel_params_limits.keys():
                if self.kernel_params_limits[key][..., 1] != math.inf:
                    kernel_params[key] = torch.sigmoid(kernel_params_t[key]) * \
                        (self.kernel_params_limits[key][..., 1] - self.kernel_params_limits[key][..., 0]) + \
                        self.kernel_params_limits[key][..., 0]
                else:
                    kernel_params[key] = torch.exp(kernel_params_t[key]) + self.kernel_params_limits[key][..., 0]
            else:
                kernel_params[key] = torch.exp(kernel_params_t[key])
        if self.sn_range_limits[1] != math.inf:
            sn = torch.sigmoid(sn_t) * (self.sn_range_limits[1] - self.sn_range_limits[0]) + self.sn_range_limits[0]
        else:
            sn = torch.exp(sn_t) + self.sn_range_limits[0]
        return kernel_params, sn

    def normalize_y(self, y):
        """
        Normalize y using the stored y_min and y_max inside the object, so that it is bounded between 0 and 1
        Args:
            y (torch.Tensor): un-normalized target value. Shape=(Ny,)
        Returns:
            y_norm (torch.Tensor): normalized target value. Shape=(Ny,)
        """
        return (torch.Tensor(y) - self.y_min) / (self.y_max - self.y_min)

    def denormalize_y(self, y_norm):
        """
        Denormalize y using the stored y_min and y_max inside the object.
        Args:
            y_norm (torch.Tensor): normalized target value. Shape=(Ny,)
        Returns:
            y (torch.Tensor): un-normalized target value. Shape=(Ny,)
        """
        return torch.Tensor(y_norm) * (self.y_max - self.y_min) + self.y_min

    def compute_inv_k(self, dist_mat, sn, kernel_params):
        """
        Compute the covariance matrix, its cholesky factorization and inverse.
        Cholesky factorization allows to perform the inverse with more stability,
        and compute the determinant rapidly (using the product of the diagonal terms).
        The inverse of the covariance matrix is needed for inference, and computation of
        the marginal likelihood (used as the metric to optimize).
        Args:
            dist_mat (torch.Tensor): l1 distance matrix between two identical sets of points. Shape=(N, N)
            sn (torch.Tensor): signal noise. Shape=(1,)
        Returns:
            inv_k (torch.Tensor): inverse of the covariance matrix. Shape=(N, N)
            k_cholesky (torch.Tensor): Cholesky decomposition of the covariance matrix. Shape=(N, N)
        """
        k = self.compute_kernel(dist_mat=dist_mat, kernel_params=kernel_params)
        I_in = torch.eye(dist_mat.shape[0], dist_mat.shape[0])
        k_cholesky = torch.linalg.cholesky(k + sn.pow(2) * I_in)
        inv_k = torch.cholesky_inverse(k_cholesky)
        return inv_k, k_cholesky

    def predict(self, xs, x=None, y=None, kernel_params=None, inv_k=None, return_numpy=True, denormalize=True):
        """
        Predict the values for the feature xs.
        If the parameters are not provided,
        it is read from the object instead (since all parameters are stored in the object after training)
        Args:
            xs (torch.Tensor): Input features for which to predict the target value. Shape=(Np, Nf) or (Np,) if Nf is 1
            x (torch.Tensor or None): features of the points in the gp memory. Shape = (Nm, Nf)
            y (torch.Tensor or None): targets in the gp memory shape= (Nm,)
            kernel_params (dict or None): the kernel parameters with structure {'name': value}, where values are torch.Tensor
            inv_k (torch.Tensor): inverse of the covariance matrix. Shape=(Nm, Nm)
            return_numpy (bool): if set to True, the function will return numpy arrays
            denormalize (bool): if set to True, the predictions will be un-normalized
        Returns:
            mean_pred (torch.Tensor): mean value of the predictions. Shape=(Np,)
            cov_pred (torch.Tensor): covariance matrix of the predictions. Shape=(Np, Np).
                                    The sqrt of the diagonal terms can be used to represent the uncertainty
        """
        if type(xs) != torch.Tensor:
            xs = torch.Tensor(xs)
        # x must have 2 dimensions: (N, f) where N is the number of points in memory and f the number of features.
        # if xs is only 1d, resize it to (N, 1)
        if xs.dim() == 1:
            xs = xs.reshape((xs.shape[0], 1))
        # If kernel_params, inv_k, x or y are not provided to the function, take the parameters from the object,
        # which are stored after the training
        if kernel_params is None:
            if len(list(self.kernel_params.keys())) == 0:
                raise ValueError(
                    "You must first train the Gp before using the predict function if you do not specify kernel_params")
            else:
                kernel_params = self.kernel_params

        if inv_k is None:
            if self.inv_k is None:
                raise ValueError(
                    "You must first train the Gp before using the predict function if you do not specify inv_k")
            else:
                inv_k = self.inv_k

        if x is None:
            x = self.x

        if y is None:
            y = self.y

        dist_mat_xsxs = compute_dist_mat(xs, xs)
        k_xs_xs = self.compute_kernel(dist_mat_xsxs, kernel_params)

        dist_mat_xsx = compute_dist_mat(xs, x)
        k_xs_x = self.compute_kernel(dist_mat_xsx, kernel_params)

        mean_pred, cov_pred = compute_posterior(k_xs_xs, k_xs_x, y, inv_k)

        if denormalize:
            mean_pred = self.denormalize_y(mean_pred)
            cov_pred = cov_pred * (self.y_max - self.y_min) ** 2
        if return_numpy:
            mean_pred = mean_pred.detach().numpy()
            cov_pred = cov_pred.detach().numpy()

        return mean_pred, cov_pred

    def compute_log_lk_out(self, x_out, y_out, x_in, y_in, inv_k, kernel_params):
        """
        Compute the predictions at points outside the gp memory and compute the log likelihood given the
        stored data,inverse of covariance matrix and kernel parameters.
        Args:
            x_out (torch.Tensor): Features of the observed data at which to compute the predictions. Shape=(No, Nf)
            y_out (torch.Tensor): Targets of the observed data, which will be compared with the predictions (No,)
            x_in (torch.Tensor): Features of the data in the gp memory (Nm, Nf)
            y_in (torch.Tensor): Targets in the gp memory (Nm,)
            inv_k (torch.Tensor): inverse of the covariance matrix. Shape=(Nm, Nm)
            kernel_params (dict): the kernel parameters with structure {'name': value}, where values are torch.Tensor
        Returns:
            log_lk (torch.Tensor): negative log likelihood of the observed data. Should be minimized. Shape=(1,)
            fit_term (torch.Tensor): term in the log likelihood representing the fit of the predictions.
                                        The lower it is, the better the fit is. Shape=(1,)
            sig_term (torch.Tensor): term in the log likelihood representing the uncertainty of the predictions.
                                        The lower it is, the lower the uncertainty is. Shape= (1,)
        """
        mean_pred_val, cov_pred_val = self.predict(x_out, x_in, y_in, kernel_params=kernel_params, inv_k=inv_k,
                                                   return_numpy=False, denormalize=False)
        sig_i_2 = cov_pred_val.diag()
        fit_term = ((y_out - mean_pred_val).pow(2) / (2 * sig_i_2)).sum(0)
        sig_term = 0.5 * torch.log(sig_i_2).sum(0)
        log_lk = (fit_term + sig_term + 0.5 * math.log(2 * math.pi))
        return log_lk, fit_term, sig_term

    def compute_log_marg_lk_in(self, y, inv_k, k_cholesky):
        """
        Compute the log marginal likelihood of the data inside the gp memory
        Args:
            y (torch.Tensor): Features of the data in the gp memory (N,)
            inv_k (torch.Tensor): inverse of the covariance matrix. Shape=(N, N)
            k_cholesky (torch.Tensor): Cholesky decomposition of the covariance matrix. Shape=(N, N)
        Returns:
            log_marg_lk (torch.Tensor): negative log likelihood of the data in memory. Should be minimized. Shape=(1,)
            fit_term (torch.Tensor): term in the log likelihood representing the fit of the predictions.
                                        The lower it is, the better the fit is. Shape=(1,)
            complexity_term (torch.Tensor): term in the log likelihood representing the complexity of the predictions
                                            (determinant of the covariance matrix). The lower it is, the less complex the predictions are.
                                            It is a regularization term to avoid overfit.
        """
        complexity_term = 2 * k_cholesky.diagonal().log().sum()
        fit_term = y.t() @ inv_k @ y
        # can use torch.cholesky_solve(self.y[..., None], k_cholesky) instead of inv_k @ y,
        # but here the inverse is reused from before
        log_marg_lk = 0.5 * (fit_term + complexity_term + k_cholesky.shape[0] * math.log(2 * math.pi))
        return log_marg_lk, fit_term, complexity_term

    def train(self, x_val=None, y_val=None, n_restarts=5, prop_in=0.5, n_iters=250, lr=1e-3, step_val=5):
        """
        Train the kernel hyperparameters so as to minimize the log likelihood of the data using gradient descent with
        LBFGS.
        At each iteration, some random points are used in the gp memory and the rest are used to verify predictions
        outside the gp memory such that better generalization is obtained.
        The value to minimize is the sum of the negative log marginal likelihood of the data in memory
        and the negative log likelihood of the data outside the memory.
        Multiple restarts are used, and the parameters leading to the best validation loss is kept.
        If x_val and y_val are provided, the validation loss is the negative log likelihood of (x_val, y_val).
        If not, the sum of negative log likelihood of the training data is used for validation instead.

        When the training is finished, the parameters are stored inside self.kernel_params and self.sn.
        self.inv_k is then computed

        Args:
            x_val (numpy.array or torch.Tensor): features of the validation data.
                                            If it is None, the training loss will be used for validation
            y_val (numpy.array or torch.Tensor): targets of the validation data.
                                            If it is None, the training loss will be used for validation
            n_restarts (int): Number of restart with different initialization values
            prop_in (float): proportion of data for each iteration that is used inside the gp memory.
                                It should be high enough, so that pattern are observable in the memory
                                of the gp and low enough so that generalization can be tested.
                                Set it to 1 to take all training points in the gp memory, as is commonly the case
            n_iters (int): number of training iterations
            lr (float): learning rate
            step_val (int): The validation loss will not be computed at each training iteration, but every step_val steps
        """
        best_marg_lk = math.inf
        best_val_loss = math.inf
        best_params = {}
        best_sn = None

        batch_size = int(self.x.shape[0] * prop_in)

        dist_mat = compute_dist_mat(self.x, self.x)

        if len(x_val) != 0 and len(y_val) != 0:
            x_val = torch.Tensor(x_val)
            y_val = self.normalize_y(y_val)
            use_val = True
        else:
            use_val = False

        for idx_restart in range(n_restarts):
            kernel_params, sn = self.init_params()
            # optimize the transformed parameters to allow bounded optimization
            kernel_params_t, sn_t = self.transform_constraints(kernel_params, sn)

            for key in kernel_params_t:
                kernel_params_t[key].requires_grad = True
            sn_t.requires_grad = True

            params = [kernel_params_t[key] for key in kernel_params_t.keys()]
            params.append(sn_t)

            optimizer = torch.optim.LBFGS(
                [{'params': params}],
                lr=lr,
                line_search_fn='strong_wolfe')

            try:
                for idx_iter in range(n_iters):
                    # if prop_in != 1, select some points to train and the rest to validate,
                    # in order to train the kernel parameters with better generalization.
                    # Otherwise, all training points are used to train
                    if prop_in != 1:
                        idx_min = torch.randint(low=0, high=self.x.shape[0] - batch_size, size=(1,))[0]
                        idxs_in = torch.arange(idx_min, idx_min + batch_size)
                        idxs_out = torch.arange(0, self.x.shape[0])
                        idxs_out = torch.cat((idxs_out[:idx_min], idxs_out[idx_min + batch_size:]))
                        dist_mat_in = dist_mat[idxs_in, ...][:, idxs_in, :]
                    else:
                        dist_mat_in = dist_mat
                        idxs_in = torch.arange(0, self.x.shape[0])

                    def closure():
                        optimizer.zero_grad()

                        kernel_params, sn = self.inverse_constraints(kernel_params_t, sn_t)

                        inv_k_in, k_cholesky_in = self.compute_inv_k(dist_mat=dist_mat_in, sn=sn,
                                                                     kernel_params=kernel_params)

                        log_marg_lk_in, fit_term_in, complexity_term_in = \
                            self.compute_log_marg_lk_in(y=self.y[idxs_in], inv_k=inv_k_in, k_cholesky=k_cholesky_in)

                        if prop_in != 1:
                            log_marg_lk_out, fit_term_out, sig_term_out = \
                                self.compute_log_lk_out(x_out=self.x[idxs_out], y_out=self.y[idxs_out], x_in=self.x[idxs_in],
                                                      y_in=self.y[idxs_in], inv_k=inv_k_in, kernel_params=kernel_params)

                            log_marg_lk = log_marg_lk_in + log_marg_lk_out

                            print("Restart: " + str(idx_restart) +
                                  " - iter: " + str(idx_iter) +
                                  " - Log marg: " + str(log_marg_lk.item()) +
                                  " - Log marg in: " + str(log_marg_lk_in.item()) +
                                  " - fit term in: " + str(fit_term_in.item()) +
                                  " - complexity term in: " + str(complexity_term_in.item()) +
                                  " - Log marg out: " + str(log_marg_lk_out.item()) +
                                  " - fit term out: " + str(fit_term_out.item()) +
                                  " - sig term out: " + str(sig_term_out.item()))
                        else:
                            log_marg_lk = log_marg_lk_in
                            print("Restart: " + str(idx_restart) +
                                  " - iter: " + str(idx_iter) +
                                  " - Log marg: " + str(log_marg_lk.item()) +
                                  " - fit term: " + str(fit_term_in.item()) +
                                  " - complexity term: " + str(complexity_term_in.item()))

                        log_marg_lk.backward()

                        return log_marg_lk

                    log_marg_lk = optimizer.step(closure)

                    if idx_iter % step_val == 0:
                        kernel_params, sn = self.inverse_constraints(kernel_params_t, sn_t)
                        if use_val:
                                inv_k = self.compute_inv_k(dist_mat=dist_mat, sn=sn, kernel_params=kernel_params)
                                # k = self.compute_kernel(dist_mat=dist_mat, kernel_params=kernel_params)
                                # inv_k = torch.cholesky_inverse(torch.linalg.cholesky(k + sn.pow(2) * I))
                                val_loss, val_fit_term, val_sig_term = \
                                        self.compute_log_lk_out(x_out=x_val,  y_out=y_val, x_in=self.x,
                                                                     y_in=self.y, inv_k=inv_k, kernel_params=kernel_params)

                                val_loss += float(log_marg_lk)
                        else:
                            val_loss = float(log_marg_lk)

                        if val_loss < best_val_loss:
                            print("val loss improved from " + str(float(best_val_loss)) + " to " + str(float(val_loss)))
                            if use_val:
                                print("val fit term: " + str(val_fit_term.item()) + " - val sig term: " + str(val_sig_term.item()))
                            str_params = generate_string_kernel_params(kernel_params)
                            print(str_params + "\nsn: " + str(sn.item()))

                            best_params = kernel_params
                            best_sn = sn
                            best_marg_lk = log_marg_lk
                            best_val_loss = val_loss

            except Exception as e:
                print(e)

        print("best marg lk: " + str(best_marg_lk) +
                " - best val loss: " + str(best_val_loss) +
                " - best params: " + str(best_params) +
                " - best sn: " + str(best_sn))

        self.kernel_params = best_params
        self.sn = best_sn

        print("Computing inverse...")
        self.inv_k, _ = self.compute_inv_k(dist_mat=dist_mat, sn=self.sn, kernel_params=self.kernel_params)

    def plot_spectral_density(self, max_f, kernel_params=None, n_pts=1e5):
        """
        Plot the spectral density of the learned kernel for each dimension separately
        If kernel_params is None, use the values stored in the object (stored after training)
        Args:
            max_f (float): max frequency to plot (inverse of the minimum l1 distance in the data to see all frequencies)
            kernel_params (dict): the kernel parameters with structure {'name': value}, where values are torch.Tensor
            n_pts (int): number of points that will be used to represent the plot
        """
        if kernel_params is None:
            kernel_params = self.kernel_params

        for idx_feat in range(self.x.shape[1]):
            plt.figure()
            f_axis, spectre = self.compute_spectral_density(max_f=max_f,
                                                              idx_feat=idx_feat,
                                                              n_pts=n_pts,
                                                              kernel_params=kernel_params)
            plt.plot(f_axis, spectre)
            plt.title("Feature " + str(idx_feat))
            plt.xlabel("frequency")
            plt.ylabel("log amplitude")
            plt.show()

    def plot_cov_fct(self, x, kernel_params=None, sn=None):
        """
        Plot the correlation of the learned kernel for each dimension separately in function of the distance between points.
        If kernel_params is None, use the values stored in the object (stored after training)
        Args:
            x (numpy.array or torch.Tensor): feature data, used to find the max and min distance between points,
                                                and the mean values for each dimension. Shape=(Np, Nf) or (Np,) if Nf is 1
            kernel_params (dict): the kernel parameters with structure {'name': value}, where values are torch.Tensor
            sn (torch.Tensor): signal noise. Shape=(1,)
        """
        if kernel_params is None:
            kernel_params = self.kernel_params
        if sn is  None:
            sn = self.sn
        if type(x) != torch.Tensor:
            x = torch.Tensor(x)
        if x.dim() == 1:
            x = x[:, None]
        x_mean = x.mean(0)
        max_dist = x.max(0).values - x.min(0).values
        x_step = (x[1:] - x[:-1]).min(0).values

        for idx_feat in range(self.x.shape[1]):
            # If there is multiple dimensions, look at each dimension separately and fix the other dimensions to the mean
            x_feat = torch.arange(0, max_dist[idx_feat], x_step[idx_feat])
            x_inputs = x_mean[None, :].repeat(x_feat.shape[0], 1)
            x_inputs[:, idx_feat] = x_feat
            dist_mat = compute_dist_mat(x_inputs, x_inputs)
            k = self.compute_kernel(dist_mat=dist_mat, kernel_params=kernel_params).detach()
            correlation = (k[0] / (k[0, 0] + sn.pow(2))).detach()
            plt.figure()
            plt.plot(x_feat, correlation)
            plt.title("Feature " + str(idx_feat))
            plt.xlabel("distance")
            plt.ylabel("correlation")
            plt.show()
