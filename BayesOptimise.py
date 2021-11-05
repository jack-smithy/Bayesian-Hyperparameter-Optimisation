import numpy as np
from numpy import random

import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import Matern

from scipy.stats import norm
from scipy.optimize import minimize


class BayesOpt():
    def __init__(self, gaussian_process, loss, bounds=(0,10)):
        """ BayesOpt

        Bayesian Optimisation Class

        Arguments:
        ----------
            gaussian_process: GaussianProcessRegressor object.
                Gaussian process trained on previously evaluated hyperparameters.
            loss: Numpy array.
                Numpy array that contains the values off the loss function for the previously
                evaluated hyperparameters.
            maximise: Boolean.
                Boolean flag that indicates whether the loss function is to be maximised or minimised.
        """

        self.gaussian_process = gaussian_process
        self.loss = loss

        self.bounds = bounds

        self.maximise = False,

        self.gp_params=None,
        self.x0=None 
        

    def expected_improvement(self, x, n_params=1):
        x_to_predict = x.reshape(-1, n_params)

        mu, sigma = self.gaussian_process.predict(x_to_predict, return_std=True)

        if self.maximise == True:
            opt_loss = np.max(self.loss)
        else:
            opt_loss = np.min(self.loss)

        scale = (-1)**(not self.maximise)
    
        return -1*self.expected_improvement

    def sample_next_hyperparam(self, aqf, bounds=(0,10), n_restarts=25):
        best_x = 0
        best_aq = 1
        n_params = bounds.shape[0]

        for start_point in np.random.uniform(bounds[:,0], bounds[:,1], size=(n_restarts, n_params)):
            res = minimize(func=aqf,
                           x0=start_point.reshape(1, -1),
                           bounds=bounds,
                           method='L-BFGS-B',
                           args=(self.gaussian_process, self.loss, self.maximise, n_params))

        if res.func < best_aq:
            best_aq = res.func
            best_x = res.x

        return best_x

    def bayesian_optimisation(self, n_iters, loss_func, bounds, n_pre_samples, alpha=1e-5, epsilon=1e-7):
        x_arr = np.array([])
        y_arr = np.array([])

        n_params = bounds.shape[0]

        if self.x0 is None:
            for params in np.random.uniform(bounds[:,0], bounds[:,1], (n_pre_samples, n_params)):
                x_arr = np.append(x_arr, params)
                y_arr = np.append(y_arr, loss_func(params))
        else:
            for params in self.x0:
                x_arr = np.append(x_arr, params)
                y_arr = np.append(y_arr, loss_func(params))

        if self.gp_params is not None:
            model = gp.GaussianProcessRegressor(**self.gp_params)
        else:
            kernel = Matern
            model = gp.GaussianProcessRegressor(kernel=kernel,
                                                alpha=alpha,
                                                n_restarts_optimizer=10,
                                                normalize_y=True)

        for n in range(n_iters):
            model.fit(x_arr, y_arr)

            next_sample = self.sample_next_hyperparameter(self, self.expected_improvement, model, y_arr, bounds = bounds,
                                                          n_restarts=100)

            if np.any(np.abs(next_sample - x_arr) <= epsilon):
                next_sample = np.random.uniform(bounds[:,0], bounds[:,1], bounds.shape[0])

            score = loss_func(next_sample)

            x_arr = np.append(x_arr, next_sample)
            y_arr = np.append(y_arr, score)

            return x_arr, y_arr


            

        

