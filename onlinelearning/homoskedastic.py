import numpy as np
from utils import incr_inv, kernel_rbf



class GaussianProcessRegressorRBF: 


    def __init__(self, kernel_L, regularization, input_dim=1, incr_update=True):

        self.kernel_L = kernel_L
        self.regularization = regularization  
        self.input_dim = input_dim
        self.incr_update = incr_update
        self.reset()


    def add_sample(self, x, y):

        if isinstance(x, np.ndarray):
            assert x.ndim == 2 and x.shape == (1, self.input_dim), "add_sample(): Error in input"
        if isinstance(y, np.ndarray):
            assert (y.ndim == 1 and y.shape == (1, )) or (y.ndim == 2 and y.shape == (1, 1)), "add_sample(): Error in input"
        
        x = np.array([x]).reshape(1, self.input_dim)
        y = np.array([y]).reshape(1, 1)
        self.n_samples += 1
        if self.x_vect is None:
            self.x_vect = x
            self.y_vect = y
            self.K_matrix = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) + self.regularization
            self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(1))
            self.info_gain = (0.5 * np.log(1 + 1 / self.regularization))
        else:
            self.x_vect = np.vstack((self.x_vect, x))
            self.y_vect = np.vstack((self.y_vect, y))
            K_star = kernel_rbf(self.x_vect[:-1, :].reshape(-1, self.input_dim), 
                                self.x_vect[-1, :].reshape(1, self.input_dim), self.kernel_L)
            elem = kernel_rbf(self.x_vect[-1, :].reshape(1, self.input_dim), 
                              self.x_vect[-1, :].reshape(1, self.input_dim), self.kernel_L)
            sigma_i = elem - K_star.T @ self.K_matrix_inv @ K_star
            self.info_gain += (0.5 * np.log(1 + sigma_i / self.regularization))
            elem = np.array(elem + self.regularization).reshape(1, 1)
            self.K_matrix = np.vstack((np.hstack((self.K_matrix, K_star)), np.hstack((K_star.T, elem))))
            if self.incr_update:
                self.K_matrix_inv = incr_inv(self.K_matrix_inv, K_star, K_star.T, elem)
            else: 
                self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(self.K_matrix.shape[0]))
            

    def compute(self, x):

        assert x.ndim == 2 and x.shape[1] == self.input_dim, "compute(): Error in input"
        
        n = x.shape[0]
        postmean = np.zeros(n)
        postvariance = np.zeros(n)
        for i in range(n):
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            postmean[i] = K_star.T @ self.K_matrix_inv @ self.y_vect
            postvariance[i] = kernel_rbf(x[i, :].reshape(1, self.input_dim), x[i, :].reshape(1, self.input_dim), self.kernel_L) - K_star.T @ self.K_matrix_inv @ K_star
        return postmean, postvariance
    

    def get_info_gain(self):

        return self.info_gain[0, 0] if isinstance(self.info_gain, np.ndarray) else self.info_gain
        

    def reset(self):

        self.n_samples = 0
        self.info_gain = None
        self.x_vect = None
        self.y_vect = None
