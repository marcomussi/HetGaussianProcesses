import numpy as np
from utils import kernel_rbf



class GaussianProcessRegressorRBF: 


    def __init__(self, kernel_L, regularization, input_dim=1):
        
        self.kernel_L = kernel_L
        self.regularization = regularization  
        self.input_dim = input_dim


    def load_data(self, x, y):
        
        assert x.ndim == 2 and x.shape[1] == self.input_dim, \
                "load_data() function: Error in input"
        assert (y.ndim == 2 and y.shape == (x.shape[0], 1)) or (
            y.ndim == 1 and y.shape == (x.shape[0], )), \
                "load_data() function: Error in input"
        
        self.n_samples = x.shape[0]
        self.x_vect = x.reshape(self.n_samples, self.input_dim)
        self.y_vect = y.reshape(self.n_samples, 1)
        self.K_matrix = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) \
            + self.regularization * np.eye(self.n_samples)
        self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(self.n_samples))


    def get_eigvals(self):
        
        return np.linalg.svd(self.K_matrix, compute_uv=False, hermitian=True)


    def compute(self, x):
        
        assert x.ndim == 2 and x.shape[1] == self.input_dim, "compute() function: Error in input"
        
        n = x.shape[0]
        postmean = np.zeros(n)
        postvariance = np.zeros(n)
        for i in range(n):
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            postmean[i] = K_star.T @ self.K_matrix_inv @ self.y_vect
            postvariance[i] = kernel_rbf(x[i, :].reshape(1, self.input_dim), x[i, :].reshape(1, self.input_dim), 
                self.kernel_L) - K_star.T @ self.K_matrix_inv @ K_star
        return postmean, postvariance
