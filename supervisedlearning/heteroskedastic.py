import numpy as np
from utils import kernel_rbf, aggregate_dataset



class HeteroskedasticGaussianProcessRegressorRBF: 


    def __init__(self, kernel_L, regularization, input_dim=1, use_stable_model=True):
        
        self.kernel_L = kernel_L
        self.regularization = regularization  
        self.input_dim = input_dim
        self.use_stable_model = use_stable_model


    def load_data(self, x, y):
        
        assert x.ndim == 2 and x.shape[1] == self.input_dim, "load_data(): Error in input"
        assert (y.ndim == 2 and y.shape == (x.shape[0], 1)) or (
            y.ndim == 1 and y.shape == (x.shape[0], )), "load_data(): Error in input"
        
        self.x_vect, self.y_vect, samples_vect = aggregate_dataset(x, y.ravel())
        n = self.x_vect.shape[0]
        self.samples_vect = samples_vect.reshape(n, )
        if self.use_stable_model:
            D = np.diag(self.samples_vect**(1/2))
            self.K_matrix = D @ kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) @ D \
                + self.regularization * np.eye(n)
            self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(n))
            self.K_matrix_inv = D @ self.K_matrix_inv @ D
        else:
            self.K_matrix = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) \
                + np.diag(self.regularization / self.samples_vect)
            self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(n))


    def get_eigvals(self):
        
        return np.linalg.svd(self.K_matrix, compute_uv=False, hermitian=True)


    def compute(self, x):
        
        assert x.ndim == 2 and x.shape[1] == self.input_dim, "compute(): Error in input"
        
        n = x.shape[0]
        postmean = np.zeros(n)
        postvariance = np.zeros(n)
        for i in range(n):
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            postmean[i] = K_star.T @ self.K_matrix_inv @ self.y_vect
            postvariance[i] = kernel_rbf(x[i, :].reshape(1, self.input_dim), x[i, :].reshape(
                1, self.input_dim), self.kernel_L) - K_star.T @ self.K_matrix_inv @ K_star
        return postmean, postvariance
