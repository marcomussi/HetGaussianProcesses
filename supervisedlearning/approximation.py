import numpy as np
from utils import kernel_rbf, aggregate_dataset, associate_nearest



class ApproximatedHeteroskedasticGaussianProcessRegressorRBF: 


    def __init__(self, kernel_L, regularization, epsilon=None, input_dim=1, range_min=0, range_max=1):
        
        if input_dim != 1:
            raise NotImplementedError("Not implemented yet for input_dim != 1")
        if epsilon is not None:
            if (epsilon <= 0 or epsilon > 1):
                raise ValueError("epsilon must be in (0,1] or None")

        self.kernel_L = kernel_L
        self.regularization = regularization  
        self.input_dim = input_dim
        self.epsilon = epsilon

        if epsilon != None:
            self.cover_dist = 2 * (-np.log(1 - self.epsilon)/self.kernel_L)
            self.cover_x = np.linspace(range_min, range_max, int((range_max - range_min) / self.cover_dist) + 2)


    def load_data(self, x, y):
        
        assert x.ndim == 2 and x.shape[1] == self.input_dim, "load_data() function: Error in input"
        assert (y.ndim == 2 and y.shape == (x.shape[0], 1)) or (
            y.ndim == 1 and y.shape == (x.shape[0], )), "load_data() function: Error in input"
        
        if self.epsilon != None:
            x_ravel = x.ravel()
            x_covered = associate_nearest(x_ravel, self.cover_x)
            x_covered = x_covered.reshape(-1, 1)
        else:
            x_covered = x

        self.x_vect, self.y_vect, samples_vect = aggregate_dataset(x_covered, y.ravel())
        self.samples_vect = samples_vect.reshape(self.x_vect.shape[0], )
        D = np.diag(self.samples_vect**(1/2))
        self.K_matrix = D @ kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) @ D \
            + self.regularization * np.eye(self.x_vect.shape[0])
        self.K_matrix_inv = D @ np.linalg.solve(self.K_matrix, np.eye(self.x_vect.shape[0])) @ D


    def compute(self, x):
        
        assert x.ndim == 2 and x.shape[1] == self.input_dim, "compute() function: Error in input dimension"
        
        n = x.shape[0]
        postmean = np.zeros(n)
        postvariance = np.zeros(n)
        for i in range(n):
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            postmean[i] = K_star.T @ self.K_matrix_inv @ self.y_vect
            postvariance[i] = kernel_rbf(x[i, :].reshape(1, self.input_dim), x[i, :].reshape(
                1, self.input_dim), self.kernel_L) - K_star.T @ self.K_matrix_inv @ K_star
        return postmean, postvariance
