import numpy as np
from utils import kernel_rbf



class HeteroskedasticGaussianProcessRegressorRBF: 


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
            assert (y.ndim == 1 and y.shape == (1, )) or (y.ndim == 2 and y.shape == (1, 1)), \
                "add_sample(): Error in input"
        
        x = np.array([x]).reshape(1, self.input_dim)
        y = np.array([y]).reshape(1, 1)
        
        if self.x_vect is None:

            self.x_vect = x
            self.y_vect = y
            self.num_samples = [1]

            self.K_matrix_noreg = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L)
            self.K_matrix = self.K_matrix_noreg + self.regularization
            self.K_matrix_inv = 1 / self.K_matrix # 1 element
        
        else:
                    
            matches = np.where(np.all(self.x_vect == x, axis=1))[0]

            if len(matches) > 0:

                pos_first_found = matches[0]
                self.y_vect[pos_first_found] = (
                    self.y_vect[pos_first_found] * self.num_samples[pos_first_found] + y
                    ) / (self.num_samples[pos_first_found] + 1)
                
                self.num_samples[pos_first_found] = self.num_samples[pos_first_found] + 1

                D = np.diag(np.array(self.num_samples)**(1/2)) # can be made better

                aux_mx = D @ self.K_matrix @ D + self.regularization * np.eye(self.K_matrix.shape[0])
                self.K_matrix_inv = np.linalg.solve(aux_mx, np.eye(self.K_matrix.shape[0]))
                self.K_matrix_inv = D @ self.K_matrix_inv @ D
            
            else: 

                self.x_vect = np.vstack((self.x_vect, x))
                self.y_vect = np.vstack((self.y_vect, y))
                self.num_samples.append(1)
                D = np.diag(np.array(self.num_samples)**(1/2))
                self.K_matrix = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) # we can just add new row/col instead
                aux_mx = D @ self.K_matrix @ D + self.regularization * np.eye(self.K_matrix.shape[0])
                self.K_matrix_inv = np.linalg.solve(aux_mx, np.eye(self.K_matrix.shape[0]))
                self.K_matrix_inv = D @ self.K_matrix_inv @ D

    
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


    def get_info_gain(self):

        temp = self.regularization / np.array(self.num_samples)
        D = np.diag(temp ** -0.5)
        _, value = np.linalg.slogdet(np.eye(D.shape[0]) + (1/self.regularization) * D @ (self.K_matrix) @ D)
        return 0.5 * value
    

    def reset(self):

        self.x_vect = None
        self.y_vect = None
        self.num_samples = []
