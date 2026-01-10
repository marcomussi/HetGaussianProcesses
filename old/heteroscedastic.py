import numpy as np
from utils import kernel_rbf



class HeteroscedasticGaussianProcessRegressorRBF: 
    """
    Implements a Heteroscedastic Gaussian Process Regressor with an RBF kernel.
    """


    def __init__(self, kernel_L, sigma_sq_process, input_dim=1, one_sample_mod=False):
        """
        Initializes the Heteroscedastic Gaussian Process Regressor with an RBF kernel.

        Args:
            kernel_L (float): The length-scale parameter (L) for the RBF kernel.
            sigma_sq_process (float): The base observation noise variance.
            input_dim (int, optional): The dimensionality of the input space.
                                       Defaults to 1.
            one_sample_mod (bool, optional): Selects the operating mode.
                                       Defaults to False.
        """
        self.kernel_L = kernel_L
        self.sigma_sq_process = sigma_sq_process  
        self.input_dim = input_dim
        self.one_sample_mod = one_sample_mod
        self.reset()


    def load_data(self, x, y, sigmasqs):
        """
        Loads a batch of data with specified noise variances for each point.

        This method is *only* available if `one_sample_mod=False`.

        Args:
            x (numpy.ndarray): 2D array of input data, shape (n, input_dim).
            y (numpy.ndarray): 1D or 2D array of output data, shape (n,) or (n, 1).
            sigmasqs (numpy.ndarray): 1D array of noise variances for each data
                                      point, shape (n,).

        Raises:
            ValueError: If called when `one_sample_mod=True`.
        """
        if self.one_sample_mod:
            raise ValueError("load_data() cannot be used with one_sample_mod=True")
        
        n = x.shape[0]
        
        self.x_vect = np.array([x]).reshape(n, self.input_dim)
        self.y_vect = np.array([y]).reshape(n, 1)
        self.sigmasqs = np.array([sigmasqs]).reshape(n,)
        
        self.K_matrix = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) + np.diag(self.sigmasqs)
        self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(n))

    
    def add_sample(self, x, y, sample_weight=1):
        """
        Adds a single sample.

        This method is *only* available if `one_sample_mod=True`.

        Args:
            x (float, int, or numpy.ndarray): The new input point.
                Shape (1, input_dim).
            y (float, int, or numpy.ndarray): The new output value.
                Shape (1,) or (1, 1).
            sample_weight (float or int, optional): The weight of this sample.
                Used to update the average $y$ and the sample count $N_i$.
                Defaults to 1.

        Raises:
            ValueError: If called when `one_sample_mod=False` or if
                        input shapes are incorrect.
        """
        if not self.one_sample_mod:
            raise ValueError("add_sample() cannot be used with one_sample_mod=False")
        if isinstance(x, np.ndarray):
            assert x.ndim == 2 and x.shape == (1, self.input_dim), "add_sample() function: Error in input"
        if isinstance(y, np.ndarray):
            assert (y.ndim == 1 and y.shape == (1, )) or (y.ndim == 2 and y.shape == (1, 1)), \
                "add_sample() function: Error in input"
        
        x = np.array([x]).reshape(1, self.input_dim)
        y = np.array([y]).reshape(1, 1)
        
        if self.x_vect is None:

            self.x_vect = x
            self.y_vect = y

            self.num_samples = [sample_weight]
            
            self.K_matrix_noreg = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L)
            self.K_matrix = self.K_matrix_noreg + (self.sigma_sq_process / sample_weight)
            self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(1))
        
        else:

            matches = np.where(np.all(self.x_vect == x, axis=1))[0]

            if len(matches) > 0:

                pos_first_found = matches[0]
                self.y_vect[pos_first_found] = (
                    self.y_vect[pos_first_found] * self.num_samples[pos_first_found] + y * sample_weight
                    ) / (self.num_samples[pos_first_found] + sample_weight)
                
                self.num_samples[pos_first_found] = self.num_samples[pos_first_found] + sample_weight
                
                self.K_matrix = self.K_matrix_noreg + np.diag(self.sigma_sq_process / np.array(self.num_samples))
                self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(self.K_matrix.shape[0]))
            
            else: 

                self.x_vect = np.vstack((self.x_vect, x))
                self.y_vect = np.vstack((self.y_vect, y))
                self.num_samples.append(sample_weight)
                
                self.K_matrix_noreg = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L)
                self.K_matrix = self.K_matrix_noreg + np.diag(self.sigma_sq_process / np.array(self.num_samples))
                self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(self.K_matrix.shape[0]))


    def compute(self, x):
        """
        Computes the posterior mean and variance at given test points.

        Args:
            x (numpy.ndarray): A 2D array of test points to predict at,
                               with shape (n_test_points, input_dim).

        Returns:
            tuple: A tuple `(mu, sigmasq)`:
                - **mu (numpy.ndarray)**: 1D array of posterior mean values
                  $f(x)$. Shape (n_test_points,).
                - **sigmasq (numpy.ndarray)**: 1D array of posterior variance
                  values $Var[f(x)]$. Shape (n_test_points,).

        Note:
            The returned `sigmasq` is the variance of the *noiseless function*
            $f(x)$. To get the predictive variance for a new *observation* $y^*$,
            you must add a noise term (e.g., `self.sigma_sq_process` if
            assuming a single new sample).
            
        Raises:
            ValueError: If input $x$ is not a 2D array or has the wrong
                        number of features (input_dim).
        """
        assert x.ndim == 2, "compute() function: Error in input dimension"
        assert x.shape[1] == self.input_dim, "compute() function: Error in input dimension"
        
        n = x.shape[0]
        mu = np.zeros(n)
        sigmasq = np.zeros(n)
        
        for i in range(n):
            
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            mu[i] = K_star.T @ self.K_matrix_inv @ self.y_vect
            sigmasq[i] = kernel_rbf(x[i, :].reshape(1, self.input_dim), x[i, :].reshape(1, self.input_dim), 
                                    self.kernel_L) - K_star.T @ self.K_matrix_inv @ K_star

        return mu, sigmasq


    def get_info_gain(self):
        """
        Computes the information gain $I(y; f)$ for the heteroscedastic model.

        Returns:
            float: The computed information gain.
        """
        if self.one_sample_mod:
            temp = self.sigma_sq_process / np.array(self.num_samples)
        else:
            temp = self.sigmasqs
        
        D = np.diag(temp ** -0.5)
        _, value = np.linalg.slogdet(D @ (self.K_matrix - np.diag(temp)) @ D + np.eye(D.shape[0]))
        return 0.5 * value
    

    def reset(self):
        """
        Reset the regressor.
        """
        self.x_vect = None
        self.y_vect = None
        
        if self.one_sample_mod:
            self.num_samples = []
