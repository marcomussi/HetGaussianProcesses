import numpy as np
from utils import incr_inv, kernel_rbf



class GaussianProcessRegressorRBF: 
    """
    Implements a Gaussian Process Regressor with an RBF kernel.
    """


    def __init__(self, kernel_L, sigma_sq_process, input_dim=1, keep_info_gain_estimate=False, incr_inverse=True):
        """
        Initializes the Gaussian Process Regressor with an RBF kernel.

        Args:
            kernel_L (float): The length-scale parameter (L) for the RBF kernel.
                              Smaller 'L' implies a smoother function.
            sigma_sq_process (float): The variance of the observation noise.
                                      This accounts for uncertainty in the 
                                      $y$ values.
            input_dim (int, optional): The dimensionality of the input space.
                                       Defaults to 1.
            keep_info_gain_estimate (bool, optional): If True, the model will
                                       incrementally compute the information
                                       gain with each added sample.
                                       Defaults to False.
        """

        self.kernel_L = kernel_L
        self.sigma_sq_process = sigma_sq_process  
        self.input_dim = input_dim
        self.keep_info_gain_estimate = keep_info_gain_estimate
        self.incr_inverse = incr_inverse

        self.reset()


    def load_data(self, x, y):
        """
        Loads a batch of data into the model, overwriting any existing data.

        Args:
            x (numpy.ndarray): A 2D array of input data, shape (n_samples, input_dim).
            y (numpy.ndarray): A 1D or 2D array of output data, shape (n_samples,)
                               or (n_samples, 1).
        """

        self.n_samples = x.shape[0]
        
        self.x_vect = np.array([x]).reshape(self.n_samples, self.input_dim)
        self.y_vect = np.array([y]).reshape(self.n_samples, 1)
        
        self.K_matrix = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) \
                        + self.sigma_sq_process * np.eye(self.n_samples)
        self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(self.n_samples))
        
        if self.keep_info_gain_estimate: 
            _, value = np.linalg.slogdet(self.K_matrix/self.sigma_sq_process)
            self.info_gain = 0.5 * value
    

    def add_sample(self, x, y):
        """
        Adds a single new data point ($x, y$) to the model incrementally.

        Args:
            x (float, int, or numpy.ndarray): The new input point.
                Can be a scalar if input_dim=1, or an array of shape
                (1, input_dim).
            y (float, int, or numpy.ndarray): The new output value.
                Can be a scalar or an array of shape (1,) or (1, 1).
                
        Raises:
            ValueError: If provided $x$ or $y$ are numpy arrays with
                        incompatible shapes.
        """

        if isinstance(x, np.ndarray):
            assert x.ndim == 2 and x.shape == (1, self.input_dim), "add_sample() function: Error in input"
        if isinstance(y, np.ndarray):
            assert (y.ndim == 1 and y.shape == (1, )) or (y.ndim == 2 and y.shape == (1, 1)), \
                "add_sample() function: Error in input"
        
        x = np.array([x]).reshape(1, self.input_dim)
        y = np.array([y]).reshape(1, 1)
        
        self.n_samples += 1
        
        if self.x_vect is None:
            
            self.x_vect = x
            self.y_vect = y

            self.K_matrix = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) + self.sigma_sq_process
            self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(1))
            
            if self.keep_info_gain_estimate:
                self.info_gain = (0.5 * np.log(1 + 1 / self.sigma_sq_process))
        
        else:

            self.x_vect = np.vstack((self.x_vect, x))
            self.y_vect = np.vstack((self.y_vect, y))
            
            K_star = kernel_rbf(self.x_vect[:-1, :].reshape(-1, self.input_dim), 
                                self.x_vect[-1, :].reshape(1, self.input_dim), self.kernel_L)
            
            elem = kernel_rbf(self.x_vect[-1, :].reshape(1, self.input_dim), 
                              self.x_vect[-1, :].reshape(1, self.input_dim), self.kernel_L)
            
            if self.keep_info_gain_estimate:
                sigma_i = elem - K_star.T @ self.K_matrix_inv @ K_star
                self.info_gain += (0.5 * np.log(1 + sigma_i / self.sigma_sq_process))
            
            elem = np.array(elem + self.sigma_sq_process).reshape(1, 1)
            self.K_matrix = np.vstack((np.hstack((self.K_matrix, K_star)), np.hstack((K_star.T, elem))))
            if self.incr_inverse:
                self.K_matrix_inv = incr_inv(self.K_matrix_inv, K_star, K_star.T, elem)
            else: 
                self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(self.K_matrix.shape[0]))
            

    def compute(self, x):
        """
        Computes the posterior mean and variance at given test points.

        Args:
            x (numpy.ndarray): A 2D array of test points to predict at,
                               with shape (n_test_points, input_dim).

        Returns:
            tuple: A tuple `(mu, sigma)`:
                - **mu (numpy.ndarray)**: 1D array of posterior mean values
                  $f(x)$. Shape (n_test_points,).
                - **sigma (numpy.ndarray)**: 1D array of posterior variance
                  values $Var[f(x)]$. Shape (n_test_points,).

        Note:
            The returned `sigma` is the variance of the *noiseless function* $f(x)$.
            To get the predictive variance for a new *observation* $y^*$,
            you must add the noise variance: `sigma + self.sigma_sq_process`.
            
        Raises:
            ValueError: If input $x$ is not a 2D array or has the wrong
                        number of features (input_dim).
        """

        assert x.ndim == 2 and x.shape[1] == self.input_dim, "compute() function: Error in input"
        
        n = x.shape[0]
        
        mu = np.zeros(n)
        sigma = np.zeros(n)
        
        for i in range(n):
            
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            mu[i] = K_star.T @ self.K_matrix_inv @ self.y_vect
            sigma[i] = kernel_rbf(x[i, :].reshape(1, self.input_dim), x[i, :].reshape(1, self.input_dim), 
                                  self.kernel_L) - K_star.T @ self.K_matrix_inv @ K_star
        
        return mu, sigma
    

    def get_info_gain(self):
        """
        Returns the currently computed information gain.

        Returns:
            float: The total information gain $I(y; f)$.

        Raises:
            ValueError: If `keep_info_gain_estimate` was set to `False`
                        during initialization.
        """

        if self.keep_info_gain_estimate:
            return self.info_gain[0, 0] if isinstance(self.info_gain, np.ndarray) else self.info_gain
        else:
            raise ValueError("Info Gain not computed, use flag "
                             "keep_info_gain_estimate=True during initialization")
        

    def reset(self):
        """
        Reset the regressor.
        """

        self.n_samples = 0
        self.info_gain = None
        self.x_vect = None
        self.y_vect = None
