import numpy as np
from regressors.homoscedastic import GaussianProcessRegressorRBF
from regressors.heteroscedastic import HeteroscedasticGaussianProcessRegressorRBF



class IGPUCB:
    """
    Implements the Improved Gaussian Process Upper Confidence Bound (IGP-UCB)
    algorithm for multi-armed bandits.
    """
    
    
    def __init__(self, n_actions, action_dim, actions, kernel_L, sigma_sq, 
                 B, delta, het=True, incr_inverse=False):
        """
        Initializes the IGP-UCB agent.

        Args:
            n_actions (int): The total number of available actions.
            action_dim (int): The dimension of each action vector.
            actions (numpy.ndarray): A 2D array of shape (n_actions, action_dim)
                                     listing all discrete actions.
            kernel_L (float): The length-scale parameter for the RBF kernel.
            sigma_sq (float): The base noise variance of observations.
            B (float): A scaling parameter for the UCB bonus term.
            delta (float): The confidence parameter.
            het (bool, optional): If True, uses the heteroscedastic GP
                for sample averaging. If False, uses the standard
                (homoscedastic) GP. Defaults to True.
            incr_inverse (bool, optional): If het=False, control the use of 
            icremental inverse to speed up matrix inversion in standard 
            (homoscedastic) GP. Ignored if het=True. Defaults to False.
        
        Raises:
            ValueError: If action dimensions are incorrect or kernel_L is not positive.
        """
        
        if not (actions.ndim == 2 and actions.shape == (n_actions, action_dim)):
            raise ValueError(f"Error in action dimension. Expected ({n_actions}, {action_dim}), "
                             f"got {actions.shape}")
        
        self.n_actions = n_actions
        self.action_dim = action_dim
        self.actions = actions
        self.kernel_L = kernel_L
        self.sigma_sq = sigma_sq
        self.B = B
        self.delta = delta
        self.het = het
        self.incr_inverse = incr_inverse
        
        self.reset()

        
    def pull(self):
        """
        Selects the next action to play based on the UCB maximization.

        Returns:
            numpy.ndarray: The selected action vector of shape (action_dim,).
        """
        
        if self.no_samples:
            
            self.last_action = self.actions[np.random.choice(self.n_actions), :]
        
        else:
            
            mu, sigmasq = self.regressor.compute(self.actions)
            beta = self.B + np.sqrt(self.sigma_sq) * np.sqrt(2 * (
                self.regressor.get_info_gain() + 1 + np.log(1/self.delta)))
            ucbs = mu + beta * np.sqrt(sigmasq)
            self.last_action = self.actions[np.argmax(ucbs.ravel()), :]
        
        return self.last_action
        
        
    def update(self, reward, sample_weight=1):
        """
        Updates the GP model with a reward for the last action selected by pull().

        Args:
            reward (float): The observed reward.
            sample_weight (float or int, optional): The weight of this sample.
                Only used if `het=True`. Defaults to 1.
        
        Raises:
            ValueError: If `pull()` has not been called before `update()`.
            ValueError: If `sample_weight` is not positive.
        """
        
        if self.last_action is None:
            raise ValueError("No action has been selected yet. Call pull() before update().")
        
        if self.het:
            if sample_weight <= 0:
                raise ValueError("sample_weight must be positive")
            self.regressor.add_sample(self.last_action.reshape(1, self.action_dim), 
                                  np.array(reward).reshape(1, 1), 
                                  sample_weight=sample_weight)
        else:
            if sample_weight != 1:
                raise ValueError("sample_weight must 1 if het=False")
            self.regressor.add_sample(self.last_action.reshape(1, self.action_dim), 
                                      np.array(reward).reshape(1, 1))
        
        self.no_samples = False
        

    def reset(self):
        """
        Resets the agent to its initial state.
        """

        self.no_samples = True
        self.last_action = None
        
        if self.het:
            self.regressor = HeteroscedasticGaussianProcessRegressorRBF(
                self.kernel_L, self.sigma_sq, 
                input_dim=self.action_dim, one_sample_mod=True)
        else:
            self.regressor = GaussianProcessRegressorRBF(
                self.kernel_L, self.sigma_sq, 
                input_dim=self.action_dim, keep_info_gain_estimate=True, 
                incr_inverse=self.incr_inverse)
