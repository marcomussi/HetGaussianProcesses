import numpy as np



class NoiseEnv:
    """
    A simple environment simulator for a multi-armed bandit problems.
    """


    def __init__(self, actions, values, noise_sigmasq, bernoulli=False):
        """
        Initializes the noise environment.

        Args:
            actions (numpy.ndarray): A 2D array of actions, shape (n_actions, action_dim).
                                     e.g., [[0,0], [0,1], [1,0], [1,1]]
            values (numpy.ndarray): A 1D array of mean rewards, shape (n_actions,).
                                    Must be in the same order as `actions`.
            noise_sigmasq (float): The variance of the Gaussian noise.
                                   This parameter is ignored if `bernoulli=True`.
            bernoulli (bool, optional): If True, switches to Bernoulli reward mode.
                                        `values` should be probabilities [0, 1].
                                        Defaults to False (Gaussian mode).

        Raises:
            ValueError: If `actions` or `values` are not numpy arrays or
                        if their dimensions are mismatched.
        """
        
        if not isinstance(actions, np.ndarray):
            raise ValueError(f"NoiseEnv: 'actions' must be a numpy array, got {type(actions)}")
        if not isinstance(values, np.ndarray):
            raise ValueError(f"NoiseEnv: 'values' must be a numpy array, got {type(values)}")
        if values.ndim != 1:
             raise ValueError(f"NoiseEnv: 'values' must be a 1D array, got ndim={values.ndim}")
        if actions.ndim != 2:
             raise ValueError(f"NoiseEnv: 'actions' must be a 2D array, got ndim={actions.ndim}")
        if actions.shape[0] != values.shape[0]:
            raise ValueError(f"NoiseEnv: Mismatch in dimensions. actions has {actions.shape[0]} "
                             f"rows but values has {values.shape[0]} elements.")
        
        self.actions = actions
        self.values = values
        self.bernoulli = bernoulli
        
        if self.bernoulli:
            self.noise_sigmasq = 1/4
            if np.any(values < 0) or np.any(values > 1):
                print("Warning: Bernoulli mode selected, but 'values' "
                      "contain entries outside the [0, 1] probability range.")
        else: 
            self.noise_sigmasq = noise_sigmasq
        
        self.action_to_id = {tuple(self.actions[i]): i for i in range(self.actions.shape[0])}
        
        np.random.seed(0)

    
    def step(self, action):
        """
        Takes an action and returns a noisy reward.

        Args:
            action (list, tuple, or numpy.ndarray): The action taken by the agent.
                Must be one of the actions from the `self.actions` array.

        Returns:
            float or int: A single reward sample.
                - (int) 0 or 1 if `bernoulli=True`.
                - (float) A noisy value if `bernoulli=False`.
        
        Raises:
            KeyError: If the `action` is not in the recognized list of actions.
        """
        action_tuple = tuple(action)
        action_id = self.action_to_id[action_tuple]
        
        if self.bernoulli:
            return np.random.binomial(1, self.values[action_id])
        else:
            mean_reward = self.values[action_id]
            std_dev = np.sqrt(self.noise_sigmasq)
            return np.random.normal(mean_reward, std_dev)

    
    def reset(self, seed):
        """
        Resets the environment's random number generator seed.

        Args:
            seed (int): The new seed for `numpy.random`.
        """
        np.random.seed(seed)