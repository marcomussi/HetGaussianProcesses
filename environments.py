import numpy as np



class NoiseEnv:


    def __init__(self, actions, values, noise_sigmasq, seed=0):
        
        if not isinstance(actions, np.ndarray) or not isinstance(values, np.ndarray):
            raise ValueError(f"NoiseEnv: both 'actions' and values' must be numpy arrays")
        if values.ndim != 1:
             raise ValueError(f"NoiseEnv: 'values' must be a 1D array, got ndim={values.ndim}")
        if actions.ndim != 2:
             raise ValueError(f"NoiseEnv: 'actions' must be a 2D array, got ndim={actions.ndim}")
        if actions.shape[0] != values.shape[0]:
            raise ValueError(f"NoiseEnv: Mismatch in dimensions. actions has {actions.shape[0]} rows but values has {values.shape[0]} elements.")
        
        self.actions = actions
        self.values = values
        self.noise_sigma = np.sqrt(noise_sigmasq)
        self.action_to_id = {tuple(self.actions[i]): i for i in range(self.actions.shape[0])}
        self.reset(seed)

    
    def step(self, action):
        
        return np.random.normal(self.values[self.action_to_id[tuple(action)]], self.noise_sigma)
    

    def reset(self, seed):

        np.random.seed(seed)



class ContinuousNoiseEnv:


    def __init__(self, noise_sigmasq, seed=0, distr_mu = 0.5, distr_sigma=0.08):
        self.distr_mu = distr_mu
        self.distr_sigma = distr_sigma
        self.noise_sigma = np.sqrt(noise_sigmasq)
        self.reset(seed)


    def step(self, action):
        
        return np.random.normal(np.exp(-((action - self.distr_mu)**2) / (2 * self.distr_sigma**2)), self.noise_sigma)
    

    def get_expected(self, actions):
        
        return np.exp(-((actions - self.distr_mu)**2) / (2 * self.distr_sigma**2))


    def reset(self, seed):

        np.random.seed(seed)