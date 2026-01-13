import numpy as np
from onlinelearning.homoskedastic import GaussianProcessRegressorRBF
from onlinelearning.heteroskedastic import HeteroskedasticGaussianProcessRegressorRBF



class IGPUCB:


    def __init__(self, n_actions, action_dim, actions, kernel_L, sigma_sq_process, B, delta, het=True, incr_update=False):
        
        if not (actions.ndim == 2 and actions.shape == (n_actions, action_dim)):
            raise ValueError(f"Error in action dimension. Expected ({n_actions}, {action_dim}), got {actions.shape}")
        
        self.n_actions = n_actions
        self.action_dim = action_dim
        self.actions = actions
        self.kernel_L = kernel_L
        self.sigma_sq_process = sigma_sq_process
        self.B = B
        self.delta = delta
        self.het = het
        self.incr_update = incr_update

        if het and incr_update:
            raise NotImplementedError("not implemented yet")
        
        self.reset()

        
    def pull(self):
        
        if self.no_samples:
            self.last_action = self.actions[np.random.choice(self.n_actions), :]
        else:
            postmean, postvariance = self.regressor.compute(self.actions)
            beta = self.B + np.sqrt(self.sigma_sq_process) * np.sqrt(2 * (
                self.regressor.get_info_gain() + 1 + np.log(1/self.delta)))
            ucbs = postmean + beta * np.sqrt(postvariance)
            self.last_action = self.actions[np.argmax(ucbs.ravel()), :]
        
        return self.last_action
        
        
    def update(self, reward):
        
        if self.last_action is None:
            raise ValueError("No action has been selected yet. Call pull() before.")
        
        self.regressor.add_sample(self.last_action.reshape(1, self.action_dim), 
                                  np.array(reward).reshape(1, 1))
        
        self.no_samples = False
        

    def reset(self):

        self.no_samples = True
        self.last_action = None
        
        if self.het:
            self.regressor = HeteroskedasticGaussianProcessRegressorRBF(
                self.kernel_L, 1, input_dim=self.action_dim, 
                incr_update=self.incr_update)
        else:
            self.regressor = GaussianProcessRegressorRBF(
                self.kernel_L, 1, input_dim=self.action_dim, 
                incr_update=self.incr_update)
