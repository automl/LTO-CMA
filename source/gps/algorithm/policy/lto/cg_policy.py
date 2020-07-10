import numpy as np

from gps.algorithm.policy.policy import Policy
from gps.proto.gps_pb2 import CUR_LOC

class ConjugateGradientPolicy(Policy):
    def __init__(self, agent, learning_rate, cond, noise_var = None):
        Policy.__init__(self)
        
        self.agent = agent
        self.learning_rate = learning_rate
        if noise_var is not None:
            self.sqrt_noise_var = np.sqrt(noise_var)
        self.cond = cond    # cond, not m
        self.reset()
    
    def act(self, x, obs, t, noise=None):
        assert(t == self.prev_t + 1)
        self.prev_t = t
        
        cur_loc = self.agent.unpack_data_x(x, data_types=[CUR_LOC])
        grad = self.agent.fcns[self.cond]['fcn_obj'].grad(cur_loc[:,None])[:,0]
        if self.prev_dir is None:
            cur_dir = -grad
        else:
            beta = np.dot(grad, grad) / float(np.dot(self.prev_grad, self.prev_grad))
            cur_dir = -grad + beta*self.prev_dir
        
        u = self.learning_rate*cur_dir
        
        self.prev_dir = cur_dir
        self.prev_grad = grad
        u = self.add_noise(u, noise, t)
        if noise is not None:
            u += self.sqrt_noise_var * noise
        return u
    
    def reset(self):
        self.prev_dir = None
        self.prev_grad = None
        self.prev_t = -1
