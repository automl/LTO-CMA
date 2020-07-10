import numpy as np

from gps.algorithm.policy.policy import Policy
from gps.proto.gps_pb2 import CUR_LOC

class MomentumPolicy(Policy):
    def __init__(self, agent, learning_rate, momentum, cond, noise_var = None):
        Policy.__init__(self)
        
        self.agent = agent
        self.learning_rate = learning_rate
        self.momentum = momentum
        if noise_var is not None:
            self.sqrt_noise_var = np.sqrt(noise_var)
        self.cond = cond    # cond, not m
        self.reset()
    
    def act(self, x, obs, t, noise=None):
        assert(t == self.prev_t + 1)
        self.prev_t = t
        
        cur_loc = self.agent.unpack_data_x(x, data_types=[CUR_LOC])
        grad = self.agent.fcns[self.cond]['fcn_obj'].grad(cur_loc[:,None])[:,0]
        if self.prev_update is None:
            self.prev_update = np.zeros((grad.shape[0],))
        u = self.momentum*self.prev_update - self.learning_rate*grad
        self.prev_update = u
        if noise is not None:
            u += self.sqrt_noise_var * noise
        return u
    
    def reset(self):
        self.prev_update = None
        self.prev_t = -1
