import numpy as np

from gps.algorithm.policy.policy import Policy
from gps.proto.gps_pb2 import CUR_LOC

class GradientDescentPolicy(Policy):
    def __init__(self, agent, learning_rate, cond, noise_var = None):
        Policy.__init__(self)
        
        self.agent = agent
        self.learning_rate = learning_rate
        if noise_var is not None:
            self.sqrt_noise_var = np.sqrt(noise_var)
        self.cond = cond    # cond, not m
    
    def act(self, x, obs, t, noise=None):
        
        cur_loc = self.agent.unpack_data_x(x, data_types=[CUR_LOC], condition=self.cond)
        grad = self.agent.fcns[self.cond]['fcn_obj'].grad(cur_loc[:,None])[:,0]
        
        u = -self.learning_rate*grad
        if noise is not None:
            u += self.sqrt_noise_var * noise
        return u
