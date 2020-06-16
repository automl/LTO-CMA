import numpy as np

from gps.algorithm.policy.policy import Policy
from gps.proto.gps_pb2 import CUR_LOC

class LBFGSPolicy(Policy):
    def __init__(self, agent, learning_rate, mem_len, cond, noise_var = None):
        Policy.__init__(self)
        
        self.agent = agent
        self.learning_rate = learning_rate
        self.mem_len = mem_len
        if noise_var is not None:
            self.sqrt_noise_var = np.sqrt(noise_var)
        self.cond = cond    # cond, not m
        self.reset()
    
    def act(self, x, obs, t, noise=None):
        assert(t == self.prev_t + 1)
        self.prev_t = t
        
        cur_loc = self.agent.unpack_data_x(x, data_types=[CUR_LOC])
        grad = self.agent.fcns[self.cond]['fcn_obj'].grad(cur_loc[:,None])[:,0]
        
        if self.s_k is None:
            self.s_k = np.empty((grad.shape[0],self.mem_len-1))
            self.s_k.fill(np.nan)
            self.y_k = np.empty((grad.shape[0],self.mem_len-1))
            self.y_k.fill(np.nan)
            self.r_k = np.empty((self.mem_len-1,))
            self.r_k.fill(np.nan)
        else:
            self.s_k[:,1:] = self.s_k[:,:-1]
            self.s_k[:,0] = cur_loc - self.prev_loc
            self.y_k[:,1:] = self.y_k[:,:-1]
            self.y_k[:,0] = grad - self.prev_grad
            self.r_k[1:] = self.r_k[:-1]
            self.r_k[0] = 1. / (np.dot(self.y_k[:,0], self.s_k[:,0]) + 1e-8)
        
        a_k = np.empty((min(t,self.mem_len-1),))
        a_k.fill(np.nan)
        
        q = grad
        for i in range(min(t,self.mem_len-1)):
            a_k[i] = self.r_k[i] * np.dot(self.s_k[:,i],q)
            q = q - a_k[i] * self.y_k[:,i]
        
        if t == 0:
            z = q
        else:
            z = np.dot(self.s_k[:,0], self.y_k[:,0]) / np.dot(self.y_k[:,0], self.y_k[:,0]) * q
        
        for i in range(min(t,self.mem_len-1)-1,-1,-1):
            b = self.r_k[i] * np.dot(self.y_k[:,i],z)
            z = z + self.s_k[:,i]*(a_k[i] - b)
        
        cur_dir = -z
        
        assert(not np.any(np.isnan(cur_dir)))
        
        u = self.learning_rate*cur_dir
        
        self.prev_loc = cur_loc
        self.prev_grad = grad
        if noise is not None:
            u += self.sqrt_noise_var * noise
        return u
    
    def reset(self):
        self.s_k = None
        self.y_k = None
        self.r_k = None
        self.prev_loc = None
        self.prev_grad = None
        self.prev_t = -1
