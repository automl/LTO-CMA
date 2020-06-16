""" This file defines an environment for the Box2D PointMass simulator. """
import numpy as np
from collections import deque

from gps.proto.gps_pb2 import CUR_LOC, PAST_OBJ_VAL_DELTAS, PAST_GRADS, CUR_GRAD, PAST_LOC_DELTAS

class LTOWorld(object):
    def __init__(self, fcn, dim, init_loc, history_len):
        self.fcn = fcn
        self.dim = dim
        self.init_loc = init_loc
        self.history_len = history_len
        self.past_locs = deque(maxlen=history_len)
        self.past_obj_vals = deque(maxlen=history_len)
        self.past_grads = deque(maxlen=history_len)

    def run(self, batch_size = None):
        """Initiates the first time step"""
        self.fcn.new_sample(batch_size=batch_size)
        self.cur_loc = self.init_loc
        self.cur_obj_val = self.fcn.evaluate(self.cur_loc)
        self.cur_grad = self.fcn.grad(self.cur_loc)
        
    # action is of shape (dU,)
    def run_next(self, action, batch_size = None):
        """Moves forward in time one step"""
        self.fcn.new_sample(batch_size=batch_size)
        self.past_locs.append(self.cur_loc)
        self.past_obj_vals.append(self.cur_obj_val)
        self.past_grads.append(self.cur_grad)
        self.cur_loc = self.cur_loc + action[:,None]
        self.cur_obj_val = self.fcn.evaluate(self.cur_loc)
        self.cur_grad = self.fcn.grad(self.cur_loc)

    def reset_world(self):
        self.past_locs.clear()
        self.past_obj_vals.clear()
        self.past_grads.clear()

    def get_state(self):
        past_obj_val_deltas = []
        for i in range(1,len(self.past_obj_vals)):
            past_obj_val_deltas.append((self.past_obj_vals[i] - self.past_obj_vals[i-1]) / float(self.past_obj_vals[i-1]))
        if len(self.past_obj_vals) > 0:
            past_obj_val_deltas.append((self.cur_obj_val - self.past_obj_vals[-1]) / float(self.past_obj_vals[-1]))
        past_obj_val_deltas = np.array(past_obj_val_deltas)
        
        past_loc_deltas = []
        for i in range(1,len(self.past_locs)):
            past_loc_deltas.append(self.past_locs[i] - self.past_locs[i-1])
        if len(self.past_locs) > 0:
            past_loc_deltas.append(self.cur_loc - self.past_locs[-1])
            past_loc_deltas = np.vstack(past_loc_deltas)[:,0]
        else:
            past_loc_deltas = np.zeros((0,))
        
        if len(self.past_grads) > 0:
            past_grads = np.vstack(self.past_grads)[:,0]
        else:
            past_grads = np.zeros((0,))
        
        past_obj_val_deltas = np.hstack((np.zeros((self.history_len-past_obj_val_deltas.shape[0],)),past_obj_val_deltas))
        past_grads = np.hstack((np.zeros((self.history_len*self.dim-past_grads.shape[0],)),past_grads))
        past_loc_deltas = np.hstack((np.zeros((self.history_len*self.dim-past_loc_deltas.shape[0],)),past_loc_deltas))
        cur_loc = self.cur_loc[:,0]
        cur_grad = self.cur_grad[:,0]
        
        state = {CUR_LOC: cur_loc,
                 PAST_OBJ_VAL_DELTAS: past_obj_val_deltas,
                 PAST_GRADS: past_grads, 
                 CUR_GRAD: cur_grad, 
                 PAST_LOC_DELTAS: past_loc_deltas
                }
        
        return state
