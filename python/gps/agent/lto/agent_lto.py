from copy import deepcopy
import numpy as np
from gps.agent.agent import Agent
from gps.proto.gps_pb2 import ACTION
from gps.sample.sample import Sample
from gps.agent.lto.lto_world import LTOWorld

class AgentLTO(Agent):
    
    def __init__(self, hyperparams):
        Agent.__init__(self, hyperparams)
        
        self._setup_conditions()
        self._setup_worlds()
    
    def _setup_conditions(self):
        self.conds = self._hyperparams['conditions']
        self.fcns = self._hyperparams['fcns']
        self.history_len = self._hyperparams['history_len']
        
    def _setup_worlds(self):
        self._worlds = [LTOWorld(self.fcns[i]['fcn_obj'], self.fcns[i]['dim'], self.fcns[i]['init_loc'], self.history_len) for i in range(self.conds)]
        self.x0 = []
        
        for i in range(self.conds):
            self._worlds[i].reset_world()
            self._worlds[i].run(batch_size="all")      # Get noiseless initial state
            x0 = self.get_vectorized_state(self._worlds[i].get_state())
            self.x0.append(x0)
    
    def sample(self, policy, condition, verbose=False, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.

        Args:
            policy: Policy to to used in the trial.
            condition (int): Which condition setup to run.
            verbose (boolean): Whether or not to plot the trial (not used here).
            save (boolean): Whether or not to store the trial into the samples.
            noisy (boolean): Whether or not to use noise during sampling.
        """
        self._worlds[condition].reset_world()
        self._worlds[condition].run()
        new_sample = self._init_sample(self._worlds[condition].get_state())
        U = np.zeros([self.T, self.dU])
        if noisy:
            noise = np.random.randn(self.T, self.dU)
        else:
            noise = np.zeros((self.T, self.dU))
        policy.reset()      # To support non-Markovian policies
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            U[t, :] = policy.act(X_t, obs_t, t, noise[t, :])
            if (t+1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    self._worlds[condition].run_next(U[t, :])
                self._set_sample(new_sample, self._worlds[condition].get_state(), t)
        new_sample.set(ACTION, U)
        policy.finalize()
        if save:
            self._samples[condition].append(new_sample)
        return new_sample
    
    def _init_sample(self, init_X):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = Sample(self)
        self._set_sample(sample, init_X, -1)
        return sample

    def _set_sample(self, sample, X, t):
        for sensor in X.keys():
            sample.set(sensor, np.array(X[sensor]), t=t+1)
