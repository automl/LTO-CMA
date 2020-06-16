import numpy as np
#import HPOlib.HPOlib.benchmarks.svm_on_grid as SVM
#import HPOlib.HPOlib.benchmarks.lda_on_grid as LDA
#import HPOlib.HPOlib.benchmarks.logreg_on_grid as LogReg
from hpolib.benchmarks.ml import svm_benchmark, logistic_regression, fully_connected_network
from collections import deque
from cma.evolution_strategy import CMAEvolutionStrategy, CMAOptions
from gps.agent.lto.mnist_nn import MNIST_NN
from gps.proto.gps_pb2 import CUR_LOC, PAST_OBJ_VAL_DELTAS, CUR_PS, CUR_SIGMA, PAST_LOC_DELTAS, PAST_SIGMA
import threading
import concurrent.futures

def _norm(x): return np.sqrt(np.sum(np.square(x)))
class CMAESWorld(object):
    def __init__(self, dim, init_loc, init_sigma, init_popsize, history_len, fcn=None, hpolib=False, benchmark=None):
        if fcn is not None:
            self.fcn = fcn
        else:
            self.fcn = None
        self.hpolib = hpolib
        self.benchmark = benchmark

        #Download benchmark datasets
        if benchmark is not None:
            if benchmark == 'SvmOnMnist':
                self.b = svm_benchmark.SvmOnMnist()
                self.bounds = [[-10, -10], [10, 10]]
            elif benchmark == 'SvmOnVehicle':
                self.b = svm_benchmark.SvmOnVehicle()
                self.bounds = [[-10, -10], [10, 10]]
            elif benchmark == 'SvmOnCovertype':
                self.b = svm_benchmark.SvmOnCovertype()
                self.bounds = [[-10, -10], [10, 10]]
            elif benchmark == 'LogisticRegression10CVOnMnist':
                self.b = logistic_regression.LogisticRegression10CVOnMnist()
                self.bounds = [[-6, 0, 20, 0], [0, 1, 2000, 0.75]]
            elif benchmark == 'FCNetOnMnist':
                self.b = fully_connected_network.FCNetOnMnist()
                self.bounds = [[-6, -8, 32, -3, 0, 0.3, 5, 5, 0, 0], [0, -1, 512, -1, 1, 0.999, 12, 12, 0.99, 0.99]]
            else:
                self.benchmark = None
                self.b = None

        else:
            self.bounds = [None, None]
        self.dim = dim
        self.init_loc = init_loc
        self.init_sigma = init_sigma
        self.init_popsize = init_popsize
        self.fbest = None
        self.history_len = history_len
        self.past_locs = deque(maxlen=history_len)
        self.past_obj_vals = deque(maxlen=history_len)
        self.past_sigma = deque(maxlen=history_len)
        self.solutions = None
        self.func_values = []
        self.f_vals = deque(maxlen=self.init_popsize)
        self.lock = threading.Lock()
        self.chi_N = dim**0.5 * (1 - 1. / (4.*dim) + 1. / (21.*dim**2))

    def fit(self, x):
        if self.fcn is not None:
            loss = self.fcn(x)
        else:
            loss = MNIST_NN(batch_size=64).evaluate(x)
        return loss


    def eval(self):
        self.func_values = []
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.init_popsize) as executor:
            for i in range(self.init_popsize):
             futures.append(executor.submit(self.fit, self.solutions[i]))
            for future in concurrent.futures.as_completed(futures):
                try:
                    self.func_values.append(future.result())
                except Exception as exc:
                    print('Exception was generated:%s'%exc)
        #print(self.func_values)

    def hpolib_benchmarks(self):
        self.func_values = []
        for solution in self.solutions:
            res = self.b.objective_function(configuration=solution)
            self.func_values.append(res['function_value'])

    def hpolib_svm(self):
        self.func_values = []
        for solution in self.solutions:
            params = {"C": solution[0], "alpha": solution[1], "epsilon":solution[2]}
            params["C"] = np.clip(params["C"], 0, 24)
            params["alpha"] = np.clip(params["alpha"], 0, 13)
            params["epsilon"] = np.clip(params["epsilon"], 0, 3)
            #f_val = SVM.save_svm_on_grid(params)
            self.func_values.append(f_val)

    def hpolib_lda(self):
        self.func_values = []
        for solution in self.solutions:
            params = {"Kappa": solution[0], "Tau": solution[1], "S":solution[2]}
            #f_val = LDA.save_lda_on_grid(params)
            self.func_values.append(f_val)

    def hpolib_logreg(self):
        self.func_values = []
        for solution in self.solutions:
            params = {"lrate":solution[0], "l2_reg": solution[1], "batchsize": solution[2], "n_epochs": solution[3]}

            #f_val = LogReg.save_logreg_on_grid(params)
            self.func_values.append(f_val)


    def run(self, batch_size="all", ltorun=False):
        """Initiates the first time step"""
        #self.fcn.new_sample(batch_size=batch_size)
        self.cur_loc = self.init_loc
        self.cur_sigma = self.init_sigma
        self.cur_ps = 0
        if self.fcn is not None:
            self.cur_obj_val = self.fcn.evaluate(self.init_loc)
        else:
            res = self.b.objective_function(configuration=self.init_loc)
            self.cur_obj_val = res['function_value']
        self.es = CMAEvolutionStrategy(self.cur_loc, self.init_sigma, {'popsize': self.init_popsize, 'bounds': self.bounds})
        if self.fcn is not None:
            self.solutions, self.func_values = self.es.ask_and_eval(self.fcn)
        else:
            self.solutions = self.es.ask()
            self.hpolib_benchmarks()
        self.fbest = self.func_values[np.argmin(self.func_values)]
        self.f_difference = np.abs(np.amax(self.func_values) - self.cur_obj_val)/float(self.cur_obj_val)
        self.velocity = np.abs(np.amin(self.func_values) - self.cur_obj_val)/float(self.cur_obj_val)
        self.es.mean_old = self.es.mean
        self.past_locs.append([self.f_difference, self.velocity])
    # action is of shape (dU,)
    def run_next(self, action, batch_size=None):
        #self.fcn.new_sample(batch_size=batch_size)
        self.past_locs.append([self.f_difference, self.velocity])
        if not self.es.stop():
            """Moves forward in time one step"""
            sigma = action
            self.es.tell(self.solutions, self.func_values)
            self.es.sigma = min(max(sigma, 0.05), 10)
            if self.fcn is not None:
                self.solutions, self.func_values = self.es.ask_and_eval(self.fcn)
            else:
                self.solutions = self.es.ask()
                self.hpolib_benchmarks()
        self.f_difference = np.nan_to_num(np.abs(np.amax(self.func_values) - self.cur_obj_val)/float(self.cur_obj_val))
        self.velocity = np.nan_to_num(np.abs(np.amin(self.func_values) - self.cur_obj_val)/float(self.cur_obj_val))
        self.fbest = min(self.es.best.f, np.amin(self.func_values))
        #self.past_locs.append(self.f_difference)
        self.past_obj_vals.append(self.cur_obj_val)
        self.past_sigma.append(self.cur_sigma)
        self.cur_ps = _norm(self.es.adapt_sigma.ps) / self.chi_N - 1
        self.cur_loc = self.es.best.x
        self.cur_sigma = self.es.sigma
        self.cur_obj_val = self.es.best.f

    def reset_world(self):
        self.past_locs.clear()
        self.past_obj_vals.clear()
        self.past_sigma.clear()
        self.cur_loc = self.init_loc
        self.cur_sigma = self.init_sigma
        self.cur_ps = 0
        self.func_values = []


    def get_state(self):
        past_obj_val_deltas = []
        for i in range(1,len(self.past_obj_vals)):
            past_obj_val_deltas.append((self.past_obj_vals[i] - self.past_obj_vals[i-1]+1e-3) / float(self.past_obj_vals[i-1]))
        if len(self.past_obj_vals) > 0:
            past_obj_val_deltas.append((self.cur_obj_val - self.past_obj_vals[-1]+1e-3)/ float(self.past_obj_vals[-1]))
        past_obj_val_deltas = np.array(past_obj_val_deltas).reshape(-1)

        past_loc_deltas = []
        for i in range(len(self.past_locs)):
            past_loc_deltas.append(self.past_locs[i])
        #if len(self.past_locs) > 0:
        #    past_loc_deltas.append(self.past_locs[-1])
        #    past_loc_deltas = np.vstack(past_loc_deltas)[:,0]
        #else:
        past_loc_deltas = np.array(past_loc_deltas).reshape(-1)
        past_sigma_deltas = []
        for i in range(len(self.past_sigma)):
            past_sigma_deltas.append(self.past_sigma[i])
        past_sigma_deltas = np.array(past_sigma_deltas).reshape(-1)
        past_obj_val_deltas = np.hstack((np.zeros((self.history_len-past_obj_val_deltas.shape[0],)), past_obj_val_deltas))
        past_loc_deltas = np.hstack((np.zeros((self.history_len*2-past_loc_deltas.shape[0],)), past_loc_deltas))
        past_sigma_deltas = np.hstack((np.zeros((self.history_len-past_sigma_deltas.shape[0],)), past_sigma_deltas))

        cur_loc = self.cur_loc
        cur_ps = self.cur_ps
        cur_sigma = self.cur_sigma

        state = {CUR_LOC: cur_loc,
                 PAST_OBJ_VAL_DELTAS: past_obj_val_deltas,
                 CUR_PS: cur_ps,
                 CUR_SIGMA: cur_sigma,
                 PAST_LOC_DELTAS: past_loc_deltas,
                 PAST_SIGMA: past_sigma_deltas
                }
        return state

