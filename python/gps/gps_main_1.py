# import logging
# import imp
# import os
# import os.path
# import sys
# import argparse
# import time
# import numpy as np
# import random

# # Add gps/python to path so that imports work.
# sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
# import gps as gps_globals
# from gps.utility.display import Display
# from gps.sample.sample_list import SampleList

# class GPSMain(object):
    # #""" Main class to run algorithms and experiments. """
    # def __init__(self, config):
        # """
        # Initialize GPSMain
        # Args:
            # config: Hyperparameters for experiment
        # """
        # self._hyperparams = config
        # self._conditions = config['common']['conditions']
        # if 'train_conditions' in config['common']:
            # self._train_idx = config['common']['train_conditions']
            # self._test_idx = config['common']['test_conditions']
        # else:
            # self._train_idx = range(self._conditions)
            # config['common']['train_conditions'] = config['common']['conditions']
            # self._hyperparams=config
            # self._test_idx = self._train_idx

        # self._data_files_dir = config['common']['data_files_dir']

        # self.agent = config['agent']['type'](config['agent'])
        # self.disp = Display(config['common'])     # For logging

        # config['algorithm']['agent'] = self.agent
        # self.algorithm = config['algorithm']['type'](config['algorithm'])

    # def run(self):

        # itr_start = 0

        # for itr in range(itr_start, self._hyperparams['iterations']):
            # for m, cond in enumerate(self._train_idx):
                # for i in range(self._hyperparams['num_samples']):
                    # self._take_sample(itr, cond, m, i)

            # traj_sample_lists = [self.agent.get_samples(cond, -self._hyperparams['num_samples']) for cond in self._train_idx]

            # # Clear agent samples.
            # self.agent.clear_samples()

            # self.algorithm.iteration(traj_sample_lists)

            # #pol_sample_lists = self._take_policy_samples(self._train_idx)

            # #self._prev_traj_costs, self._prev_pol_costs = self.disp.update(itr, self.algorithm, self.agent, traj_sample_lists, pol_sample_lists)
            # pol_sample_lists = self._take_policy_samples(self._test_idx)
            # self._prev_traj_costs, self._prev_pol_costs = self.disp.update(self.algorithm, self.agent, self._test_idx, pol_sample_lists)

            # self.algorithm.policy_opt.policy.pickle_policy(self.algorithm.policy_opt._dO, self.algorithm.policy_opt._dU, self._data_files_dir + ('policy_itr_%02d' % itr))

        # #pol_sample_lists = self._take_policy_samples(self._test_idx)
        # #self._prev_traj_costs, self._prev_pol_costs = self.disp.update(self.algorithm, self.agent, self._test_idx, pol_sample_lists)

        # if 'on_exit' in self._hyperparams:
            # self._hyperparams['on_exit'](self._hyperparams)

    # def _take_sample(self, itr, cond, m, i):

        # if self.algorithm.iteration_count == 0:
            # pol = self.algorithm.cur[m].traj_distr
        # else:
            # if self.algorithm._hyperparams['sample_on_policy']:
                # pol = self.algorithm.policy_opt.policy
            # else:
                # pol = self.algorithm.cur[m].traj_distr

        # self.agent.sample(pol, cond)

    # def _take_policy_samples(self, cond_list):
        # pol_samples = [[] for _ in range(len(cond_list))]
        # for cond in range(len(cond_list)):
            # for i in range(self._hyperparams['num_samples']):
                # pol_samples[cond].append(self.agent.sample(self.algorithm.policy_opt.policy, cond_list[cond], save=False))
        # return [SampleList(samples) for samples in pol_samples]

# def main():
    # parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    # parser.add_argument('experiment', type=str, help='experiment name')
    # args = parser.parse_args()

    # exp_name = args.experiment

    # from gps import __file__ as gps_filepath
    # gps_filepath = os.path.abspath(gps_filepath)
    # gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    # exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    # hyperparams_file = exp_dir + 'hyperparams.py'

    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    # if not os.path.exists(hyperparams_file):
        # sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" % (exp_name, hyperparams_file))

    # # May be used by hyperparams.py to load different conditions
    # gps_globals.phase = "TRAIN"
    # hyperparams = imp.load_source('hyperparams', hyperparams_file)

    # seed = hyperparams.config.get('random_seed', 0)
    # random.seed(seed)
    # np.random.seed(seed)

    # gps = GPSMain(hyperparams.config)
    # gps.run()

    # if 'on_exit' in hyperparams.config:
        # hyperparams.config['on_exit'](hyperparams.config)


# if __name__ == "__main__":
    # main()
