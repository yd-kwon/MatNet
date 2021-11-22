
"""
The MIT License

Copyright (c) 2021 MatNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

##########################################################################################
# Machine Environment Config

DEBUG_MODE = False

USE_CUDA = False
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, "../..")
sys.path.insert(0, "..")

##########################################################################################
# cuda

import torch

if USE_CUDA:
    torch.cuda.set_device(CUDA_DEVICE_NUM)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

##########################################################################################
# import

import logging

from utils.utils import create_logger, copy_all_src

from FFSP.FFSP_Greedy.GreedyAlgorithm import GreedyAlgorithm
from FFSP.FFSP_GA.FFSPEnv_MH import FFSPEnv_MH as Env
from FFSP.FFSP_GA.GA import GA as Trainer

##########################################################################################
# parameters

env_params = {
    'stage_cnt': 3,
    'machine_cnt': 4,
    'job_cnt': 20,
    'min_process_time': 2,
    'max_process_time': 10,
    'distribution': 'uniform',
    'same_process_time_within_stage': False,
    'file': {
        'load_filename': '../data/unrelated_10000_problems_444_job20_2_10.pt',
        'problem_index': 0
    },
    'seed': 123456789
}

env_params['max_step_length'] = int(env_params['stage_cnt'] * env_params['job_cnt'] * env_params['max_process_time'] / env_params['machine_cnt'])

trainer_params = {
    'epochs': 5000,
    'target_fitness': None,
    'num_of_chromosome': 25,
    'num_of_gene': env_params['job_cnt'],
    'selection_method': 'softmax',
    'selection_ratio': 0.1,
    'crossover_method': 'FFSP',
    'crossover_ratio': 0.3,
    'mutation_mode': None,
    'mutation_rate': 0.3,
    'max_mutation_rate': 0.3,
    'logging': {
        'img_save_interval': 1,
        'filename': 'figure'
    }
}

logger_params = {
    'log_file': {
        'desc': 'ga',
        'filename': 'ffsp_ga_log.txt'
    }
}


def set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['num_of_chromosome'] = 4


def print_config(logger):
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################
# main

def main():
    create_logger(**logger_params)

    if DEBUG_MODE:
        set_debug_mode()

    logger = logging.getLogger('FFSP_GA_main')

    print_config(logger)

    for i in range(3):
        cmax = 0
        logger.info('problem index: {}'.format(i))
        env_params['file']['problem_index'] = i

        greedy = GreedyAlgorithm(episodes=1, batch_size=1, env_params=env_params)
        greedy.solve(method='SJFv2',  plot='render')
        end_time = greedy.env.job_end_time_on_stage.max().to(dtype=torch.long)
        schedule = greedy.env.total_schedule_info[0, :, :end_time]

        for _ in range(128):
            env = Env(**env_params)

            trainer = Trainer(env=env, schedule=schedule, trainer_params=trainer_params)

            cmax += trainer.solve()

        logger.info('problem: {} mean: {}'.format(i, cmax / 128))


if __name__ == "__main__":
    main()
