
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
# Path Config
import os
import sys

sys.path.insert(0, "../..")

##########################################################################################
# import
import logging
import torch

from FFSP.FFSP_Greedy.FFSPEnv_Greedy import FFSPEnv_Greedy as Env
from utils.utils import create_logger, get_result_folder


##########################################################################################
# main

class GreedyAlgorithm:
    def __init__(self, episodes, batch_size, env_params):
        self.logger = logging.getLogger('trainer')
        self.episodes = episodes
        self.batch_size = batch_size
        self.env = Env(**env_params)
        self.max_process_time = env_params['max_process_time']
        self.seed = env_params['seed']
        self.SJFv2_actions = None
        self.BATCH_IDX = torch.arange(self.batch_size)
        self.step = None
        self.stage_idx = None
        self.result_folder = get_result_folder()
        self.problem_idx = env_params['file']['problem_index']


    def get_action(self, method, state):
        if method == 'SJF':
            pt = state.process_time[:, :, state.current_machine_idx].clone()
            pt[:, 0] = self.max_process_time + 1
            pt[state.masking] = self.max_process_time + 2
            return pt.argmin(axis=1)
        elif method == 'SJFv2':
            if state.step_cnt == self.step and state.current_stage_idx == self.stage_idx:
                return self.SJFv2_actions[:, state.current_machine_idx_on_stage]

            self.step = state.step_cnt
            self.stage_idx = state.current_stage_idx

            current_stage_start_idx = sum(state.machine_cnt_list[:state.current_stage_idx])
            current_stage_end_idx = sum(state.machine_cnt_list[:state.current_stage_idx + 1])
            pt = state.process_time[:, :, current_stage_start_idx:current_stage_end_idx].clone()
            pt[:, 0] = self.max_process_time + 1
            pt[state.masking] = self.max_process_time + 2

            self.SJFv2_actions = torch.zeros(size=(self.batch_size, state.machine_cnt_list[state.current_stage_idx]), dtype=torch.long)

            for idx in range(state.machine_cnt_list[state.current_stage_idx]):
                # machine masking
                pt[self.BATCH_IDX, :, idx] = torch.where(state.state_schedule_info[self.BATCH_IDX, current_stage_start_idx + idx, 0] > 0,
                                                         torch.tensor(self.max_process_time + 2, dtype=torch.float32),
                                                         pt[self.BATCH_IDX, :, idx])

            for _ in range(state.machine_cnt_list[state.current_stage_idx]):
                pt_values, job_indices = pt.sort(axis=1)

                order, machine_indices = pt_values[:, 0].sort(axis=1)

                if order.eq(self.max_process_time + 2).all():
                    break

                self.SJFv2_actions[self.BATCH_IDX, machine_indices[:, 0]] = job_indices[self.BATCH_IDX, 0, machine_indices[:, 0]]

                # job masking
                pt[self.BATCH_IDX, job_indices[self.BATCH_IDX, 0, machine_indices[:, 0]]] = self.max_process_time + 2
                # machine masking
                pt[self.BATCH_IDX, :, machine_indices[:, 0]] = self.max_process_time + 2

            return self.SJFv2_actions[:, state.current_machine_idx_on_stage]
        elif method == 'LJF':
            pt = state.process_time[:, :, state.current_machine_idx].clone()
            pt[state.masking] = -1
            return pt.argmax(axis=1)
        elif method == 'RAND':
            return (state.masking ^ True).type(dtype=torch.float32).multinomial(num_samples=1).squeeze(1)
        else:
            raise Exception('method value ({}) is not defined.'.format(method))

    def solve(self, method, plot=None):
        episode = 0
        total_reward = 0
        reward_for_statistics = []

        self.env.set_seed(self.seed)

        while True:
            if episode + self.batch_size < self.episodes:
                batch = self.batch_size
            else:
                batch = self.episodes - episode

            if batch == 0:
                break

            self.env.load_problems(batch_size=batch)
            state, reward, done = self.env.reset()

            while not done:
                a = self.get_action(method, state)
                state, reward, done = self.env.step(a)

            episode += batch

            total_reward += reward.sum()
            reward_for_statistics.extend(reward.tolist())

            if episode % (self.batch_size * 80) == 0:
                print('{} {}'.format(episode, reward.mean(dtype=torch.float32)))

        if plot is not None:
            filename = os.path.join(self.result_folder, 'Figure_{}_{}_{}'.format(method, self.problem_idx, total_reward))
            self.env.render('plot', save=True, filename=filename)

        self.logger.info('{} {}'.format(method, (total_reward / self.episodes).item()))

        return total_reward


def main():
    ##########################################################################################
    # parameters

    max_episode = 1
    max_batch_size = 1

    env_params = {
        'stage_cnt': 3,
        'machine_cnt': 4,
        'job_cnt': 20,
        'min_process_time': 2,
        'max_process_time': 10,
        'seed': 123456789,
        'distribution': 'uniform',
        'same_process_time_within_stage': False,
        'file':{
            'load_filename': '../data/unrelated_10000_problems_444_job20_2_10.pt',
            'problem_index': 0
        }
    }

    env_params['max_step_length'] = int(
        env_params['stage_cnt'] * env_params['job_cnt'] * env_params['max_process_time'] / env_params['machine_cnt'])

    logger_params = {
        'log_file': {
            'desc': 'greedy',
            'filename': 'ffsp_greedy_log.txt'
        }
    }
    create_logger(**logger_params)

    solver = GreedyAlgorithm(episodes=max_episode, batch_size=max_batch_size, env_params=env_params)

    # shortest_job_first
    solver.solve(method='SJF')

    # shortest_job_first version 2
    solver.solve(method='SJFv2')

    # longest_job_first
    solver.solve(method='LJF')

    # random_job_selection
    solver.solve(method='RAND')


if __name__ == '__main__':
    import os

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    main()
