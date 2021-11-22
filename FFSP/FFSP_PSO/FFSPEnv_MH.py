
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

import logging
from typing import Union
import numpy as np

from collections.abc import Iterable
from matplotlib.colors import ListedColormap

import torch

from FFSP.FFSProblemDef import get_random_problems_by_random_state, load_problems_from_file


class FFSPEnv_MH:
    def __init__(self, stage_cnt: int = 3, machine_cnt: Union[int, list] = 4, job_cnt: int = 50,
                 min_process_time: Union[int, list] = 1, max_process_time: Union[int, list] = 6,
                 seed: Union[None, int, float] = None, same_process_time_within_stage: bool = False,
                 max_step_length: Union[int, None] = None, distribution: str = 'uniform', file=None):
        self.logger = logging.getLogger('env')

        self.stage_cnt = stage_cnt
        self.machine_cnt_list = machine_cnt if type(machine_cnt) == list else [machine_cnt] * self.stage_cnt
        self.total_machine_cnt = sum(self.machine_cnt_list)
        self.job_cnt = job_cnt
        self.min_process_time_list = min_process_time if type(min_process_time) == list \
            else [min_process_time] * self.stage_cnt
        self.max_process_time_list = max_process_time if type(max_process_time) == list \
            else [max_process_time] * self.stage_cnt
        self.max_process_time = max(self.max_process_time_list)
        self.rand = np.random.RandomState(seed=seed)
        self.seed = seed
        self.distribution = distribution
        self.MAX_STEP_LENGTH = max_step_length
        self.same_process_time_within_stage = same_process_time_within_stage
        self.file = file
        self.problem_filename = self.file['load_filename']
        self.problem_index = self.file['problem_index']

        self.batch_size = None
        self.BATCH_IDX = None

        # state information
        self.step_cnt = None
        self.current_stage_idx = None
        self.current_machine_idx_on_stage = None
        self.current_machine_idx = None
        self.total_schedule_info = None
        self.job_end_time_on_stage = None
        self.process_time = None
        self.best_reward = None
        self.best_end_process_time = None
        self.best_str_process_time = None

        self.ZERO = torch.zeros((1,), dtype=torch.long)
        self.ONE = torch.ones((1,), dtype=torch.long)

    def load_problems(self, batch_size=None, same_problem=False):
        """
        params:
        same_problem: generate one problem and duplicate batch_size if True.
        """
        self.batch_size = batch_size

        self.BATCH_IDX = torch.arange(self.batch_size, dtype=torch.long)

        # generate problems (self.process_time, self.MAX_STEP_LENGTH)
        self._generate_process_time(same_problem=same_problem)
        self.process_time = self.process_time.transpose(1, 2).to(dtype=torch.long)

        self.best_reward = self.MAX_STEP_LENGTH
        self.best_end_process_time = None
        self.best_str_process_time = None

    def reset(self):
        self.step_cnt = 0

        self.current_stage_idx = 0
        self.current_machine_idx_on_stage = 0

        self.current_machine_idx = 0

        self.total_schedule_info = torch.zeros((self.total_machine_cnt, self.MAX_STEP_LENGTH), dtype=torch.long)
        self.job_end_time_on_stage = torch.zeros((self.batch_size, self.stage_cnt, self.job_cnt + 1),
                                                 dtype=torch.long)

    def step(self, particles):
        view_pos = particles.view((self.batch_size, self.stage_cnt, self.job_cnt))
        all_machine_id = view_pos.floor().to(dtype=torch.long)

        sorted_index = (view_pos - all_machine_id).argsort(dim=2)

        time_table = torch.zeros((self.batch_size, self.total_machine_cnt), dtype=torch.long)

        m_str_idx = 0
        m_end_idx = 0

        cumsum_start_time = torch.zeros((self.batch_size, self.total_machine_cnt, self.job_cnt + 1), dtype=torch.long)
        cumsum_end_time = torch.zeros((self.batch_size, self.total_machine_cnt, self.job_cnt + 1), dtype=torch.long)

        for stage, m_cnt in zip(range(self.stage_cnt), self.machine_cnt_list):

            all_machine_id[:, stage, :].clamp_(min=0, max=m_cnt - 1)

            m_end_idx += m_cnt

            time_table_stage_view = time_table[:, m_str_idx:m_end_idx]
            process_time = self.process_time[:, m_str_idx:m_end_idx]

            if stage == 0:
                for j_idx in range(self.job_cnt):
                    c = sorted_index[:, stage, j_idx]

                    machine_id = all_machine_id[self.BATCH_IDX, stage, c]

                    c = c + 1

                    job_process_time = process_time[self.BATCH_IDX, machine_id, c]

                    cumsum_start_time[self.BATCH_IDX, sum(self.machine_cnt_list[:stage]) + machine_id, c] = time_table_stage_view[self.BATCH_IDX, machine_id]

                    time_table_stage_view[self.BATCH_IDX, machine_id] += job_process_time

                    self.job_end_time_on_stage[self.BATCH_IDX, stage, c] = time_table_stage_view[self.BATCH_IDX, machine_id]

                    cumsum_end_time[self.BATCH_IDX, sum(self.machine_cnt_list[:stage]) + machine_id, c] = self.job_end_time_on_stage[self.BATCH_IDX, stage, c]
            else:
                last_stage_job_end_time = self.job_end_time_on_stage[:, stage - 1]

                time = 0

                masking = torch.ones_like(last_stage_job_end_time, dtype=torch.bool)
                masking[:, 0] = 0
                machine_end_time = torch.zeros(size=(self.batch_size, self.machine_cnt_list[stage]))

                while True:
                    time += 1

                    job_list = torch.where((last_stage_job_end_time[:, 1:] <= time) * masking[:, 1:], view_pos[:, stage], torch.tensor(float('inf'), dtype=torch.float64))

                    for machine in range(self.machine_cnt_list[stage]):
                        target_job_list, target_job_idx_list = torch.where((all_machine_id[:, stage, :] >= machine) & (all_machine_id[:, stage, :] < machine + 1) & (torch.isinf(job_list) ^ True), job_list, torch.tensor(float('inf'), dtype=torch.float64)).sort(dim=1)

                        for b in range(self.batch_size):
                            if machine_end_time[b, machine] > time:
                                continue

                            if torch.isinf(target_job_list[b][0]):
                                continue

                            target_job = target_job_idx_list[b][0]

                            masking[b, target_job + 1] = 0

                            str_process_time = time
                            end_process_time = time + process_time[b, machine, target_job + 1]
                            machine_end_time[b, machine] = end_process_time

                            self.job_end_time_on_stage[b, stage, target_job + 1] = end_process_time

                            # batch_size, total_machine_cnt, job_cnt + 1
                            cumsum_start_time[b, m_str_idx + machine, target_job + 1] = str_process_time
                            cumsum_end_time[b, m_str_idx + machine, target_job + 1] = end_process_time

                    if masking.sum() == 0:
                        break

            m_str_idx = m_end_idx

        reward = self.get_reward()

        min_reward, min_idx = reward.min(dim=0)

        if self.best_reward > min_reward:
            self.best_reward = min_reward
            self.best_str_process_time = cumsum_start_time[min_idx]
            self.best_end_process_time = cumsum_end_time[min_idx]

        # for detail reward
        return reward.to(dtype=torch.float32)

    def _generate_process_time(self, same_problem=False):
        self.seed = self.rand.get_state()[1][0]

        do_nothing_process_time = torch.zeros((self.batch_size, 1, self.total_machine_cnt),
                                              dtype=torch.float32)

        if self.file is not None:
            process_time_for_each_stage = load_problems_from_file(filename=self.file['load_filename'])

            if self.file['problem_index'] is not None:
                process_time_for_each_stage = [p[[self.file['problem_index']]] for p in process_time_for_each_stage]

            if same_problem:
                process_time_for_each_stage = [pt.expand(self.batch_size, *pt.size()[1:])
                                               for pt in process_time_for_each_stage]
        elif same_problem:
            process_time_for_each_stage = get_random_problems_by_random_state(
                self.rand,
                1,
                self.machine_cnt_list,
                self.job_cnt,
                distribution=self.distribution,
                same_process_time_within_stage=self.same_process_time_within_stage,
                min_process_time_list=self.min_process_time_list,
                max_process_time_list=self.max_process_time_list)

            process_time_for_each_stage = [pt.expand(self.batch_size, *pt.size()[1:])
                                           for pt in process_time_for_each_stage]
        else:
            process_time_for_each_stage = get_random_problems_by_random_state(
                self.rand,
                self.batch_size,
                self.machine_cnt_list,
                self.job_cnt,
                distribution=self.distribution,
                same_process_time_within_stage=self.same_process_time_within_stage,
                min_process_time_list=self.min_process_time_list,
                max_process_time_list=self.max_process_time_list)

        if self.MAX_STEP_LENGTH is None:
            self.MAX_STEP_LENGTH = int(sum([max(pt.mean(dim=2, dtype=torch.float32).sum(dim=1) / m_cnt) for pt, m_cnt in
                                            zip(process_time_for_each_stage, self.machine_cnt_list)]))

        process_time_for_each_stage = torch.cat(process_time_for_each_stage, dim=2)

        self.process_time = torch.cat((do_nothing_process_time, process_time_for_each_stage.float().expand((self.batch_size, self.job_cnt, self.total_machine_cnt))), dim=1)

    def set_seed(self, seed):
        self.seed = seed
        self.rand.seed(seed)

    def is_done(self, batch=None, done_in_step=False):
        if done_in_step:
            if self.step_cnt >= self.MAX_STEP_LENGTH:
                return False

            if batch is not None:
                return self.job_end_time_on_stage[batch, :, 1:].all()

            return self.job_end_time_on_stage[:, :, 1:].all()
        return (self.job_end_time_on_stage[:, :, 1:] > 0).all() or self.step_cnt >= self.MAX_STEP_LENGTH

    def get_reward(self):
        if self.is_done():
            return self._get_max_step()
        return 0

    def _get_max_step(self):
        return torch.where(self.job_end_time_on_stage[:, -1, 1:].min(dim=-1)[0] == 0,
                           torch.tensor(self.MAX_STEP_LENGTH, dtype=torch.long),
                           self.job_end_time_on_stage.max(dim=2)[0].max(dim=1)[0])

    def stage_machine_2_machine(self, stage, machine):
        assert len(self.machine_cnt_list) > stage, 'stage number {} is out of index.'.format(stage)
        assert self.machine_cnt_list[stage] > machine, 'machine number {} is out of index.'.format(machine)

        return sum(self.machine_cnt_list[:stage]) + machine

    def machine_2_stage_machine(self, in_machine):
        machine = in_machine
        stage = 0
        for m in self.machine_cnt_list:
            if machine >= m:
                machine -= m
                stage += 1
            else:
                break

        assert stage < self.stage_cnt, 'machine number {} is out of index.'.format(in_machine)

        return stage, machine

    def render(self, pause=None, grid=True, y_tick_interval: int = 0, ylim: int = 0, save: bool=False, filename: str='Figure'):
        for m_idx in range(self.total_machine_cnt):
            for j_idx in range(1, self.job_cnt + 1):
                if self.best_end_process_time[m_idx, j_idx] == 0:
                    continue

                from_idx = self.best_str_process_time[m_idx, j_idx]
                to_idx = self.best_end_process_time[m_idx, j_idx]
                self.total_schedule_info[m_idx, from_idx:to_idx] = j_idx

        import matplotlib.pyplot as plt

        colors_list = ['red', 'orange', 'yellow', 'green', 'blue',
                       'purple', 'aqua', 'aquamarine', 'black',
                       'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chocolate',
                       'coral', 'cornflowerblue', 'darkblue', 'darkgoldenrod', 'darkgreen',
                       'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
                       'darkorchid', 'darkred', 'darkslateblue', 'darkslategrey', 'darkturquoise',
                       'darkviolet', 'deeppink', 'deepskyblue', 'dimgrey', 'dodgerblue',
                       'forestgreen', 'gold', 'goldenrod', 'gray', 'greenyellow',
                       'hotpink', 'indianred', 'khaki', 'lawngreen', 'magenta',
                       'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid',
                       'mediumpurple',
                       'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
                       'navy', 'olive', 'olivedrab', 'orangered',
                       'orchid',
                       'palegreen', 'paleturquoise', 'palevioletred', 'pink', 'plum', 'powderblue',
                       'rebeccapurple',
                       'rosybrown', 'royalblue', 'saddlebrown', 'sandybrown', 'sienna',
                       'silver', 'skyblue', 'slateblue',
                       'springgreen',
                       'steelblue', 'tan', 'teal', 'thistle',
                       'tomato', 'turquoise', 'violet', 'yellowgreen']

        reward = self.best_reward.item()
        t = self.total_schedule_info.cpu().numpy()
        r = int(reward) if ylim == 0 else ylim

        fig, axes = plt.subplots(nrows=self.stage_cnt, ncols=1, sharex='all')

        fig.suptitle('reward: {}'.format(reward),
                     verticalalignment='top',
                     horizontalalignment='center',
                     fontweight='bold')

        from_idx = 0
        to_idx = 0
        s_cnt = 0

        if not isinstance(axes, Iterable):
            axes = [axes]

        for ax in axes:
            if grid:
                ax.grid(True, which='minor')
            to_idx += self.machine_cnt_list[s_cnt]
            mask = np.ma.masked_equal(t[from_idx:to_idx, :r], value=0)
            ax.imshow(mask,
                      cmap=ListedColormap(colors_list, N=self.job_cnt),
                      vmin=1,
                      vmax=self.job_cnt,
                      aspect='auto')

            if s_cnt == self.stage_cnt - 1:
                ax.set_xlabel('Steps')
            ax.set_ylabel('Stage {}\nMachine'.format(s_cnt))
            if y_tick_interval == 0:
                ax.set_xticks(np.arange(0, r, int(r/10) + 1), minor=False)
            else:
                ax.set_xticks(np.arange(0, r, y_tick_interval), minor=False)
            ax.set_xticks(np.arange(-0.5, r), minor=True)
            ax.set_yticks(np.arange(self.machine_cnt_list[s_cnt]), minor=False)
            ax.set_yticks(np.arange(-0.5, self.machine_cnt_list[s_cnt]), minor=True)
            from_idx = to_idx
            s_cnt += 1

        if save:
            fig.savefig('{}.jpg'.format(filename), dpi=fig.dpi)
            plt.close('all')
        else:
            if pause is None:
                plt.show()
            else:
                plt.pause(pause)
                plt.close('all')


if __name__ == '__main__':
    env_params = {
        'stage_cnt': 3,
        'machine_cnt': 4,
        'job_cnt': 20,
        'min_process_time': 2,
        'max_process_time': 10,
        'distribution': 'uniform',
        'same_process_time_within_stage': True,
        'seed': 123456789
    }

    env_params['max_step_length'] = env_params['stage_cnt'] * env_params['job_cnt'] * env_params['max_process_time']

    env = FFSPEnv_MH(**env_params)

    env.load_problems(10, True)
    env.reset()
    env.step(torch.rand(10, 3, 20) * 4)
    
