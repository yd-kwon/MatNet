
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
from dataclasses import dataclass
from typing import Union
import numpy as np

from collections.abc import Iterable
from matplotlib.colors import ListedColormap

import torch

from FFSP.FFSProblemDef import get_random_problems_by_random_state, load_problems_from_file

@dataclass
class State:
    machine_cnt_list: list

    step_cnt: int

    current_stage_idx: int
    current_machine_idx_on_stage: int

    current_machine_idx: int

    # shape: (batch_size, total_machine_cnt, max_process_time)
    state_schedule_info: torch.tensor

    # shape: (batch_size, job_cnt + 1, total_machine_cnt): process time of job on machine. job 0 is do nothing.
    process_time: torch.tensor

    # shape: (batch_size, job_cnt + 1, stage_cnt)
    job_end_time_on_stage: torch.tensor

    masking: torch.tensor


class FFSPEnv_Greedy:
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
        # seed 저장.
        self.seed = seed
        self.distribution = distribution
        self.MAX_STEP_LENGTH = max_step_length
        self.same_process_time_within_stage = same_process_time_within_stage
        self.file = file

        self.batch_size = None
        self.BATCH_IDX = None

        # for debugging - start
        self.last_step_cnt = None
        self.last_stage_idx = None
        self.last_machine_idx_on_stage = None
        self.last_machine_idx = None
        self.last_action = None
        # for debugging - end

        # state information
        self.step_cnt = None
        self.current_stage_idx = None
        self.current_machine_idx_on_stage = None
        self.current_machine_idx = None
        self.total_schedule_info = None
        self.state_schedule_info = None
        self.job_end_time_on_stage = None
        self.masking = None

    def load_problems(self, batch_size=None, same_problem=False):
        """
        params:

        same_problem: generate one problem and duplicate batch_size if True.
        """
        self.batch_size = batch_size

        self.BATCH_IDX = torch.arange(self.batch_size, dtype=torch.long)

        # shape: (self.process_time, self.MAX_STEP_LENGTH)
        self._generate_process_time(same_problem=same_problem)


    def reset(self):
        self.step_cnt = 0

        self.current_stage_idx = 0
        self.current_machine_idx_on_stage = 0

        self.current_machine_idx = 0

        # save last action info
        self.last_step_cnt = None
        self.last_stage_idx = None
        self.last_machine_idx_on_stage = None
        self.last_machine_idx = None
        self.last_action = None

        self.total_schedule_info = torch.zeros((self.batch_size, self.total_machine_cnt, self.MAX_STEP_LENGTH),
                                               dtype=torch.long)
        self.state_schedule_info = self.total_schedule_info[:, :, self.step_cnt:self.step_cnt + self.max_process_time].clone()
        self.job_end_time_on_stage = torch.zeros((self.batch_size, self.job_cnt + 1, self.stage_cnt),
                                                 dtype=torch.float32)
        self.masking = torch.zeros((self.batch_size, self.job_cnt + 1), dtype=torch.bool)

        state = State(
            machine_cnt_list=self.machine_cnt_list,
            step_cnt=self.step_cnt,
            current_stage_idx=self.current_stage_idx,
            current_machine_idx_on_stage=self.current_machine_idx_on_stage,
            current_machine_idx=self.current_machine_idx,
            state_schedule_info=self.state_schedule_info,
            process_time=self.process_time,
            job_end_time_on_stage=self.job_end_time_on_stage,
            masking=self.masking
        )

        return state, 0, False

    def step(self, selected_job_idx):
        """
        selected_node_idx.shape: (batch,)
        """
        # save last action info
        self.last_step_cnt = self.step_cnt
        self.last_stage_idx = self.current_stage_idx
        self.last_machine_idx_on_stage = self.current_machine_idx_on_stage
        self.last_machine_idx = self.current_machine_idx
        self.last_action = selected_job_idx

        # move state
        # shape: (batch_size, )
        job_process_time = self.process_time[self.BATCH_IDX, selected_job_idx, self.current_machine_idx]

        max_jpt = int(job_process_time.max())
        finish_step = self.step_cnt + max_jpt
        self.total_schedule_info[self.BATCH_IDX, self.current_machine_idx, self.step_cnt:finish_step] = torch.where(
            torch.arange(max_jpt).expand((self.batch_size, max_jpt)) < job_process_time.to(
                dtype=torch.long)[:, None], selected_job_idx[:, None],
            self.total_schedule_info[self.BATCH_IDX, self.current_machine_idx, self.step_cnt:finish_step])

        # action duplicate check
        assert (self.job_end_time_on_stage[self.BATCH_IDX, selected_job_idx, self.current_stage_idx] == 0).all()

        self.job_end_time_on_stage[
            self.BATCH_IDX, selected_job_idx, self.current_stage_idx] = self.step_cnt + job_process_time
        # clear no action idx info.
        self.job_end_time_on_stage[:, 0, :] = 0

        while not self.is_done():
            # step increase
            self.current_machine_idx += 1

            if self.current_machine_idx == self.total_machine_cnt:
                self.current_machine_idx = 0
                self.step_cnt += 1

            self.current_stage_idx, self.current_machine_idx_on_stage = \
                self.machine_2_stage_machine(self.current_machine_idx)

            if self.is_done():
                break

            # check machine, job state
            self.masking = self.job_end_time_on_stage[:, :, self.current_stage_idx] > 0

            if self.masking[:, 1:].all():
                continue

            if self.current_stage_idx > 0:
                self.masking |= ((self.job_end_time_on_stage[:, :, self.current_stage_idx - 1] == 0) |
                                 (self.job_end_time_on_stage[:, :, self.current_stage_idx - 1] > self.step_cnt))
                self.masking[:, 0] = False

            if self.masking[:, 1:].all():
                continue

            self.masking |= self.total_schedule_info[:, self.current_machine_idx, [self.step_cnt]].expand(
                size=(self.batch_size, self.job_cnt + 1)
            ) > 0
            self.masking[:, 0] = False

            if self.masking[:, 1:].all():
                continue

            self.masking |= self.process_time[:, :, self.current_machine_idx] > (self.MAX_STEP_LENGTH - self.step_cnt)

            # skip if all jobs not available
            if self.masking[:, 1:].all():
                continue

            break

        self.state_schedule_info = self.total_schedule_info[:, :, self.step_cnt:self.step_cnt+self.max_process_time].clone()

        state = State(
            machine_cnt_list=self.machine_cnt_list,
            step_cnt=self.step_cnt,
            current_stage_idx=self.current_stage_idx,
            current_machine_idx_on_stage=self.current_machine_idx_on_stage,
            current_machine_idx=self.current_machine_idx,
            state_schedule_info=self.state_schedule_info,
            process_time=self.process_time,
            job_end_time_on_stage=self.job_end_time_on_stage,
            masking=self.masking
        )

        return state, self.get_reward(), self.is_done()

    def _generate_process_time(self, same_problem=False):
        self.seed = self.rand.get_state()[1][0]

        do_nothing_process_time = torch.zeros((self.batch_size, 1, self.total_machine_cnt),
                                              dtype=torch.float32)

        if self.file is not None:
            process_time_for_each_stage = load_problems_from_file(filename=self.file['load_filename'])

            if self.file['problem_index'] is not None:
                process_time_for_each_stage = [p[[self.file['problem_index']]] for p in process_time_for_each_stage]

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
        self.process_time = torch.cat((do_nothing_process_time, process_time_for_each_stage.float()), dim=1)

    def set_seed(self, seed):
        self.seed = seed
        self.rand.seed(seed)

    def is_done(self, batch=None, done_in_step=False):
        if done_in_step:
            if self.step_cnt >= self.MAX_STEP_LENGTH:
                return False

            if batch is not None:
                return self.job_end_time_on_stage[batch, 1:, :].all()

            return self.job_end_time_on_stage[:, 1:, :].all()
        return (self.job_end_time_on_stage[:, 1:, :] > 0).all() or self.step_cnt >= self.MAX_STEP_LENGTH

    def get_reward(self):
        if self.is_done():
            return self._get_max_step()
        return 0

    def _get_max_step(self):
        return torch.where(self.job_end_time_on_stage[:, 1:, -1].min(dim=-1)[0] == 0,
                           torch.tensor(self.MAX_STEP_LENGTH, dtype=torch.float32),
                           self.job_end_time_on_stage.max(dim=1)[0].max(dim=1)[0])

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

    def render(self, mode='text', pause=None, batch: Union[int, list] = None, grid=False, y_tick_interval: int = 0,
               ylim: int = 0, save: bool=False, filename: str='Figure'):
        if self.total_schedule_info is None:
            return

        if batch is None:
            batch = list(range(self.batch_size))
        elif type(batch) is int:
            batch = [batch]

        if self.last_step_cnt is None:
            if mode == 'debug':
                print('=========================================================')
                for b_idx in batch:
                    print('batch {}'.format(b_idx))
                    print(self.process_time[b_idx])
                print('=========================================================')
            return

        if mode == 'debug':
            print('step: {} stage: {} machine: {}'.format(self.last_step_cnt,
                                                          self.last_stage_idx,
                                                          self.last_machine_idx_on_stage))
            print('actions: {}'.format(self.last_action[batch]))

        if mode == 'text' or mode == 'debug':
            # "\n".join
            text = []

            rewards = self.get_reward()
            for b_idx in batch:
                t = self.total_schedule_info[b_idx]
                r = rewards[b_idx]

                text.append('batch {}'.format(b_idx))
                # " ".join
                inner_text = ['stage', 'machine']

                for i in range(r):
                    inner_text.append(str(i))

                text.append(" ".join(inner_text))

                for s_idx, s in enumerate(t[:, :r]):
                    stg, mch = self.machine_2_stage_machine(s_idx)
                    text.append('{} {} {}'.format(stg, mch, ' '.join(s.astype('str'))))

            print('{}'.format('\n'.join(text)))
        elif mode == 'plot':
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

            rewards = self._get_max_step()

            for b_idx in batch:
                t = self.total_schedule_info[b_idx].cpu().numpy()
                r = int(rewards[b_idx].item()) if ylim == 0 else ylim

                fig, axes = plt.subplots(nrows=self.stage_cnt, ncols=1, sharex='all')

                fig.suptitle('{} done.\nreward: {}'.format('' if self.is_done(b_idx) else 'not', rewards[b_idx]),
                             verticalalignment='top',
                             horizontalalignment='center',
                             fontweight='bold')

                from_idx = 0
                to_idx = 0
                s_cnt = 0

                if not isinstance(axes, Iterable):
                    axes = [axes]

                for ax in axes:
                    to_idx += self.machine_cnt_list[s_cnt]
                    mask = np.ma.masked_equal(t[from_idx:to_idx, :r], value=0)
                    ax.imshow(mask,
                              cmap=ListedColormap(colors_list, N=self.job_cnt),
                              vmin=1,
                              vmax=self.job_cnt,
                              aspect='auto')

                    if grid:
                        ax.grid(True, which='minor')
                    else:
                        for r_idx, row in enumerate(mask):
                            for c_idx, value in enumerate(row):
                                if value == '--':
                                    continue
                                elif c_idx > 0 and row[c_idx - 1] == value:
                                    continue
                                ax.text(c_idx, r_idx, value)

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
                    fig.savefig('{}_{}.jpg'.format(filename, b_idx), dpi=fig.dpi)
                    plt.close('all')
                else:
                    if pause is None:
                        plt.show()
                    else:
                        plt.pause(pause)
                        plt.close('all')
