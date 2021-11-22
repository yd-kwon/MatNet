
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

import numpy as np
import torch


def calc_lower_bound(job_cnt, stage_cnt, machine_cnt_list, process_time, problem_type='identical'):
    """
    job_cnt: # of jobs
    stage_cnt: # of stages
    machine_cnt_list[j]: # of machines on stage j

    when problem_type = 'identical'
        process_time[i, j]: job i's process time on stage j
    when problem_type = 'unrelated'
        process_time[i, j, k]: job i's process time on stage j on machine k
    """

    if problem_type == 'identical':
        if type(process_time) == np.ndarray or type(process_time) == torch.Tensor:
            assert process_time.shape == (
            job_cnt, stage_cnt), 'process time shape is not match. (expected: ({}, {}), actual: ({}))'.format(job_cnt,
                                                                                                              stage_cnt,
                                                                                                              process_time.shape)
        LB = np.zeros(shape=(stage_cnt + 1), dtype=np.long)

        LB[0] = process_time.sum(1).max()

        # RSA, LSA: (# of machines, # of stage)
        for j in range(stage_cnt):
            # RS: (# of jobs, # of stage)
            RS_J = process_time[:, j + 1:].sum(1)
            LS_J = process_time[:, :j].sum(1)
            RSA_J = np.sort(RS_J)
            LSA_J = np.sort(LS_J)

            LB[j + 1] = np.ceil(1. / machine_cnt_list[j] * (
                        LSA_J[:machine_cnt_list[j]].sum()
                        + (process_time[:, j].sum(0).to(dtype=torch.float32) if type(process_time) == torch.Tensor
                           else process_time[:, j].sum(0))
                        + RSA_J[:machine_cnt_list[j]].sum()))
            
        LBMAX = LB.max()

        return LBMAX
    elif problem_type == 'unrelated':
        total_machine_cnt = sum(machine_cnt_list)

        if type(process_time) == np.ndarray or type(process_time) == torch.Tensor:
            assert process_time.shape == (
                job_cnt,
                total_machine_cnt), 'process time shape is not match. (expected: ({}, {}), actual: ({}))'.format(
                job_cnt, total_machine_cnt, process_time.shape)

        str_machine_idx = 0
        end_machine_idx = 0

        pt = np.empty((job_cnt, stage_cnt))

        for idx, m_cnt in enumerate(machine_cnt_list):
            end_machine_idx += m_cnt

            if type(process_time) == np.ndarray:
                pt[:, idx] = process_time[:, str_machine_idx:end_machine_idx].min(1)
            elif type(process_time) == torch.Tensor:
                pt[:, idx] = process_time[:, str_machine_idx:end_machine_idx].min(1)[0]

            str_machine_idx = end_machine_idx

        LB1 = np.zeros(shape=stage_cnt, dtype=np.long)
        LB2 = np.zeros(shape=stage_cnt, dtype=np.long)

        # RSA, LSA: (# of machines, # of stage)
        for j in range(stage_cnt):
            # RS: (# of jobs, # of stage)
            RS_J = pt[:, j + 1:].sum(1)
            LS_J = pt[:, :j].sum(1)
            RSA_J = np.sort(RS_J)
            LSA_J = np.sort(LS_J)
            LB1[j] = np.ceil(1 / machine_cnt_list[j] * (
                    LSA_J[:machine_cnt_list[j]].sum()
                    + (0 if j == 0 else (pt.min(0)[:j].sum() * max(machine_cnt_list[j] - machine_cnt_list[j - 1], 0)))
                    + pt[:, j].sum(0)
                    + RSA_J[:machine_cnt_list[j]].sum()
            ))
            LB2[j] = np.ceil(1 / machine_cnt_list[j] * (
                    RSA_J[:machine_cnt_list[j]].sum()
                    + (0 if j == (stage_cnt - 1) else (pt.min(0)[j:].sum() * max(machine_cnt_list[j] - machine_cnt_list[j + 1], 0)))
                    + pt[:, j].sum(0)
                    + LSA_J[:machine_cnt_list[j]].sum()
            ))

        LBMAX = np.max((LB1, LB2))

        return LBMAX

    return
