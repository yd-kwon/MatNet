
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

import os
import logging
import torch
import numpy as np

#from FFSP.FFSP_PSO.FlexibleFlowShopEnv_v2 import FlexibleFlowShopEnv as Env
from utils.utils import get_result_folder


class PSO:
    def __init__(self, env, schedule, n_particles, epochs=5000, alpha=0.6, w=0.7, c1=0.15, c2=0.15, filename=None):
        assert c1 > 0 and c1 > 0, "cognitive(c1) and social(c2) parameter should be positive."
        assert 0 <= alpha <= 1, "decrement factor(alpha) should be between 0 and 1."


        self.logger = logging.getLogger('trainer')
        self.result_folder = get_result_folder()

        # Step 1. Initialize the parameters such as population size, maximum iteration, decrement factor,
        #       inertia weight, social and cognitive parameters.
        # population size
        self.n_particles = n_particles
        # maximum iteration
        self.epochs = epochs
        # decrement factor
        self.alpha = alpha
        # inertia weight
        self.w = w
        # social and cognitive parameters
        self.c1 = c1
        self.c2 = c2

        # Step 2. Input number of jobs, number of stages, number of machines at each stage, and processing times.
        self.problem = env

        self.problem_file_name = env.problem_filename
        self.problem_index = env.problem_index
        self.problem.load_problems(batch_size=self.n_particles)

        self.seed = env.seed

        self.sjf_particle = self.schedule_encoding(schedule)
        self.chaotic_number = ChaoticNumberGenerator(n=np.random.rand())
        self.particles_pos = self.generate_position()
        self.velocities = self.generate_velocity()

        self.max_machine_cnt = max(self.problem.machine_cnt_list)
        self.ZERO = torch.zeros(1, dtype=torch.long)
        self.ONE = torch.tensor(1)
        self.JOB_CNT = torch.tensor(self.problem.job_cnt)
        self.INDEX_MATRIX_1D = torch.arange(self.n_particles)
        self.INDEX_MATRIX_4D = torch.arange(self.problem.job_cnt).expand((self.n_particles,
                                                                          self.problem.stage_cnt,
                                                                          self.max_machine_cnt,
                                                                          self.problem.job_cnt))

        self.global_best = {}
        self.personal_best = {}

    def schedule_encoding(self, schedule):
        if schedule is None:
            return None
        # stage_cnt * jobs_cnt
        particle = torch.zeros(size=(self.problem.job_cnt * self.problem.stage_cnt + 1,), dtype=torch.float64)

        m_str_idx = self.problem.total_machine_cnt
        m_end_idx = self.problem.total_machine_cnt

        max_length = schedule.size(1)
        order_unit_precision = 1 / max_length

        for stage, m_cnt in zip(range(self.problem.stage_cnt-1, -1, -1), self.problem.machine_cnt_list):
            m_str_idx -= m_cnt

            stage_schedule = schedule[m_str_idx:m_end_idx]

            for step in range(max_length):
                particle[self.problem.job_cnt * stage + stage_schedule[:, step]] = torch.arange(m_cnt, dtype=torch.float64) + order_unit_precision * step

            m_end_idx = m_str_idx

        return particle[1:]

    def generate_position(self):
        if self.sjf_particle is None:
            # shape: (self.n_particles, stage_cnt * jobs_cnt)
            # x = x_min + (x_max - x_min) * N(0, 1)
            return torch.cat([self.chaotic_number.rand(size=(self.n_particles, self.problem.job_cnt), low=0, high=mc)
                              for mc in self.problem.machine_cnt_list],
                             dim=1)
        else:
            return torch.cat((self.sjf_particle[None], torch.cat(
                [self.chaotic_number.rand(size=(self.n_particles - 1, self.problem.job_cnt), low=0, high=mc) for mc in
                 self.problem.machine_cnt_list], dim=1)))

    def generate_velocity(self):
        # shape: (self.n_particles, stage_cnt * jobs_cnt)
        # v = v_min + (v_max - v_min) * N(0, 1)
        return torch.cat([self.chaotic_number.rand(size=(self.n_particles, self.problem.job_cnt), low=-mc, high=mc)
                          for mc in self.problem.machine_cnt_list],
                         dim=1)

    def get_schedule(self, particles):
        view_pos = particles.view((self.n_particles, self.problem.stage_cnt, self.problem.job_cnt))

        sorted_index = view_pos.argsort(dim=2)
        machine_number = view_pos.gather(dim=2, index=sorted_index).floor()

        machine_number.clamp_(min=0, max=self.max_machine_cnt-1)

        schedule = torch.stack([torch.where(machine_number == m, sorted_index, -self.ONE)
                                for m in range(self.max_machine_cnt)], dim=2)
        machine_job_index = torch.where(schedule >= 0, self.INDEX_MATRIX_4D, self.JOB_CNT).min(dim=3)[0]
        schedule = schedule + 1

        return schedule, machine_job_index

    def evaluate(self, particles):
        self.problem.reset()
        return self.problem.step(particles)

    def update_position(self):
        self.particles_pos = self.particles_pos + self.velocities

    def update_velocity(self, personal_best, global_best):
        # r1, r2: a chaotic number between 0 and 1
        r1 = self.chaotic_number.rand(size=(1,))
        r2 = self.chaotic_number.rand(size=(1,))

        self.velocities = self.w * self.velocities + self.c1 * r1 * (personal_best - self.particles_pos) +\
                          self.c2 * r2 * (global_best - self.particles_pos)

    def update_inertia_weight(self):
        self.w *= self.alpha

    def optimize(self):
        # Step 4. Get the schedule using encoding scheme as mentioned in Section 5.
        # Step 5. Evaluate each particle’s fitness (makespan).
        fitness = self.evaluate(self.particles_pos)

        # Step 6. Find out the personal_best and global_best.
        personal_best_fitness = fitness
        personal_best = self.particles_pos
        global_best_fitness = personal_best_fitness.min()
        gb_idx = personal_best_fitness.argmin()
        global_best = personal_best[gb_idx]

        self.global_best = {
            'global_best_fitness': global_best_fitness,
            'global_best': global_best
        }

        self.personal_best = {
            'personal_best_fitness': personal_best_fitness,
            'personal_best': personal_best
        }

        delta = 0
        maxt = 0

        self.logger.info('start best fitness: {}'.format(global_best_fitness))

        for e in range(self.epochs):
            # Step 8. Update velocity, position and inertia weight by using Eqs. 1, 2, and 3.
            self.update_velocity(personal_best, global_best)
            self.update_position()
            self.update_inertia_weight()

            # Step 4. Get the schedule using encoding scheme as mentioned in Section 5.
            # Step 5. Evaluate each particle’s fitness (makespan).
            fitness = self.evaluate(self.particles_pos)

            # Step 6. Find out the personal_best and global_best.
            personal_best_idx = torch.where(personal_best_fitness > fitness, self.ONE, self.ZERO)
            personal_best_updated = (personal_best_idx == 1).any()

            if personal_best_updated:
                # personal best update
                personal_best_fitness = torch.stack((personal_best_fitness, fitness),
                                                    dim=1)[self.INDEX_MATRIX_1D, personal_best_idx]
                personal_best = torch.stack((personal_best, self.particles_pos),
                                            dim=1)[self.INDEX_MATRIX_1D, personal_best_idx]

                # global best update
                p_best_fitness_min, p_best_fitness_min_idx = personal_best_fitness.min(dim=0)

                if global_best_fitness > p_best_fitness_min:
                    global_best_fitness = p_best_fitness_min
                    global_best = personal_best[p_best_fitness_min_idx]

                    if self.global_best['global_best_fitness'] > global_best_fitness:
                        self.global_best = {
                            'global_best_fitness': global_best_fitness,
                            'global_best': global_best
                        }
                        self.personal_best = {
                            'personal_best_fitness': personal_best_fitness,
                            'personal_best': personal_best
                        }

            # Step 7. If no progress is observed in personal_best value for an elapsed number of iterations DELTA,
            #       carry out mutation of a particle using the mutation strategy as outlined in Section 3.3 provided
            #       DELTA is greater than a chaotic number between 0 and maximum time of no progress (MAXT).
            if personal_best_updated:
                delta = 0
                info = '({} personal best updated.)'.format(sum(personal_best_idx))
            else:
                delta += 1
                maxt = delta if delta > maxt else maxt
                info = '(delta: {} maxt: {})'.format(delta, maxt)

            if delta > self.chaotic_number.randint(size=(1,), high=maxt):
                # mutate global best
                stage_num = self.chaotic_number.randint(size=(1,), high=self.problem.stage_cnt).item()
                low = stage_num * self.problem.job_cnt
                high = (stage_num + 1) * self.problem.job_cnt

                selected_machines = self.chaotic_number.randint(size=(2,), low=low, high=high)
                m_global_best = global_best.clone()
                m_global_best[selected_machines] = global_best[selected_machines.flip(dims=(0,))]
                m_global_best_fitness = self.evaluate(m_global_best.expand(self.n_particles, *m_global_best.size()))

                if global_best_fitness > m_global_best_fitness[0]:
                    global_best = m_global_best
                    global_best_fitness = m_global_best_fitness[0]

                    if self.global_best['global_best_fitness'] > global_best_fitness:
                        self.global_best = {
                            'global_best_fitness': global_best_fitness,
                            'global_best': global_best
                        }

                selected_machines = self.chaotic_number.randint(size=(self.n_particles, 2), low=low, high=high)

                personal_best[self.INDEX_MATRIX_1D, selected_machines[:, 0]],\
                personal_best[self.INDEX_MATRIX_1D, selected_machines[:, 1]]\
                    = personal_best[self.INDEX_MATRIX_1D, selected_machines[:, 1]],\
                      personal_best[self.INDEX_MATRIX_1D, selected_machines[:, 0]]
                info += ' - mutation occurred!'

                personal_best_fitness = self.evaluate(personal_best)

                # global best update
                p_best_fitness_min, p_best_fitness_min_idx = personal_best_fitness.min(dim=0)

                if global_best_fitness > p_best_fitness_min:
                    global_best_fitness = p_best_fitness_min
                    global_best = personal_best[p_best_fitness_min_idx]

                    if self.global_best['global_best_fitness'] > global_best_fitness:
                        self.global_best = {
                            'global_best_fitness': global_best_fitness,
                            'global_best': global_best
                        }
                        self.personal_best = {
                            'personal_best_fitness': personal_best_fitness,
                            'personal_best': personal_best
                        }

            if e % 50 != 0:
                continue
            self.logger.info('epoch: {:4} global best: {} p_b best: {} {}'.format(e,
                                                                                  global_best_fitness,
                                                                                  min(personal_best_fitness),
                                                                                  info))

        # plot best image
        self.logger.info(self.global_best['global_best'].numpy())
        filename = os.path.join(self.result_folder, 'Figure_PSO_with_SJFv2_{}_{}'.format(self.problem_index,
                                                                              int(self.global_best['global_best_fitness'])))
        self.problem.render(save=True, filename=filename)

        return int(self.global_best['global_best_fitness'])


class ChaoticNumberGenerator:
    def __init__(self, r=4, n=0.1):
        self.R = r
        self.N = n

    def rand(self, size, low=0, high=1):
        total = 1

        for s in size:
            total *= s

        return torch.tensor([self._rand(low, high) for _ in range(total)], dtype=torch.float64).reshape(size)

    def _rand(self, low, high):
        self.N = self.R * self.N * (1 - self.N)

        return low + self.N * (high - low)

    def randint(self, size, low=0, high=1):
        return self.rand(size, low, high).to(dtype=torch.long)
