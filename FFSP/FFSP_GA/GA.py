
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

from utils.utils import get_result_folder

class GA:
    def __init__(self, env, trainer_params, schedule=None):
        self.logger = logging.getLogger('trainer')
        self.result_folder = get_result_folder()

        self.env = env

        self.epochs = trainer_params['epochs']
        self.target_fitness = trainer_params['target_fitness']
        self.num_of_chromosome = trainer_params['num_of_chromosome']
        self.num_of_gene = trainer_params['num_of_gene']
        self.selection_method = trainer_params['selection_method']
        self.crossover_method = trainer_params['crossover_method']
        self.crossover_ratio = trainer_params['crossover_ratio']
        self.mutation_mode = trainer_params['mutation_mode']
        self.init_mutation_rate = trainer_params['mutation_rate']
        self.mutation_rate = trainer_params['mutation_rate']
        self.max_mutation_rate = trainer_params['max_mutation_rate']
        self.img_save_interval = trainer_params['logging']['img_save_interval']
        self.img_filename = trainer_params['logging']['filename']

        self.sjf_particle = self.schedule_encoding(schedule)

        self.ZERO = torch.zeros(1, dtype=torch.long)

    def schedule_encoding(self, schedule):
        if schedule is None:
            return None
        # stage_cnt * jobs_cnt
        particle = torch.zeros(size=(self.env.job_cnt * self.env.stage_cnt + 1,), dtype=torch.float32)

        m_str_idx = self.env.total_machine_cnt
        m_end_idx = self.env.total_machine_cnt

        max_length = schedule.size(1)
        order_unit_precision = 1 / max_length

        for stage, m_cnt in zip(range(self.env.stage_cnt-1, -1, -1), self.env.machine_cnt_list):
            m_str_idx -= m_cnt

            stage_schedule = schedule[m_str_idx:m_end_idx]

            for step in range(max_length):
                particle[self.env.job_cnt * stage + stage_schedule[:, step]] = torch.arange(m_cnt, dtype=torch.float32) + order_unit_precision * step

            m_end_idx = m_str_idx

        return particle[1:]

    def initialize_chromosome(self):
        if self.sjf_particle is None:
            # shape: (self.n_particles, stage_cnt * jobs_cnt)
            # x = x_min + (x_max - x_min) * N(0, 1)
            return torch.cat([torch.rand(self.num_of_chromosome, self.env.job_cnt) * mc
                              for mc in self.env.machine_cnt_list],
                             dim=1)
        else:
            return torch.cat((self.sjf_particle[None], torch.cat(
                [torch.rand(self.num_of_chromosome - 1, self.env.job_cnt) * mc for mc in
                 self.env.machine_cnt_list], dim=1))).to(dtype=torch.float64)

    @staticmethod
    def softmax(a):
        return torch.softmax(a, dim=0)

    def crossover(self, parents, parents_idx):
        if self.crossover_method == 'OX1':
            division_point = torch.randint(self.num_of_gene, size=(1,))
            offspring = parents[0][:division_point]
            for gene in parents[1]:
                if gene not in offspring:
                    offspring = torch.cat((offspring, gene[None]), dim=0)
        elif self.crossover_method == 'PBX':
            set_of_positions = torch.randint(self.num_of_gene, size=(int(self.num_of_gene * self.crossover_ratio),))
            offspring = torch.zeros_like(parents[0])
            offspring[set_of_positions] = parents[0][set_of_positions]

            gene_idx = -1

            for idx in range(self.num_of_gene):
                if idx in set_of_positions:
                    continue

                while True:
                    gene_idx += 1

                    if parents[1][gene_idx] not in offspring:
                        offspring[idx] = parents[1][gene_idx]
                        break
        elif self.crossover_method == 'FFSP':
            selected_integer = torch.randint(0, 2, size=(self.env.job_cnt * self.env.stage_cnt,))
            selected_fraction = torch.randint(0, 2, size=(self.env.job_cnt * self.env.stage_cnt,))
            offspring_int = parents.t()[torch.arange(self.env.job_cnt * self.env.stage_cnt), selected_integer]
            offspring_frac = parents.t()[torch.arange(self.env.job_cnt * self.env.stage_cnt), selected_fraction]
            offspring_int = offspring_int.floor()
            offspring_frac = offspring_frac - offspring_frac.floor()
            offspring = offspring_int + offspring_frac

        return offspring

    def mutation(self, offspring):
        if self.mutation_mode is None:
            mode = torch.randint(4, size=(1,))
        else:
            mode = self.mutation_mode

        indices = torch.randint(self.num_of_gene, size=(2,))

        # exchange
        if mode == 0:
            offspring[indices] = offspring[indices.flip(dims=(0,))]
        # inverse
        elif mode == 1:
            low = min(indices)
            high = max(indices)
            offspring[low:high] = torch.flip(offspring[low:high], dims=(0,))
        # insert
        elif mode == 2:
            if indices[0] < indices[1]:
                offspring = torch.cat((offspring[:indices[0]], offspring[indices[0] + 1:indices[1]],
                                       offspring[indices[0]][None], offspring[indices[1]:]))
            elif indices[0] > indices[1]:
                offspring = torch.cat((offspring[:indices[1]], offspring[indices[0]][None],
                                       offspring[indices[1]:indices[0]], offspring[indices[0] + 1:]))
        # insert
        elif mode == 3:
            low = 0
            high = 0
            for mc in self.env.machine_cnt_list:
                high += mc
                index = torch.randint(low=low, high=high, size=(1,))
                offspring[index[0]] = torch.rand(1) * mc
                low = high

        return offspring

    def generate_offspring(self, chromosomes, fitness):
        if self.selection_method == 'roulette_wheel':
            p = fitness / sum(fitness)
        elif self.selection_method == 'softmax':
            p = self.softmax(fitness)

        new_generation = torch.stack([chromosomes[fitness.argmax()]])

        for _ in range(self.num_of_chromosome - 1):
            parents_idx = torch.multinomial(p, 2)
            parents = chromosomes[parents_idx]
            offspring = self.crossover(parents, parents_idx)

            if torch.rand(1) < self.mutation_rate:
                offspring = self.mutation(offspring)

            new_generation = torch.cat((new_generation, offspring[None]), dim=0)

        return new_generation

    def get_fitness(self, chromosomes):
        self.env.reset()
        cmax = self.env.step(chromosomes)
        return -cmax, cmax

    def solve(self):
        self.env.load_problems(self.num_of_chromosome, same_problem=True)

        chromosomes = self.initialize_chromosome()

        fitness, cmax = self.get_fitness(chromosomes)

        best_fitness = fitness.max()
        best_solution = chromosomes[fitness.argmax()]
        best_cmax = cmax[fitness.argmax()]

        self.logger.info('init. best cmax: {:.15f}'.format(best_cmax))

        for e in range(self.epochs):
            if self.target_fitness and self.target_fitness <= best_fitness and e > 99:
                break

            offspring = self.generate_offspring(chromosomes, fitness)

            offspring_fitness, offspring_cmax = self.get_fitness(offspring)

            if best_fitness < offspring_fitness.max():
                best_fitness = offspring_fitness.max()
                best_solution = offspring[offspring_fitness.argmax()]
                best_cmax = offspring_cmax[offspring_fitness.argmax()]
                self.mutation_rate = self.init_mutation_rate
            else:
                self.mutation_rate = min(self.mutation_rate * 2, self.max_mutation_rate)

            population = torch.cat((chromosomes, offspring))
            population_fitness = torch.cat((fitness, offspring_fitness))

            if self.selection_method == 'roulette_wheel':
                p = fitness / sum(fitness)
            elif self.selection_method == 'softmax':
                p = self.softmax(fitness)

            chosen_idx = torch.multinomial(p, self.num_of_chromosome - 1, replacement=False)
            chosen_idx = torch.cat((chosen_idx, population_fitness.argmax()[None]))
            chromosomes = population[chosen_idx]
            fitness = population_fitness[chosen_idx]

            if e % 50 != 0:
                continue

            self.logger.info('{:3} epochs. best Cmax: {:.15f} mutation rate: {}'.format(e,
                                                                                        best_cmax,
                                                                                        self.mutation_rate))

        self.logger.info('best score: {:.15f}'.format(best_cmax))
        self.logger.info('best solution index: {}'.format(best_solution))

        # plot best image
        self.get_fitness(best_solution.expand(self.num_of_chromosome, *best_solution.size()))
        filename = os.path.join(self.result_folder, 'Figure_GA_{}_{}'.format(self.env.file['problem_index'], best_cmax))
        self.env.render(save=True, filename=filename)

        return best_cmax
