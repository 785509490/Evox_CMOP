from typing import Callable, Optional

import torch
import time
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary, DE_crossover
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import nd_environmental_selection_cons, tournament_selection_multifit, nd_environmental_selection, dominate_relation_cons, dominate_relation
from evox.utils import clamp


class NSGA2(Algorithm):
    """
    A tensorized implementation of the Non-dominated Sorting Genetic Algorithm II (NSGA-II)
    for multi-objective optimization problems.

    :references:
        [1] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II,"
            IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, 2002.
            Available: https://ieeexplore.ieee.org/document/996017

        [2] Z. Liang, H. Li, N. Yu, K. Sun, and R. Cheng, "Bridging Evolutionary Multiobjective Optimization and
            GPU Acceleration via Tensorization," IEEE Transactions on Evolutionary Computation, 2025. Available:
            https://ieeexplore.ieee.org/document/10944658
    """

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        max_gen: int = 100,
        selection_op: Optional[Callable] = None,
        mutation_op: Optional[Callable] = None,
        crossover_op: Optional[Callable] = None,
        device: torch.device | None = None,
    ):
        """Initializes the NSGA-II algorithm.

        :param pop_size: The size of the population.
        :param n_objs: The number of objective functions in the optimization problem.
        :param lb: The lower bounds for the decision variables (1D tensor).
        :param ub: The upper bounds for the decision variables (1D tensor).
        :param selection_op: The selection operation for evolutionary strategy (optional).
        :param mutation_op: The mutation operation, defaults to `polynomial_mutation` if not provided (optional).
        :param crossover_op: The crossover operation, defaults to `simulated_binary` if not provided (optional).
        :param device: The device on which computations should run (optional). Defaults to PyTorch's default device.
        """

        super().__init__()
        self.pop_size = pop_size
        self.n_objs = n_objs
        if device is None:
            device = torch.get_default_device()
        # check
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1
        assert lb.dtype == ub.dtype and lb.device == ub.device
        self.dim = lb.shape[0]
        # write to self
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op
        self.crossover_op = crossover_op

        if self.selection is None:
            self.selection = tournament_selection_multifit
        if self.mutation is None:
            self.mutation = polynomial_mutation
        if self.crossover is None:
            self.crossover = simulated_binary



        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb

        self.pop = Mutable(population)
        self.fit = Mutable(torch.empty((self.pop_size, self.n_objs), device=device).fill_(torch.inf))
        self.rank = Mutable(torch.empty(self.pop_size, device=device).fill_(torch.inf))
        self.dis = Mutable(torch.empty(self.pop_size, device=device).fill_(-torch.inf))
        self.cons = None

    def init_step(self):
        """
        Perform the initialization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        fitness = self.evaluate(self.pop)
        if isinstance(fitness, tuple):
            self.fit = fitness[0]
            self.cons = fitness[1]
            _, _, self.rank, self.dis, self.cons = nd_environmental_selection_cons(self.pop, self.fit, self.cons, self.pop_size)
        else:
            self.fit = fitness
            _, _, self.rank, self.dis = nd_environmental_selection(self.pop, self.fit, self.pop_size)

    def step(self):
        """Perform the optimization step of the workflow."""
        if self.crossover is DE_crossover:
            CR = torch.ones((self.pop_size, self.dim))
            F = torch.ones((self.pop_size, self.dim))*0.5
            mating_pool = self.selection(self.pop_size*2, [-self.dis, self.rank])
            crossovered = self.crossover(self.pop, self.pop[mating_pool[:self.pop_size]], self.pop[mating_pool[self.pop_size:]], CR, F)
        else:
            mating_pool = self.selection(self.pop_size, [-self.dis, self.rank])
            crossovered = self.crossover(self.pop[mating_pool])
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)

        merge_cons = None
        iscons = False


        if isinstance(off_fit, tuple):
            iscons = True
            off_cons = off_fit[1]
            off_fit = off_fit[0]
            merge_cons = torch.cat([self.cons, off_cons], dim=0)
        merge_pop = torch.cat([self.pop, offspring], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)


        if iscons:
            self.pop, self.fit, self.rank, self.dis, self.cons = nd_environmental_selection_cons(merge_pop, merge_fit, merge_cons, self.pop_size)
        else:
            self.pop, self.fit, self.rank, self.dis = nd_environmental_selection(merge_pop, merge_fit,self.pop_size)





