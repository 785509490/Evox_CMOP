import math
from typing import Callable, Optional

import torch
from evox.core import Algorithm, Mutable, vmap
from evox.operators.crossover import simulated_binary_half, DE_crossover
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp
from evox.operators.selection import nd_environmental_selection_cons


def pbi(f, w, z):
    norm_w = torch.norm(w, dim=1)
    f = f - z
    d1 = torch.sum(f * w, dim=1) / norm_w
    d2 = torch.norm(f - (d1[:, None] * w / norm_w[:, None]), dim=1)
    return d1 + 5 * d2


def tchebycheff(f, w, z):
    return torch.max(torch.abs(f - z) * w, dim=1)[0]


def tchebycheff_norm(f, w, z, z_max):
    return torch.max(torch.abs(f - z) / (z_max - z) * w, dim=1)[0]


def modified_tchebycheff(f, w, z):
    return torch.max(torch.abs(f - z) / w, dim=1)[0]


def weighted_sum(f, w):
    return torch.sum(f * w, dim=1)


def shuffle_rows(matrix: torch.Tensor) -> torch.Tensor:
    """
    Shuffle each row of the given matrix independently without using a for loop.

    Args:
        matrix (torch.Tensor): A 2D tensor.

    Returns:
        torch.Tensor: A new tensor with each row shuffled differently.
    """
    rows, cols = matrix.size()

    permutations = torch.argsort(torch.rand(rows, cols, device=matrix.device), dim=1)
    return matrix.gather(1, permutations)

class PPS(Algorithm):

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        aggregate_op=("tchebycheff", "tchebycheff"),
        max_gen:int=100,
        selection_op: Optional[Callable] = None,
        mutation_op: Optional[Callable] = None,
        crossover_op: Optional[Callable] = None,
        device: torch.device | None = None,
    ):
        """Initializes the TensorMOEA/D algorithm.

        :param pop_size: The size of the population.
        :param n_objs: The number of objective functions in the optimization problem.
        :param lb: The lower bounds for the decision variables (1D tensor).
        :param ub: The upper bounds for the decision variables (1D tensor).
        :param aggregate_op: The aggregation function to use for the algorithm (optional).
        :param selection_op: The selection operation for evolutionary strategy (optional).
        :param mutation_op: The mutation operation, defaults to `polynomial_mutation` if not provided (optional).
        :param crossover_op: The crossover operation, defaults to `simulated_binary` if not provided (optional).
        :param device: The device on which computations should run (optional). Defaults to PyTorch's default device.
        """

        super().__init__()
        self.pop_size = pop_size
        self.n_objs = n_objs
        device = torch.get_default_device() if device is None else device
        # check
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1
        assert lb.dtype == ub.dtype and lb.device == ub.device
        self.dim = lb.shape[0]
        # write to self
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)
        self.gen = 0
        self.last_gen = 20
        self.max_gen = max_gen
        self.Tc  = 0.9 * self.max_gen
        self.change_threshold = 1e-1
        self.search_stage = 1
        self.max_change = 1
        self.epsilon_k = 0
        self.epsilon_0 = 0
        self.cp = 2
        self.alpha = 0.95
        self.tao = 0.05

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

        if self.mutation is None:
            self.mutation = polynomial_mutation
        if self.crossover is None:
            self.crossover = simulated_binary_half

        w, _ = uniform_sampling(self.pop_size, self.n_objs)
        w = w.to(device=device)

        self.pop_size = w.size(0)
        assert self.pop_size > 10, "Population size must be greater than 10. Please reset the population size."
        self.n_neighbor = int(math.ceil(self.pop_size / 10))

        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb

        neighbors = torch.cdist(w, w)
        self.neighbors = torch.argsort(neighbors, dim=1, stable=True)[:, : self.n_neighbor]
        self.w = w

        self.pop = Mutable(population)

        self.fit = Mutable(torch.full((self.pop_size, self.n_objs), torch.inf, device=device))
        self.cons = None
        self.z = Mutable(torch.zeros((self.n_objs,), device=device))
        self.ideal_points = Mutable(torch.zeros((1, self.n_objs), device=device))
        self.nadir_points = Mutable(torch.zeros((1, self.n_objs), device=device))
        self.aggregate_func1 = self.get_aggregation_function(aggregate_op[0])
        self.aggregate_func2 = self.get_aggregation_function(aggregate_op[1])

        self.archpop = self.pop
        self.archfit = self.fit
        self.archcons = self.cons

    def get_aggregation_function(self, name: str) -> Callable:
        aggregation_functions = {
            "pbi": pbi,
            "tchebycheff": tchebycheff,
            "tchebycheff_norm": tchebycheff_norm,
            "modified_tchebycheff": modified_tchebycheff,
            "weighted_sum": weighted_sum,
        }
        if name not in aggregation_functions:
            raise ValueError(f"Unsupported function: {name}")
        return aggregation_functions[name]

    def init_step(self):
        """
        Perform the initialization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        fitness = self.evaluate(self.pop)
        if isinstance(fitness, tuple):
            self.fit = fitness[0]
            self.cons = fitness[1]
            self.archpop = self.pop
            self.archfit = self.fit
            self.archcons = self.cons
        else:
            self.fit = fitness
        self.z = torch.min(self.fit, dim=0)[0]

    @staticmethod
    def calc_maxchange(ideal_points, nadir_points, gen, last_gen):
        delta_value = 1e-6 * torch.ones(1, ideal_points.size(1), device=ideal_points.device)
        rz = torch.abs((ideal_points[gen, :] - ideal_points[gen - last_gen + 1, :]) / torch.max(ideal_points[gen - last_gen + 1, :], delta_value))
        nrz = torch.abs((nadir_points[gen, :] - nadir_points[gen - last_gen + 1, :]) / torch.max(nadir_points[gen - last_gen + 1, :], delta_value))
        return torch.max(torch.cat((rz, nrz), dim=0))

    @staticmethod
    def update_epsilon(tao, epsilon_k, epsilon_0, rf, alpha, gen, Tc, cp):
        if rf < alpha:
            return (1 - tao) * epsilon_k
        else:
            return epsilon_0 * ((1 - (gen / Tc)) ** cp)

    def step(self):
        """Perform the optimization step of the workflow."""
        parent = shuffle_rows(self.neighbors)
        if self.crossover is DE_crossover:
            CR = torch.ones((self.pop_size, self.dim))
            F = torch.ones((self.pop_size, self.dim))*0.5
            selected_p = torch.cat([self.pop[parent[:, 0]], self.pop[parent[:, 1]], self.pop[parent[:, 2]]], dim=0)
            crossovered = self.crossover(selected_p[:self.pop_size], selected_p[self.pop_size : self.pop_size*2], selected_p[self.pop_size*2 : ], CR, F)
        else:
            selected_p = torch.cat([self.pop[parent[:, 0]], self.pop[parent[:, 1]]], dim=0)
            crossovered = self.crossover(selected_p)
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)
        #if isinstance(off_fit, tuple):
        off_cons = off_fit[1]
        off_fit = off_fit[0]
        cv = torch.sum(torch.clamp(self.cons, min=0), dim=1, keepdim=True)
        cv_off = torch.sum(torch.clamp(off_cons, min=0), dim=1, keepdim=True)
        rf = (cv <= 1e-6).sum().item() / self.pop_size
        temp  = torch.cat([self.pop, self.fit, cv], dim=1)
        self.z = torch.min(self.z, torch.min(off_fit, dim=0)[0])
        if self.gen == 0:
            self.ideal_points[0,:] = self.z
        else:
            z = self.z.unsqueeze(0)
            self.ideal_points = torch.cat([self.ideal_points, self.z.unsqueeze(0)], dim=0)
        D = self.pop.size(1)
        M = self.fit.size(1)
        if self.gen == 0:
            self.nadir_points[0, :] = torch.max(temp[:, D:D + M], dim=0)[0]
        else:
            b = torch.max(temp[:, D:D + M], dim=0)[0].unsqueeze(0)
            self.nadir_points = torch.cat([self.nadir_points, b], dim=0)


        if self.gen >= self.last_gen:
            self.max_change = self.calc_maxchange(self.ideal_points, self.nadir_points, self.gen, self.last_gen)
        # The value of e(k) and the search strategy are set.
        if self.gen < self.Tc:
            if self.max_change <= self.change_threshold and self.search_stage == 1:
                self.search_stage = -1
                self.epsilon_0 = temp[:, -1].max().item()
                self.epsilon_k = self.epsilon_0
            if self.search_stage == -1:
                self.epsilon_k = self.update_epsilon(self.tao, self.epsilon_k, self.epsilon_0, rf, self.alpha, self.gen, self.Tc, self.cp)
        else:
            self.epsilon_k = 0


        sub_pop_indices = torch.arange(0, self.pop_size, device=self.pop.device)
        update_mask = torch.zeros((self.pop_size,), dtype=torch.bool, device=self.pop.device)

        def body(ind_p, ind_obj, cv_new, search_stage):
            g_old = self.aggregate_func1(self.fit[ind_p], self.w[ind_p], self.z)
            g_new = self.aggregate_func1(ind_obj, self.w[ind_p], self.z)
            cv_old = torch.sum(torch.clamp(self.cons[ind_p], min=0), dim=1, keepdim=True).squeeze()
            temp_mask = update_mask.clone()
            if search_stage == 1:
                temp_mask = torch.scatter(temp_mask, 0, ind_p, g_old > g_new)
            else:
                temp_mask = torch.scatter(temp_mask, 0, ind_p,  ((g_old > g_new) & (((cv_old <= self.epsilon_k) & (cv_new <= self.epsilon_k)) | (cv_old == cv_new)) | (cv_new < cv_old) ))
            return torch.where(temp_mask, -1, sub_pop_indices.clone())

        search_stage_tensor = self.search_stage
        cv_new = cv_off[self.neighbors].squeeze()
        replace_indices = vmap(body, in_dims=(0, 0, 0, None))(self.neighbors, off_fit, cv_new, search_stage_tensor)

        def update_population(sub_indices, population, pop_obj, pop_cons, w_ind):
            f = torch.where(sub_indices[:, None] == -1, off_fit, pop_obj)
            x = torch.where(sub_indices[:, None] == -1, offspring, population)
            cons = torch.where(sub_indices[:, None] == -1, off_cons, pop_cons)

            if self.search_stage == 1:
                idx = torch.argmin(self.aggregate_func2(f, w_ind[None, :], self.z))
            else:
                cvt = torch.sum(torch.clamp(cons, min=0), dim=1)
                min_value = cvt.min()
                min_mask = (cvt == min_value)
                count_true = min_mask.sum()
                sub_f = torch.where(min_mask[:, None], f, torch.tensor(1000, device=f.device))
                idx = torch.where(count_true > 1, torch.argmin(self.aggregate_func2(sub_f, w_ind[None, :], self.z)), torch.argmin(cvt))

            return x[idx], f[idx], cons[idx]

        self.pop, self.fit, self.cons = vmap(update_population, in_dims=(1, 0, 0, 0, 0))(replace_indices, self.pop, self.fit, self.cons, self.w)
        self.gen += 1
        merge_pop = torch.cat([self.archpop, self.pop], dim=0)
        merge_fit = torch.cat([self.archfit, self.fit], dim=0)
        merge_cons = torch.cat([self.archcons, self.cons], dim=0)
        self.archpop, self.archfit, _, _, self.archcons = nd_environmental_selection_cons(merge_pop, merge_fit, merge_cons, self.pop_size)
        if self.gen >= self.max_gen:
            self.pop, self.fit, _, _, self.cons = nd_environmental_selection_cons(merge_pop, merge_fit, merge_cons, self.pop_size)
