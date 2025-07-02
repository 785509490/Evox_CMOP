import math
from typing import Callable, Optional

import torch
from evox.core import Algorithm, Mutable, vmap
from evox.operators.crossover import simulated_binary_half, DE_crossover
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp


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

def lhs_population(pop_size, dim, lb, ub, device="cuda"):
    # step = 1/pop_size 的分段，先生成中心点
    seg = torch.linspace(0, 1, pop_size + 1, device=device)[:-1]
    center = seg + torch.rand(pop_size, device=device) / pop_size   # (pop_size,)

    # 对每个维度独立打乱
    pop = torch.stack([center[torch.randperm(pop_size)] for _ in range(dim)], dim=1)

    # 映射到 [lb, ub]
    return lb + pop * (ub - lb)

class GMPEA2(Algorithm):
    """
    TensorMOEA/D

    This is a tensorized implementation of the original MOEA/D algorithm, which incorporates GPU acceleration
    for improved computational performance in solving multi-objective optimization problems.

    :references:
        [1] Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition,"
            IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712-731, 2007. Available:
            https://ieeexplore.ieee.org/document/4358754

        [2] Z. Liang, H. Li, N. Yu, K. Sun, and R. Cheng, "Bridging Evolutionary Multiobjective Optimization and
            GPU Acceleration via Tensorization," IEEE Transactions on Evolutionary Computation, 2025. Available:
            https://ieeexplore.ieee.org/document/10944658

    :note: This implementation differs from the original MOEA/D algorithm by incorporating tensorization for
            GPU acceleration, significantly improving performance for large-scale optimization tasks.
    """

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        aggregate_op=("modified_tchebycheff", "modified_tchebycheff"),
        max_gen: int = 100,
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
        self.max_gen = max_gen
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
        self.n_neighbor2 = int(math.ceil(self.pop_size / 50))

        length = ub - lb
        #population = torch.rand(self.pop_size, self.dim, device=device)
        population = lhs_population(self.pop_size, self.dim, lb, ub, device=device)

        #population2 = torch.rand(self.pop_size, self.dim, device=device)
        population2 = lhs_population(self.pop_size, self.dim, lb, ub, device=device)

        neighbors = torch.cdist(w, w)
        self.neighbors = torch.argsort(neighbors, dim=1, stable=True)[:, : self.n_neighbor]
        self.neighbors2 = torch.argsort(neighbors, dim=1, stable=True)[:, : self.n_neighbor2]
        self.w = w
        self.gen = 0
        self.pop = Mutable(population)
        self.pop2 = Mutable(population2)
        self.fit = Mutable(torch.full((self.pop_size, self.n_objs), torch.inf, device=device))
        self.fit2 = Mutable(torch.full((self.pop_size, self.n_objs), torch.inf, device=device))
        self.z = Mutable(torch.zeros((self.n_objs,), device=device))
        self.z2 = Mutable(torch.zeros((self.n_objs,), device=device))
        self.z_max = Mutable(torch.zeros((self.n_objs,), device=device))
        self.z_max2 = Mutable(torch.zeros((self.n_objs,), device=device))
        if n_objs > 2:
            aggregate_op = ("pbi", "pbi")
        self.aggregate_func1 = self.get_aggregation_function(aggregate_op[0])
        self.aggregate_func2 = self.get_aggregation_function(aggregate_op[1])
        self.cons = None
        self.cons2 = None

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
        combined_tensor = torch.cat([self.pop, self.pop2], dim=0)
        fitness = self.evaluate(combined_tensor)
        if isinstance(fitness, tuple):
            fit = fitness[0]
            cons = fitness[1]
            row = self.pop_size
            self.fit = fit[:row]
            self.fit2 = fit[row:]
            self.cons = cons[:row]
            self.cons2 = cons[row:]
        else:
            self.fit = fitness
        self.z = torch.min(self.fit, dim=0)[0]
        self.z2 = torch.min(self.fit2, dim=0)[0]

    def step(self):
        """Perform the optimization step of the workflow."""
        self.z_max = torch.max(self.fit, dim=0).values
        self.z_max2 = torch.max(self.fit2, dim=0).values
        parent = shuffle_rows(self.neighbors)
        parent2 = shuffle_rows(self.neighbors2)
        if self.crossover is DE_crossover:
            CR = torch.ones((self.pop_size, self.dim))
            F = torch.ones((self.pop_size, self.dim))*0.5
            selected_p = torch.cat([self.pop[parent[:, 0]], self.pop[parent[:, 1]], self.pop[parent[:, 2]]], dim=0)
            selected_p2 = torch.cat([self.pop2[parent2[:, 0]], self.pop2[parent2[:, 1]], self.pop2[parent2[:, 2]]], dim=0)
            crossovered = self.crossover(selected_p[:self.pop_size], selected_p[self.pop_size : self.pop_size*2], selected_p[self.pop_size*2 : ], CR, F)
            crossovered2 = self.crossover(selected_p2[:self.pop_size], selected_p2[self.pop_size : self.pop_size*2], selected_p2[self.pop_size*2 : ], CR, F)
        else:
            selected_p = torch.cat([self.pop[parent[:, 0]], self.pop[parent[:, 1]]], dim=0)
            selected_p2 = torch.cat([self.pop2[parent2[:, 0]], self.pop2[parent2[:, 1]]], dim=0)
            crossovered = self.crossover(selected_p)
            crossovered2 = self.crossover(selected_p2)
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        offspring2 = self.mutation(crossovered2, self.lb, self.ub)
        offspring2 = clamp(offspring2, self.lb, self.ub)
        mergeOff = torch.cat([offspring, offspring2], dim=0)
        offT_fit = self.evaluate(mergeOff)
        offT_cons = offT_fit[1]
        offT_fit = offT_fit[0]

        off_cons1 = offT_cons[:self.pop_size]
        off_fit1 = offT_fit[:self.pop_size]
        cv_off1 = torch.sum(torch.clamp(off_cons1, min=0), dim=1, keepdim=True)

        off_cons2 = offT_cons[self.pop_size:]
        off_fit2 = offT_fit[self.pop_size:]
        cv_off2 = torch.sum(torch.clamp(off_cons2, min=0), dim=1, keepdim=True)

        self.z = torch.min(self.z, torch.min(off_fit1, dim=0)[0])
        self.z2 = torch.min(self.z2, torch.min(off_fit2, dim=0)[0])

        sub_pop_indices = torch.arange(0, self.pop_size, device=self.pop.device)
        update_mask = torch.zeros((self.pop_size,), dtype=torch.bool, device=self.pop.device)


        def body4(ind_obj, ind_obj2, cv_new, cv_new2):
            if self.aggregate_func1 == tchebycheff_norm:
                g_new2 = self.aggregate_func1(ind_obj2, self.w, self.z2, self.z_max2)
                g_new = self.aggregate_func1(ind_obj, self.w, self.z, self.z_max)
            else:
                g_new2 = self.aggregate_func1(ind_obj2, self.w, self.z2)
                g_new = self.aggregate_func1(ind_obj, self.w, self.z)

            off_mask2 = (g_new > g_new2).unsqueeze(-1)
            g1 = g_new.squeeze()  # (60,)
            g2 = g_new2.squeeze()
            cv1 = cv_new.squeeze()
            cv2 = cv_new2.squeeze()

            off_mask1 = (((g2 > g1) & (cv2 == cv1)) | (cv2 > cv1)).unsqueeze(-1)  # (60,1)
            return off_mask1, off_mask2

        def body3(ind_p, ind_p2, ind_obj, ind_obj2, cv_new, cv_new2):
            if self.aggregate_func1 == tchebycheff_norm:
                g_old2 = self.aggregate_func1(self.fit2[ind_p2], self.w[ind_p2], self.z2, self.z_max2)
                g_new2 = self.aggregate_func1(ind_obj2, self.w[ind_p2], self.z2, self.z_max2)
                g_old = self.aggregate_func1(self.fit[ind_p], self.w[ind_p], self.z, self.z_max)
                g_new = self.aggregate_func1(ind_obj, self.w[ind_p], self.z, self.z_max)
            else:
                g_old2 = self.aggregate_func1(self.fit2[ind_p2], self.w[ind_p2], self.z2)
                g_new2 = self.aggregate_func1(ind_obj2, self.w[ind_p2], self.z2)
                g_old = self.aggregate_func1(self.fit[ind_p], self.w[ind_p], self.z)
                g_new = self.aggregate_func1(ind_obj, self.w[ind_p], self.z)
            cv_old = torch.sum(torch.clamp(self.cons[ind_p], min=0), dim=1, keepdim=True).squeeze()

            temp_mask2 = update_mask.clone()
            temp_mask2 = torch.scatter(temp_mask2, 0, ind_p2, g_old2 > g_new2)
            temp_mask = update_mask.clone()
            temp_mask = torch.scatter(temp_mask, 0, ind_p,
                                          (((g_old > g_new) & (cv_old == cv_new)) | (cv_new < cv_old)))
            return torch.where(temp_mask, -1, sub_pop_indices.clone()), torch.where(temp_mask2, -1, sub_pop_indices.clone())

        offMask, offMask2 = body4(off_fit1, off_fit2, cv_off1, cv_off2)
        off_fit1 = torch.where(offMask, off_fit1, off_fit2)
        cv_off1 = torch.where(offMask, cv_off1, cv_off2)
        offspring = torch.where(offMask, offspring, offspring2)

        off_fit2 = torch.where(offMask, off_fit2, off_fit1)
        cv_off2 = torch.where(offMask, cv_off2, cv_off1)
        offspring2 = torch.where(offMask, offspring2, offspring)

        cv_new1 = cv_off1[self.neighbors].squeeze()
        cv_new2 = cv_off2[self.neighbors2].squeeze()

        replace_indices, replace_indices2 = vmap(body3, in_dims=(0, 0, 0, 0, 0, 0))(self.neighbors, self.neighbors2, off_fit1, off_fit2, cv_new1, cv_new2)


        def update_population(sub_indices, population, pop_obj, pop_cons, w_ind, ispop2):
            if ispop2:
                f = torch.where(sub_indices[:, None] == -1, off_fit2, pop_obj)
                x = torch.where(sub_indices[:, None] == -1, offspring2, population)
                cons = torch.where(sub_indices[:, None] == -1, off_cons2, pop_cons)
                if self.aggregate_func1 == tchebycheff_norm:
                    idx = torch.argmin(self.aggregate_func2(f, w_ind[None, :], self.z2, self.z_max2))
                else:
                    idx = torch.argmin(self.aggregate_func2(f, w_ind[None, :], self.z2))
            else:
                f = torch.where(sub_indices[:, None] == -1, off_fit1, pop_obj)
                x = torch.where(sub_indices[:, None] == -1, offspring, population)
                cons = torch.where(sub_indices[:, None] == -1, off_cons1, pop_cons)
                cvt = torch.sum(torch.clamp(cons, min=0), dim=1)
                min_value = cvt.min()
                min_mask = (cvt == min_value)
                count_true = min_mask.sum()
                sub_f = torch.where(min_mask[:, None], f, torch.tensor(1000, device=f.device))
                if self.aggregate_func1 == tchebycheff_norm:
                    idx = torch.where(count_true > 1, torch.argmin(self.aggregate_func2(sub_f, w_ind[None, :], self.z, self.z_max)), torch.argmin(cvt))
                else:
                    idx = torch.where(count_true > 1, torch.argmin(self.aggregate_func2(sub_f, w_ind[None, :], self.z)), torch.argmin(cvt))
            return x[idx], f[idx], cons[idx]

        self.pop, self.fit, self.cons = vmap(update_population, in_dims=(1, 0, 0, 0, 0, None))(replace_indices, self.pop, self.fit, self.cons, self.w, 0)
        self.pop2, self.fit2, self.cons2 = vmap(update_population, in_dims=(1, 0, 0, 0, 0, None))(replace_indices2, self.pop2, self.fit2, self.cons2, self.w, 1)
        self.gen += 1
        # if self.gen >= self.max_gen:
        #     g = self.aggregate_func1(self.fit, self.w, self.z)
        #     mask = (g < 1.1).squeeze(-1)
        #     self.fit = self.fit[mask]
        #     self.pop = self.pop[mask]
        #     self.cons = self.cons[mask]

