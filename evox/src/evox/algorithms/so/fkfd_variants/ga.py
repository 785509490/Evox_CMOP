from typing import Literal
import numpy as np
import copy
import torch
from evox.core import Algorithm, Mutable
from evox.fkfd.Convert import SetParameter
from typing import List

class GA(Algorithm):
    """
    GA algorithm for FKFD

    """

    def __init__(
            self,
            scenario_idx: int,
            pop_size: int,
            device: torch.device | None = None,
    ):
        """
        Initialize the DE algorithm with the given parameters.

        :param pop_size: The size of the population.
        :param device: The device to use for tensor computations. Defaults to None.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device

        # Validate input parameters
        assert pop_size >= 1

        # Initialize parameters
        self.pop_size = pop_size
        self.Prob = SetParameter(scenario_idx)
        self.dim = len(self.Prob.data_len)


        # Move bounds to the specified device and add batch dimension
        self.lb = torch.ones(self.pop_size, self.dim)
        self.ub = self.Prob.data_len.repeat(self.pop_size, 1)

        # Initialize population uniformly within bounds
        # pop = torch.rand(self.pop_size, self.dim, device=device)
        # pop = torch.mul(pop, self.ub)
        # pop = torch.ceil(pop)
        num_target = len(self.Prob.Target.order)
        # 深拷贝Prob以避免修改原数据
        prob = copy.deepcopy(self.Prob)
        population = []
        pop = []
        for _ in range(self.pop_size):
            # 初始化个体
            decs = np.zeros(num_target)
            # 为每个目标选择拦截方案
            for j in range(num_target):
                target_idx = prob.targetInd[j]
                schemes = prob.data[target_idx - 1]
                if len(schemes) > 0:
                    # 选择耗时最短的方案（简化版，实际需实现轮盘赌）
                    #waste_time = schemes[:, 3] - schemes[:, 0]
                    hit_p = -schemes[:,-1]
                    selected_idx = roulette_selection(hit_p, 1)
                    # 记录方案信息
                    decs[j] = int(selected_idx[0])
            pop.append(decs)
        pop = torch.stack([torch.from_numpy(arr) for arr in pop])

        # Mutable attributes to store population and fitness
        self.pop = Mutable(pop)
        self.fit = Mutable(torch.full((self.pop_size,), torch.inf, device=device))

    def step(self):
        device = self.pop.device

        # 轮盘赌法选择
        safe_fitness = self.fit + 1e-10
        cumulative = torch.cumsum(safe_fitness, dim=0)  # 形状: (M,)
        total = cumulative[-1]  # 总和
        cumulative_probs = cumulative / total
        targets = torch.rand(self.pop_size, device=device)

        selected_indices = torch.searchsorted(cumulative_probs,
                                              targets,
                                              right=True)
        selected_indices = torch.clamp(selected_indices,
                                       min=self.pop.size(0) - 1)
        select_pop = self.pop[selected_indices]

        # 交叉变异
        N, n_target = select_pop.shape
        p2_indices = torch.randint(0, N, (N,), device=device)
        p2_decs = select_pop[p2_indices]  # (N, n_target)

        cross_mask = torch.rand(N, n_target, device=device) < 20 / n_target
        offspring = torch.where(cross_mask, p2_decs, select_pop)

        mutate_mask = torch.rand(N, n_target, device=device) < 10 / n_target
        rule_counts = self.ub
        random_picks = (torch.rand((N, n_target), device=device) * rule_counts).long()
        random_picks = random_picks.to(dtype=offspring.dtype)
        offspring = torch.where(mutate_mask, random_picks, offspring)

        new_fitness = self.evaluate(offspring)
        # 选择
        compare_mask = (new_fitness < self.fit)
        compare_mask = compare_mask.unsqueeze(-1)
        self.pop = torch.where(compare_mask, offspring, self.pop)
        self.fit = torch.where(compare_mask.squeeze(), new_fitness, self.fit)

    def init_step(self):
        """
        Perform the initial evaluation of the population's fitness and proceed to the first optimization step.
        This method evaluates the fitness of the initial population and then calls the `step` method to perform the first optimization iteration.
        """
        self.fit = self.evaluate(self.pop)
        self.step()

def roulette_selection(z: List[float], N: int) -> List[int]:
    """
    轮盘赌选择算法
    :param z: 适应度值列表（需要是非负数）
    :param N: 需要选择的个体数量
    :return: 被选中的个体索引列表（0-based）
    """
    # 计算累积概率
    cumulative = np.cumsum(z)
    total = cumulative[-1]  # 获取最后一个元素的累计值（总和）

    # 归一化累积概率
    cumulative = cumulative / total

    selected_idx = []
    for _ in range(N):
        # 生成随机目标值
        target = np.random.rand()

        # 查找第一个大于等于目标的索引
        # 使用二分查找优化性能（替代顺序查找）
        index = np.searchsorted(cumulative, target, side='right')

        # 处理边界情况
        index = min(index, len(z) - 1)
        selected_idx.append(index)

    return selected_idx