import numpy as np
import matplotlib.pyplot as plt
from Convert import SetParameter
from utilsmat import load_mat_file
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple
import copy
import random
from newCalobj import cal_obj


# 安装依赖（PyCharm中可忽略，确保已安装）
# pip install numpy matplotlib
@dataclass
class Individual:
    """个体数据结构（对应MATLAB的empty_pop结构）"""
    decs: np.ndarray = None  # 方案编号
    decs1: List[List] = None  # 方案详细信息
    decs2: List[List] = None  # 接力制导信息
    objs: np.ndarray = None  # 目标函数值
    every_objs: np.ndarray = None  # 各子目标值
    radar: np.ndarray = None  # 雷达通道占用
    missile: np.ndarray = None  # 剩余弹量
    TargetUsedRadar: List = None  # 雷达通道使用记录
    interceptT: List = None  # 拦截耗时


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


def environmental_selection(population, offspring):
    """
    环境选择函数
    :param population: 初始种群（列表，每个元素是一个字典）
    :param offspring: 后代种群（列表，每个元素是一个字典）
    :return: 更新后的种群
    """
    # 创建一个新的种群列表
    n = len(population)

    # 比较 Population 和 Offspring 的 objs 属性
    for i in range(n):
        # 获取 Population 和 Offspring 的 objs 值
        pop_objs = population[i].objs
        off_objs = offspring[i].objs

        # 判断 Offspring 是否优于 Population
        if off_objs > pop_objs:
            # 如果 Offspring 更优，替换 Population 中的个体
            population[i] = offspring[i]

    return population


class GeneticAlgorithm:
    def __init__(self, scenario_idx, init_pop_size,max_gen):

        """
        GA基本参数：
        """
        # 加载场景数据文件
        self.Prob = SetParameter(scenario_idx)
        self.init_pop_size = init_pop_size
        self.max_gen = max_gen

    # def initialize_population(self) -> List[Individual]:
    def initialize_population(
            self,
            return_format: str = "list"  # 可选 "list" 或 "numpy"
    ) -> Union[List[Individual], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """种群初始化"""
        num_target = len(self.Prob.Target.order)

        # 深拷贝Prob以避免修改原数据
        prob = copy.deepcopy(self.Prob)

        population = []
        pop = []
        for _ in range(self.init_pop_size):
            # 初始化个体
            ind = Individual(
                decs=np.zeros(num_target),
                decs1=[[] for _ in range(num_target)],
                decs2=[[] for _ in range(num_target)],
                TargetUsedRadar=[copy.deepcopy(prob.Radar.radar) for _ in range(num_target)],
                interceptT=[]
            )
            decs = np.zeros(num_target)

            # 为每个目标选择拦截方案
            for j in range(num_target):
                target_idx = prob.targetInd[j]
                schemes = prob.data[target_idx - 1]

                # 对前75%目标选择最优概率方案
                # if j < num_target * 0.75:
                #     max_prob = np.max(schemes[:, -1])
                #     valid_schemes = schemes[schemes[:, -1] == max_prob]
                #     prob.data[0][target_idx] = valid_schemes
                #     schemes = valid_schemes

                if len(schemes) > 0:
                    # 选择耗时最短的方案（简化版，实际需实现轮盘赌）
                    waste_time = schemes[:, 3] - schemes[:, 0]
                    selected_idx = roulette_selection(waste_time, 1)

                    # 记录方案信息
                    decs[j] = int(selected_idx[0])
                    ind.decs[j] = int(selected_idx[0])
                    selected_scheme = schemes[int(selected_idx[0])]
                    ind.decs1[j] = [
                        selected_scheme[1],  # 发射车
                        selected_scheme[2],  # 雷达
                        selected_scheme[-1],  # 概率
                        selected_scheme[0],  # 打击时刻
                        selected_scheme[3],  # 命中时刻
                        selected_scheme[5:8]  # 坐标
                    ]
            pop.append(decs)
            population.append(ind)

        return pop

    def operation(self, population):
        """交叉变异操作"""
        Prob = self.Prob
        # 数据提取
        target = Prob.targetInd
        N = len(population)
        n_target = len(target)
        offspring = population.copy()
        num = n_target / 3
        num1 = n_target / 6
        for i in range(N):
            # 交叉
            p1_decs = population[i].decs

            # 随机选择另一个个体作为父代
            p2_idx = random.sample(range(N), 1)[0]
            p2_decs = population[p2_idx].decs

            # 随机选择交叉位置
            ind = np.where(np.random.rand(n_target) < num / n_target)[0]

            for j in range(len(ind)):
                cur_pos = ind[j]
                p1_decs[cur_pos] = p2_decs[cur_pos]

            # 变异
            ind = np.where(np.random.rand(n_target) < num1 / n_target)[0]
            for j in range(len(ind)):
                cur_pos = ind[j]
                cur_target = target[cur_pos]

                # 获取当前目标的规则数量
                cur_target_scheme = Prob.data[cur_target - 1]
                cur_pick = np.random.randint(0, len(cur_target_scheme))  # 随机选取规则

                p1_decs[cur_pos] = cur_pick

            # 更新后代
            offspring[i].decs = p1_decs
        return offspring

    def evolve(self):
        """执行进化过程"""
        """种群初始化"""
        population = self.initialize_population()
        # 计算目标函数（需实现）
        population = cal_obj(population, self.Prob)

        """更新当前最优解"""
        all_objs = np.array([ind.objs for ind in population])
        best_fitness = np.max(all_objs)
        best_ind = np.argmax(all_objs)
        bestSol = copy.deepcopy(population[best_ind])

        for gen in range(self.max_gen):
            # 选择-交叉-变异
            selected_idx = roulette_selection(all_objs, self.init_pop_size)
            offspring = copy.deepcopy([population[idx] for idx in selected_idx])
            offspring = self.operation(offspring)
            # 计算目标函数（需实现）
            offspring = cal_obj(offspring, self.Prob)
            # 环境选择
            population = environmental_selection(population, offspring)

            # 返回最优解
            new_all_objs = np.array([ind.objs for ind in population])
            new_best_fitness = np.max(new_all_objs)
            new_best_ind = np.argmax(new_all_objs)
            bestSol = copy.deepcopy(population[new_best_ind])
            all_objs = new_all_objs.copy()
            # 打印进度
            print(f"Generation {gen}: Best Fitness = {new_best_fitness:.2f}")

        return bestSol


if __name__ == "__main__":
    ga = GeneticAlgorithm(scenario_idx=0,init_pop_size=100,max_gen=100)
    best_solution = ga.evolve()
    print(f"Optimal Value: {best_solution.objs:.2f}")
