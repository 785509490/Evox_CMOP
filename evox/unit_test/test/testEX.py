import time
import torch
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from evox.algorithms import NSGA2, MOEAD, CCMO, NSGA3, TensorMOEAD, PPS, CMOEA_MS, GMPEA, GMPEA2, EMCMO
from evox.metrics import igd
from evox.problems.numerical import (
    C1_DTLZ1, DTLZ1, DTLZ3, DTLZ2, C2_DTLZ2, C1_DTLZ3, C3_DTLZ4,
    DC1_DTLZ1, DC1_DTLZ3, DC2_DTLZ1, DC2_DTLZ3, DC3_DTLZ1, DC3_DTLZ3,
    MW1, MW2, MW3, MW4, MW5, MW6, MW7, MW8, MW9, MW10, MW11, MW12, MW13, MW14,
    LIRCMOP1, LIRCMOP2, LIRCMOP3, LIRCMOP4, LIRCMOP5, LIRCMOP6, LIRCMOP7,
    LIRCMOP8, LIRCMOP9, LIRCMOP10, LIRCMOP11, LIRCMOP12, LIRCMOP13, LIRCMOP14
)
from evox.workflows import StdWorkflow, EvalMonitor
from evox.operators.crossover import simulated_binary, DE_crossover


class BatchExperiment:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda')
        torch.set_default_device(self.device)
        print(f"Using device: {torch.get_default_device()}")

        # 创建保存目录
        self.base_dir = Path(config.get('base_dir', 'EXdata'))
        self.base_dir.mkdir(exist_ok=True)

    def tensor_to_serializable(self, tensor):
        """将torch tensor转换为可序列化的格式"""
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy().tolist()
        return tensor

    def check_experiment_exists(self, algorithm_name: str, problem_name: str, run_id: int) -> bool:
        """检查实验文件是否已存在"""
        # 创建算法特定的目录路径
        algo_dir = self.base_dir / algorithm_name

        # 构造文件名和路径
        filename = f"{algorithm_name}_{problem_name}_{run_id}.json"
        filepath = algo_dir / filename

        return filepath.exists()

    def run_single_experiment(self, algorithm_name: str, algorithm_class, problem_name: str,
                              problem_class, run_id: int) -> Dict[str, Any]:
        """运行单次实验"""
        print(f"Running {algorithm_name} on {problem_name} - Run {run_id}")

        # 初始化问题和算法
        prob = problem_class()
        pf = prob.pf()

        # 获取算法参数
        algo_params = {
            'pop_size': self.config['pop_size'],
            'n_objs': prob.m,
            'lb': -torch.zeros(prob.d),
            'ub': torch.ones(prob.d),
            'max_gen': self.config['max_gen'],
            'crossover_op': self.config['crossover_op']
        }

        # 处理不同算法的特定参数
        if algorithm_name in ['NSGA3', 'TensorMOEAD']:
            # 某些算法可能需要特殊处理
            pass

        algo = algorithm_class(**algo_params)

        monitor = EvalMonitor()
        workflow = StdWorkflow(algo, prob, monitor)

        # 记录数据
        experiment_data = {
            'algorithm': algorithm_name,
            'problem': problem_name,
            'run_id': run_id,
            'config': {
                'max_gen': self.config['max_gen'],
                'pop_size': self.config['pop_size'],
                'crossover_op': str(self.config['crossover_op']),
                'device': self.device
            },
            'generations': []
        }

        # 开始实验
        start_time = time.time()
        workflow.init_step()

        for gen in range(self.config['max_gen']):
            workflow.step()

            # 获取当前代数据
            current_time = time.time() - start_time
            if type(workflow.algorithm) == type(PPS):
                fit = workflow.algorithm.archfit
            else:
                fit = workflow.algorithm.fit

            # 处理约束（如果存在）
            cons = None
            if hasattr(workflow.algorithm, 'cons') and workflow.algorithm.cons is not None:
                if type(workflow.algorithm) == type(PPS):
                    cons = workflow.algorithm.archcons
                else:
                    cons = workflow.algorithm.cons

            # 获取决策变量
            pop = None
            if hasattr(workflow.algorithm, 'pop') and workflow.algorithm.pop is not None:
                if type(workflow.algorithm) == type(PPS):
                    pop = workflow.algorithm.archpop
                else:
                    pop = workflow.algorithm.pop
            elif hasattr(workflow.algorithm, 'population') and workflow.algorithm.population is not None:
                pop = workflow.algorithm.population

            # 移除NaN值
            if fit is not None:
                valid_mask = ~torch.isnan(fit).any(dim=1)
                fit_clean = fit[valid_mask]
                if cons is not None:
                    cons_clean = cons[valid_mask]
                else:
                    cons_clean = None
                if pop is not None:
                    pop_clean = pop[valid_mask]
                else:
                    pop_clean = None
            else:
                fit_clean = None
                cons_clean = None
                pop_clean = None

            # 记录这一代的数据
            gen_data = {
                'generation': gen + 1,
                'time': current_time,
                'fit': self.tensor_to_serializable(fit_clean),
                #'cons': self.tensor_to_serializable(cons_clean),
                # 'pop': self.tensor_to_serializable(pop_clean)
            }

            # 计算IGD（如果可能）
            if fit_clean is not None and len(fit_clean) > 0:
                try:
                    igd_value = igd(fit_clean, pf).item()
                    gen_data['igd'] = igd_value
                except:
                    gen_data['igd'] = None
            else:
                gen_data['igd'] = None

            # 只记录每10代以及最后一代
            if ((gen + 1) % 10 == 0) or (gen == self.config['max_gen'] - 1):
                experiment_data['generations'].append(gen_data)

            # 打印进度
            if (gen + 1) % 100 == 0:
                igd_str = f"IGD: {gen_data['igd']:.6f}" if gen_data['igd'] is not None else "IGD: N/A"
                print(f"  Gen {gen + 1}/{self.config['max_gen']}, {igd_str}, Time: {current_time:.2f}s")

        total_time = time.time() - start_time
        experiment_data['total_time'] = total_time

        print(f"  Completed in {total_time:.2f} seconds")
        return experiment_data

    def save_experiment_data(self, data: Dict[str, Any], algorithm_name: str):
        """保存实验数据到JSON文件"""
        # 创建算法特定的目录
        algo_dir = self.base_dir / algorithm_name
        algo_dir.mkdir(exist_ok=True)

        # 文件名格式: ALGORITHM_PROBLEM_RUN.json
        filename = f"{algorithm_name}_{data['problem']}_{data['run_id']}.json"
        filepath = algo_dir / filename

        # 保存数据
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"  Data saved to: {filepath}")

    def run_batch_experiments(self):
        """运行批量实验"""
        algorithms = self.config['algorithms']
        problems = self.config['problems']
        num_runs = self.config['num_runs']

        total_experiments = len(algorithms) * len(problems) * num_runs
        current_exp = 0
        skipped_count = 0

        print(
            f"Starting batch experiments: {len(algorithms)} algorithms × {len(problems)} problems × {num_runs} runs = {total_experiments} total experiments")
        print("=" * 80)

        for algo_name, algo_class in algorithms.items():
            print(f"Algorithm: {algo_name}")
            print("-" * 40)

            for prob_name, prob_class in problems.items():
                print(f"  Problem: {prob_name}")

                for run_id in range(1, num_runs + 1):
                    current_exp += 1

                    # 检查实验文件是否已存在
                    if self.check_experiment_exists(algo_name, prob_name, run_id):
                        skipped_count += 1
                        print(
                            f"    Run {run_id}/{num_runs} (Overall: {current_exp}/{total_experiments}) - SKIPPED (file exists)")
                        continue

                    print(f"    Run {run_id}/{num_runs} (Overall: {current_exp}/{total_experiments}) - RUNNING")

                    try:
                        # 运行实验
                        experiment_data = self.run_single_experiment(
                            algo_name, algo_class, prob_name, prob_class, run_id
                        )

                        # 保存数据
                        self.save_experiment_data(experiment_data, algo_name)

                    except Exception as e:
                        print(f"    ERROR in {algo_name} on {prob_name} run {run_id}: {str(e)}")
                        continue

                print()
            print()

        print("=" * 80)
        print("All experiments completed!")
        print(f"Total experiments: {total_experiments}")
        print(f"Skipped (already exists): {skipped_count}")
        print(f"Actually run: {total_experiments - skipped_count}")

        if skipped_count > 0:
            print(f"\nNote: {skipped_count} experiments were skipped because result files already exist.")
            print("Delete the corresponding JSON files if you want to re-run those experiments.")


def main():
    # 实验配置
    config = {
        # 公共参数
        'max_gen': 1000,
        'pop_size': 990,
        'crossover_op': DE_crossover,
        'num_runs': 5,  # 独立重复实验次数
        'device': 'cuda',
        'base_dir': 'E:/codePY/newevox-main/data',

        # 算法配置
        'algorithms': {
            'GMPEA2': GMPEA2,
            'NSGA2': NSGA2,
            'CCMO': CCMO,
            'CMOEA_MS': CMOEA_MS,
            'PPS': PPS,
            'EMCMO': EMCMO,
        },

        # 问题配置
        'problems': {
            'C1_DTLZ1': C1_DTLZ1,
            'C2_DTLZ2': C2_DTLZ2,
            'C1_DTLZ3': C1_DTLZ3,
            'C3_DTLZ4': C3_DTLZ4,
            'DC1_DTLZ1': DC1_DTLZ1,
            'DC1_DTLZ3': DC1_DTLZ3,
            'DC2_DTLZ1': DC2_DTLZ1,
            'DC2_DTLZ3': DC2_DTLZ3,
            'DC3_DTLZ1': DC3_DTLZ1,
            'DC3_DTLZ3': DC3_DTLZ3,
            # 'LIRCMOP1': LIRCMOP1,
            # 'LIRCMOP2': LIRCMOP2,
            # 'LIRCMOP3': LIRCMOP3,
            # 'LIRCMOP4': LIRCMOP4,
            # 'LIRCMOP5': LIRCMOP5,
            # 'LIRCMOP6': LIRCMOP6,
            # 'LIRCMOP7': LIRCMOP7,
            # 'LIRCMOP8': LIRCMOP8,
            # 'LIRCMOP9': LIRCMOP9,
            # 'LIRCMOP10': LIRCMOP10,
            # 'LIRCMOP11': LIRCMOP11,
            # 'LIRCMOP12': LIRCMOP12,
            # 'LIRCMOP13': LIRCMOP13,
            # 'LIRCMOP14': LIRCMOP14,
        }
    }

    # 创建并运行批量实验
    batch_exp = BatchExperiment(config)
    batch_exp.run_batch_experiments()


if __name__ == "__main__":
    main()