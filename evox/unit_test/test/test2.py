import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入问题类用于获取PF
from evox.problems.numerical import LIRCMOP1


def load_and_plot_final_objectives(json_file_path: str):
    """
    从JSON文件中读取最后一代的fit数据并画出来
    """
    # 检查文件是否存在
    json_path = Path(json_file_path)
    if not json_path.exists():
        print(f"Error: File {json_file_path} not found!")
        return

    # 读取JSON文件
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded: {json_path}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 提取基本信息
    algorithm = data.get('algorithm', 'Unknown')
    problem = data.get('problem', 'Unknown')
    run_id = data.get('run_id', 'Unknown')

    print(f"Algorithm: {algorithm}")
    print(f"Problem: {problem}")
    print(f"Run ID: {run_id}")

    # 获取所有代数的数据
    generations = data.get('generations', [])
    if not generations:
        print("No generation data found!")
        return

    print(f"Total generations: {len(generations)}")

    # 提取最后一代的fit数据
    last_generation = generations[-1]
    final_fit = last_generation.get('fit', [])
    final_igd = last_generation.get('igd', None)
    final_generation_num = last_generation.get('generation', len(generations))

    if not final_fit:
        print("No fit data found in the last generation!")
        return

    print(f"Final generation: {final_generation_num}")
    print(f"Final IGD: {final_igd}")
    print(f"Number of solutions: {len(final_fit)}")

    # 转换为numpy数组
    final_fit = np.array(final_fit)
    print(f"Objective dimensions: {final_fit.shape[1]}")

    # 移除NaN值（如果有的话）
    valid_mask = ~np.isnan(final_fit).any(axis=1)
    final_fit_clean = final_fit[valid_mask]

    if len(final_fit_clean) != len(final_fit):
        print(f"Removed {len(final_fit) - len(final_fit_clean)} solutions with NaN values")

    print(f"Valid solutions: {len(final_fit_clean)}")

    # 获取问题的Pareto前沿
    pf = None
    if problem == 'LIRCMOP1':
        try:
            prob = LIRCMOP1()
            pf = prob.pf()
            if hasattr(pf, 'cpu'):
                pf = pf.cpu().numpy()
            elif hasattr(pf, 'numpy'):
                pf = pf.numpy()
            print(f"Pareto front points: {len(pf)}")
        except Exception as e:
            print(f"Error getting Pareto front: {e}")

    # 绘图
    m = final_fit_clean.shape[1]  # 目标维数

    if m == 2:
        # 2D散点图
        plt.figure(figsize=(10, 8))

        # 绘制Pareto前沿（如果存在）
        if pf is not None:
            plt.scatter(pf[:, 0], pf[:, 1], c='yellow', marker='o', alpha=0.3,
                        s=30, label='Pareto Front', zorder=1)

        # 绘制算法结果
        plt.scatter(final_fit_clean[:, 0], final_fit_clean[:, 1], c='blue', marker='o',
                    alpha=0.8, s=50, label=f'{algorithm} Solutions', zorder=2)

        plt.xlabel('Objective 1', fontsize=12)
        plt.ylabel('Objective 2', fontsize=12)
        plt.title(f'{algorithm} on {problem} - Run {run_id}\n'
                  f'Final Generation {final_generation_num} (IGD: {final_igd:.6f})',
                  fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 显示解的具体坐标
        print("\nFinal objectives (first 10 solutions):")
        for i, obj in enumerate(final_fit_clean[:10]):
            print(f"  Solution {i + 1}: [{obj[0]:.6f}, {obj[1]:.6f}]")
        if len(final_fit_clean) > 10:
            print(f"  ... and {len(final_fit_clean) - 10} more solutions")

    elif m == 3:
        # 3D散点图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制Pareto前沿（如果存在）
        if pf is not None:
            ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], c='yellow', marker='o',
                       alpha=0.3, s=30, label='Pareto Front')

        # 绘制算法结果
        ax.scatter(final_fit_clean[:, 0], final_fit_clean[:, 1], final_fit_clean[:, 2],
                   c='blue', marker='o', alpha=0.8, s=50, label=f'{algorithm} Solutions')

        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')
        ax.set_title(f'{algorithm} on {problem} - Run {run_id}\n'
                     f'Final Generation {final_generation_num} (IGD: {final_igd:.6f})')
        ax.legend()

    else:
        print(f"Plotting not supported for {m}-objective problems")
        return

    plt.tight_layout()

    # 保存图片
    save_path = f"{algorithm}_{problem}_run{run_id}_final_objectives.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")

    plt.show()

    # 显示统计信息
    print(f"\nObjective Statistics:")
    for i in range(m):
        obj_values = final_fit_clean[:, i]
        print(f"  Objective {i + 1}: min={obj_values.min():.6f}, "
              f"max={obj_values.max():.6f}, mean={obj_values.mean():.6f}")


def main():
    """
    主函数 - 测试脚本
    """
    # 默认文件路径（根据你的数据目录结构调整）
    default_path = r"E:\codePY\newevox-main\data\GMPEA2\GMPEA2_LIRCMOP12_2.json"

    # 你也可以修改为其他文件路径进行测试
    #json_file_path = input(f"Enter JSON file path [default: {default_path}]: ").strip()
    #if not json_file_path:
    json_file_path = default_path

    print(f"Loading file: {json_file_path}")
    print("=" * 60)

    # 加载并可视化
    load_and_plot_final_objectives(json_file_path)


if __name__ == "__main__":
    main()