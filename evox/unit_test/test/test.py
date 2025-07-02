import time
import torch
import matplotlib.pyplot as plt

from evox.algorithms import NSGA2, MOEAD, CCMO, NSGA3, TensorMOEAD, PPS, CMOEA_MS, GMPEA, GMPEA2, EMCMO
from evox.metrics import igd
from evox.problems.numerical import C1_DTLZ1,DTLZ1,DTLZ3,DTLZ2,C2_DTLZ2,C1_DTLZ3,C3_DTLZ4,DC1_DTLZ1,DC1_DTLZ3,DC2_DTLZ1,DC2_DTLZ3,DC3_DTLZ1,DC3_DTLZ3
from evox.problems.numerical import MW1, MW2, MW3, MW4, MW5, MW6, MW7, MW8, MW9, MW10, MW11, MW12, MW13, MW14
from evox.problems.numerical import LIRCMOP1, LIRCMOP2, LIRCMOP3 ,LIRCMOP4, LIRCMOP5, LIRCMOP6, LIRCMOP7, LIRCMOP8, LIRCMOP9, LIRCMOP10, LIRCMOP11, LIRCMOP12, LIRCMOP13, LIRCMOP14
from evox.workflows import StdWorkflow, EvalMonitor
from evox.operators.crossover import simulated_binary, DE_crossover

device = "cuda"
# Use GPU first to run the code.
torch.set_default_device(device)
print(torch.get_default_device())
max_gen = 200

# Init the problem, algorithm and workflow.
prob = MW11()
pf = prob.pf()
m = prob.m
algo = EMCMO(pop_size=200, n_objs=prob.m, lb=-torch.zeros(prob.d), ub=torch.ones(prob.d), max_gen=max_gen)


monitor = EvalMonitor()
workflow = StdWorkflow(algo, prob, monitor)

# Run the workflow
t = time.time()
workflow.init_step()
for i in range(max_gen):
    workflow.step()
    fit = workflow.algorithm.fit
    #cons = workflow.algorithm.cons
    fit = fit[~torch.isnan(fit).any(dim=1)]
    if i % 10 == 0:
        run_time = time.time() - t
        print(
            f"The IGD is {igd(fit, pf)} in {run_time:.4f} seconds at the {i + 1}th generation.")
fit = workflow.algorithm.fit
#cons = workflow.algorithm.cons
fit = fit[~torch.isnan(fit).any(dim=1)]
run_time = time.time() - t
print(
    f"The IGD is {igd(fit, pf)} in {run_time:.4f} seconds at the {max_gen}th generation.")

pf = pf.cpu()  # 这里将 pf 从 CUDA 转换为 CPU
fig = monitor.plot(problem_pf=pf)
#fig.show()



###########################画图###############################################
if m == 3:
    # 提取目标值
    objective_1 = fit[:, 0].cpu().numpy()  # 第一目标（x坐标）
    objective_2 = fit[:, 1].cpu().numpy()  # 第二目标（y坐标）
    objective_3 = fit[:, 2].cpu().numpy()  # 第三目标（z坐标）

    PF_1 = pf[:, 0].numpy()
    PF_2 = pf[:, 1].numpy()
    PF_3 = pf[:, 2].numpy()

    # 创建散点图
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(objective_1, objective_2, objective_3, c='blue', marker='o', alpha=0.99, label='Objective Values')
    ax.scatter(PF_1, PF_2, PF_3, c='yellow', marker='o', alpha=0.3, label='Pareto Front')
    ax.set_xlabel('Objective 1 (x)')
    ax.set_ylabel('Objective 2 (y)')
    ax.set_zlabel('Objective 3 (z)')
    ax.set_title('3D Scatter Plot of Objectives and Pareto Front')
    ax.legend()
    plt.show()
else:
    # 对于 m != 3 的情况（例如 m = 2）
    objective_1 = fit[:, 0].cpu().numpy()  # 第一目标（x坐标）
    objective_2 = fit[:, 1].cpu().numpy()  # 第二目标（y坐标）

    PF_1 = pf[:, 0].numpy()
    PF_2 = pf[:, 1].numpy()

    # 创建散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(PF_1, PF_2, c='yellow', marker='o', alpha=0.5, label='Pareto Front', s = 50)
    plt.scatter(objective_1, objective_2, c='blue', marker='o', alpha=0.99, label='Objective Values', s=10)
    plt.xlabel('Objective 1 (x)')
    plt.ylabel('Objective 2 (y)')
    plt.title('2D Scatter Plot of Objectives and Pareto Front')
    plt.legend()
    plt.grid(True)
    plt.show()
