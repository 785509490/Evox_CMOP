import torch
from evox.algorithms import PSO
from evox.problems.numerical import Ackley
from evox.workflows import StdWorkflow, EvalMonitor

algorithm = PSO(pop_size=100, lb=-32 * torch.ones(10), ub=32 * torch.ones(10))
problem = Ackley()
monitor = EvalMonitor()
workflow = StdWorkflow(algorithm, problem, monitor)
workflow.init_step()
for i in range(100):
    workflow.step()
    if (i + 1) % 10 == 0:
        run_time = 0
        top_fitness = monitor.topk_fitness
        print(f"The top fitness is {top_fitness} in {run_time:.4f} seconds at the {i + 1}th generation.")

monitor.plot() # or