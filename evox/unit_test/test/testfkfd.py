import torch
import time
from evox.algorithms import GA
from evox.problems.numerical import FKFD
from evox.workflows import StdWorkflow, EvalMonitor
import argparse


parser = argparse.ArgumentParser(description='xx算法')
parser.add_argument('--SCENARIO_IDX', default=0, type=int, help='xxx')
args = parser.parse_args()

if torch.cuda.is_available():
    print("CUDA is available. You can use GPU for computations.")
else:
    print("CUDA is not available. Using CPU for computations.")

# Use GPU first to run the code.
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.get_default_device())

algorithm = GA(pop_size=2000, scenario_idx=args.SCENARIO_IDX)
problem = FKFD(scenario_idx=args.SCENARIO_IDX)
monitor = EvalMonitor()
workflow = StdWorkflow(algorithm, problem, monitor)
t = time.time()
workflow.init_step()
for i in range(20):
    workflow.step()
    if (i + 1) % 10 == 0:
        run_time = time.time() - t
        top_fitness = monitor.topk_fitness
        print(f"The top fitness is {top_fitness} and max hit p is {1/top_fitness} in {run_time:.4f} seconds at the {i + 1}th generation.")

monitor.plot() #