import torch

from evox.core import Problem
from evox.operators.sampling import grid_sampling, uniform_sampling
from evox.operators.selection import non_dominate_rank
import numpy as np

class MW(Problem):
    def __init__(self, d: int = None, m: int = None, ref_num: int = 1000):
        super().__init__()
        self.d = d
        self.m = m
        self.ref_num = ref_num
        self.sample, _ = uniform_sampling(self.ref_num * self.m, self.m)  # Assuming UniformSampling is defined
        self.device = self.sample.device

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def pf(self):
        f = self.sample / 2
        return f

class MW1(MW):
    def __init__(self, d: int = 15, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        X_selected = X[:, self.m:self.d]
        power = self.d - self.m  # 这是一个常量
        g = 1 + torch.sum(1 - torch.exp(-10 * ((X_selected ** power) - 0.5 -
                                               (torch.arange(self.m, self.d, device=X.device) - 1) / (2 * self.d)).pow(
            2)),
                          dim=1, keepdim=True)
        PopObj = torch.zeros((X.size(0), 2), device=X.device)
        PopObj[:, 0] = X[:, 0]
        PopObj[:, 1] = g.squeeze() * (1 - 0.85 * PopObj[:, 0] / g.squeeze())
        l = torch.sqrt(torch.tensor(2.0, device=X.device)) * PopObj[:, 1] - torch.sqrt(
            torch.tensor(2.0, device=X.device)) * PopObj[:, 0]
        PopCon = torch.sum(PopObj, dim=1) - 1 - 0.5 * torch.sin(2 * torch.pi * l) ** 8


        return PopObj, PopCon.unsqueeze(1)

    def feas(self) -> torch.Tensor:
        x = torch.linspace(0, 1, 400)
        y = torch.linspace(0, 1.5, 400)
        x, y = torch.meshgrid(x, y)
        z = torch.full_like(x, float('nan'))
        fes = (x + y - 1 - 0.5 * torch.sin(2 * torch.pi * (torch.sqrt(torch.tensor(2.0)) * y -
                                                           torch.sqrt(torch.tensor(2.0)) * x)).pow(8) <= 0)
        z[(fes & (0.85 * x + y >= 1))] = 0
        R = torch.stack((x, y, z), dim=-1)  # stack to create a 3D tensor
        mask = ~torch.isnan(R).any(dim=-1)
        R = R[mask]

        return R.reshape(-1, 2)

    def pf(self) -> torch.Tensor:
        N = 5000
        R = torch.zeros((N, 2))
        R[:, 0] = torch.linspace(0, 1, N)
        R[:, 1] = 1 - 0.85 * R[:, 0]
        l = (torch.sqrt(torch.tensor(2.0)) * R[:, 1]) - (torch.sqrt(torch.tensor(2.0)) * R[:, 0])
        c = 1 - R[:, 0] - R[:, 1] + 0.5 * torch.sin(2 * torch.pi * l) ** 8
        valid_mask = c >= 0
        R = R[valid_mask]

        return R

class MW2(MW):
    def __init__(self, d: int = 15, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        z = 1 - torch.exp(
            -10 * (X[:, self.m:] - (torch.arange(self.m, self.d, device=X.device).reshape(1, -1) - 1) / self.d) ** 2)
        g = 1 + torch.sum(1.5 + (0.1 / self.d) * z ** 2 - 1.5 * torch.cos(2 * torch.pi * z), dim=1, keepdim=True)
        PopObj = torch.zeros(X.size(0), self.m, device=X.device)
        PopObj[:, 0] = X[:, 0]
        PopObj[:, 1] = g.squeeze() * (1 - PopObj[:, 0] / g.squeeze())
        l = torch.sqrt(torch.tensor(2.0, device=X.device)) * PopObj[:, 1] - torch.sqrt(
            torch.tensor(2.0, device=X.device)) * PopObj[:, 0]
        PopCon = torch.sum(PopObj, dim=1) - 1 - 0.5 * torch.sin(3 * torch.pi * l) ** 8
        return PopObj, PopCon.unsqueeze(1)

    def pf(self) -> torch.Tensor:
        N = 5000
        R = torch.zeros((N, 2))
        R[:, 0] = torch.linspace(0, 1, N)
        R[:, 1] = 1 - R[:, 0]
        return R

class MW3(MW):
    def __init__(self, d: int = 15, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # 计算 g
        g = 1 + torch.sum(2 * (X[:, self.m:] + (X[:, self.m - 1:-1] - 0.5) ** 2 - 1) ** 2, dim=1, keepdim=True)
        PopObj = torch.zeros(X.size(0), self.m, device=X.device)
        PopObj[:, 0] = X[:, 0]
        PopObj[:, 1] = g.squeeze() * (1 - PopObj[:, 0] / g.squeeze())
        l = torch.sqrt(torch.tensor(2.0, device=X.device)) * PopObj[:, 1] - torch.sqrt(torch.tensor(2.0, device=X.device)) * PopObj[:, 0]
        PopCon = torch.zeros(X.size(0), 2, device=X.device)
        PopCon[:, 0] = torch.sum(PopObj, dim=1) - 1.05 - 0.45 * torch.sin(0.75 * torch.pi * l) ** 6
        PopCon[:, 1] = 0.85 - torch.sum(PopObj, dim=1) + 0.3 * torch.sin(0.75 * torch.pi * l) ** 2

        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        R = torch.zeros((N, 2), dtype=torch.float32)
        R[:, 0] = torch.linspace(0, 1, N)
        R[:, 1] = 1 - R[:, 0]
        invalid = (0.85 - R[:, 0] - R[:, 1] + 0.3 * torch.sin(
            0.75 * torch.pi * torch.sqrt(torch.tensor(2.0)) * (R[:, 1] - R[:, 0])) ** 2) > 0
        while torch.any(invalid):
            R[invalid, :] *= 1.001
            invalid = (0.85 - R[:, 0] - R[:, 1] + 0.3 * torch.sin(
                0.75 * torch.pi * torch.sqrt(torch.tensor(2.0)) * (R[:, 1] - R[:, 0])) ** 2) > 0

        return R
class MW4(MW):
    def __init__(self, d: int = 15, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        K = self.d - self.m
        range_tensor = (torch.arange(self.m-1, self.d, device=X.device) - 1)
        g = torch.sum(1 - torch.exp(-10 * ((X[:, self.m - 1:] ** (K)) - 0.5 - (range_tensor / (2 * self.d))) ** 2),
                      dim=1, keepdim=True)
        PopObj = (1 + g) * torch.flip(torch.cumprod(
            torch.cat([torch.ones((X.size(0), 1), device=X.device), X[:, :self.m - 1]], dim=1), dim=1), dims=[1]) * \
                 torch.cat([torch.ones((X.size(0), 1), device=X.device), 1 - torch.flip(X[:, 0:self.m-1], dims=[1])], dim=1)
        l = PopObj[:, -1].unsqueeze(1) - torch.sum(PopObj[:, :-1], dim=1, keepdim=True)  # 结果形状 (200, 1)
        PopCon = torch.sum(PopObj, dim=1, keepdim=True) - (1 + 0.4 * torch.sin(2.5 * torch.pi * l) ** 8)

        return PopObj, PopCon
    def pf(self):
        f = self.sample
        return f

class MW5(MW):
    def __init__(self, d: int = 15, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        g = 1 + torch.sum(
            1 - torch.exp(-10 * ((X[:, self.m:] ** (self.d - self.m))
                                 - 0.5 - (torch.arange(self.m, self.d, device=X.device).view(1, -1) - 1) / (
                                             2 * self.d)) ** 2),
            dim=1, keepdim=True)
        PopObj = torch.empty((X.size(0), 2), device=X.device)
        PopObj[:, 0] = g.squeeze() * X[:, 0]
        PopObj[:, 1] = g.squeeze() * torch.sqrt(1 - (PopObj[:, 0] / g.squeeze()) ** 2)
        l1 = torch.atan(PopObj[:, 1] / PopObj[:, 0])
        l2 = 0.5 * torch.pi - 2 * torch.abs(l1 - 0.25 * torch.pi)
        PopCon = torch.empty((X.size(0), 3), device=X.device)
        PopCon[:, 0] = PopObj[:, 0] ** 2 + PopObj[:, 1] ** 2 - (1.7 - 0.2 * torch.sin(2 * l1)) ** 2
        PopCon[:, 1] = (1 + 0.5 * torch.sin(6 * l2 ** 3)) ** 2 - PopObj[:, 0] ** 2 - PopObj[:, 1] ** 2
        PopCon[:, 2] = (1 - 0.45 * torch.sin(6 * l2 ** 3)) ** 2 - PopObj[:, 0] ** 2 - PopObj[:, 1] ** 2
        return PopObj, PopCon

    def pf(self):
        R = torch.tensor([[0, 1],
                          [0.3922, 0.9199],
                          [0.4862, 0.8739],
                          [0.5490, 0.8358],
                          [0.5970, 0.8023],
                          [0.6359, 0.7719],
                          [0.6686, 0.7436],
                          [0.6969, 0.7174]])
        R = torch.cat((R, torch.flip(R, dims=[1])), dim=0)
        return R

class MW6(MW):
    def __init__(self, d: int = 15, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        M = self.m
        D = self.d
        z = 1 - torch.exp(-10 * (X[:, M:] - (torch.arange(M, D, device=X.device).view(1, -1) - 1) / D) ** 2)
        g = 1 + torch.sum(1.5 + (0.1 / D) * z ** 2 - 1.5 * torch.cos(2 * torch.pi * z), dim=1, keepdim=True)
        PopObj = torch.empty((X.size(0), 2), device=X.device)
        PopObj[:, 0] = g.squeeze() * X[:, 0] * 1.0999
        PopObj[:, 1] = g.squeeze() * torch.sqrt(1.1 * 1.1 - (PopObj[:, 0] / g.squeeze()) ** 2)
        l = torch.cos(6 * torch.atan(PopObj[:, 1] / (PopObj[:, 0] + 1e-10)) ** 4) ** 10  # 添加小常数避免除零
        PopCon = (PopObj[:, 0] / (1 + 0.15 * l)) ** 2 + (PopObj[:, 1] / (1 + 0.75 * l)) ** 2 - 1

        return PopObj, PopCon.unsqueeze(1)

    def pf(self) -> torch.Tensor:
        N = 5000
        R = torch.zeros((N, 2), dtype=torch.float32)
        R[:, 0] = torch.linspace(0, 1, N)
        R[:, 1] = 1 - R[:, 0]
        norm_factor = torch.sqrt(torch.sum(R ** 2, dim=1) / 1.21).view(-1, 1)
        R = R / norm_factor
        l = torch.cos(6 * torch.atan(R[:, 1] / (R[:, 0] + 1e-10)) ** 4) ** 10  # 加小常数避免除零
        c = 1 - (R[:, 0] / (1 + 0.15 * l)) ** 2 - (R[:, 1] / (1 + 0.75 * l)) ** 2
        R = R[c >= 0]
        return R

class MW7(MW):
    def __init__(self, d: int = 15, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        m = self.m
        g = 1 + torch.sum(2 * (X[:, m - 1:] + (X[:, m - 2:-1] - 0.5) ** 2 - 1) ** 2, dim=1, keepdim=True)
        PopObj = torch.empty((X.size(0), 2), device=X.device)
        PopObj[:, 0] = g.squeeze() * X[:, 0]
        PopObj[:, 1] = g.squeeze() * torch.sqrt(1 - (PopObj[:, 0] / (g.squeeze() + 1e-10)) ** 2)
        l = torch.atan2(PopObj[:, 1], PopObj[:, 0])
        PopCon = torch.empty((X.size(0), 2), device=X.device)
        PopCon[:, 0] = PopObj[:, 0] ** 2 + PopObj[:, 1] ** 2 - (1.2 + 0.4 * torch.sin(4 * l) ** 16) ** 2
        PopCon[:, 1] = (1.15 - 0.2 * torch.sin(4 * l) ** 8) ** 2 - PopObj[:, 0] ** 2 - PopObj[:, 1] ** 2
        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        R = torch.zeros((N, 2), dtype=torch.float32)
        R[:, 0] = torch.linspace(0, 1, N)
        R[:, 1] = 1 - R[:, 0]
        norm = torch.sqrt(torch.sum(R ** 2, dim=1)).view(-1, 1)
        R = R / norm
        invalid = ((1.15 - 0.2 * torch.sin(4 * torch.atan(R[:, 1] / (R[:, 0] + 1e-10))) ** 8) ** 2 - R[:, 0] ** 2 - R[:, 1] ** 2) > 0
        while torch.any(invalid):
            R[invalid, :] *= 1.001
            invalid = ((1.15 - 0.2 * torch.sin(4 * torch.atan(R[:, 1] / (R[:, 0] + 1e-10))) ** 8) ** 2 - R[:, 0] ** 2 - R[:, 1] ** 2) > 0
        rank = non_dominate_rank(R)
        R = R[rank == 0]
        return R

class MW8(MW):
    def __init__(self, d: int = 15, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> tuple:
        z = 1 - torch.exp(-10 * (X[:, self.m - 1:] - (torch.arange(self.m-1, self.d, device=X.device) - 1) / self.d) **2)
        g = torch.sum(1.5 + (0.1 / self.d) * z ** 2 - 1.5 * torch.cos(2 * torch.pi * z), dim=1, keepdim=True)
        a = torch.flip(torch.cumprod(
            torch.cat([
                torch.ones((X.size(0), 1), device=X.device),
                torch.cos(X[:, 0:self.m - 1] * torch.pi / 2)
            ], dim=1), dim=1), dims=[1])
        b = torch.cat([
                torch.ones((X.size(0), 1), device=X.device),
                torch.sin(torch.flip(X[:, 0:self.m - 1], dims=[1]) * torch.pi / 2)
            ], dim=1)
        PopObj = (1 + g) * a * b
        l = torch.asin(PopObj[:, -1].unsqueeze(1) / torch.sqrt(torch.sum(PopObj ** 2, dim=1, keepdim=True)))
        PopCon = torch.sum(PopObj ** 2, dim=1, keepdim=True) - (1.25 - 0.5 * torch.sin(6 * l) ** 2) ** 2

        return PopObj, PopCon

    def pf(self):
        R = self.sample
        R = R / torch.sqrt(torch.sum(R ** 2, dim=1, keepdim=True))
        condition = 1 - (1.25 - 0.5 * torch.sin(6 * torch.asin(R[:, -1]))**2) ** 2 > 0
        R = R[~condition,:]
        return R

class MW9(MW):
    def __init__(self, d: int = 15, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        g = 1 + torch.sum(
            1 - torch.exp(-10 * ((X[:, self.m-1:] ** (self.d - self.m)) - 0.5 -
             (torch.arange(self.m, self.d + 1, device=X.device).unsqueeze(0) - 1) / (2 * self.d)) ** 2),
            dim=1, keepdim=True
        )
        PopObj = torch.zeros((X.size(0), 2), device=X.device)
        PopObj[:, 0] = g.view(-1) * X[:, 0]
        PopObj[:, 1] = g.view(-1) * (1 - (PopObj[:, 0] / g.view(-1)) ** 0.6)
        T1 = (1 - 0.64 * PopObj[:, 0] ** 2 - PopObj[:, 1]) * (1 - 0.36 * PopObj[:, 0] ** 2 - PopObj[:, 1])
        T2 = 1.35 ** 2 - (PopObj[:, 0] + 0.35) ** 2 - PopObj[:, 1]
        T3 = 1.15 ** 2 - (PopObj[:, 0] + 0.15) ** 2 - PopObj[:, 1]
        PopCon = torch.min(T1, T2 * T3)
        return PopObj, PopCon.unsqueeze(1)

    def pf(self) -> torch.Tensor:
        N = 5000
        R = torch.zeros((N, 2))
        R[:, 0] = torch.linspace(0, 1, N)
        R[:, 1] = 1 - R[:, 0] ** 0.6
        T1 = (1 - 0.64 * R[:, 0] ** 2 - R[:, 1]) * (1 - 0.36 * R[:, 0] ** 2 - R[:, 1])
        T2 = 1.35 ** 2 - (R[:, 0] + 0.35) ** 2 - R[:, 1]
        T3 = 1.15 ** 2 - (R[:, 0] + 0.15) ** 2 - R[:, 1]
        invalid = torch.min(T1, T2 * T3) > 0
        while torch.any(invalid):
            R[invalid, :] *= 1.001
            T1 = (1 - 0.64 * R[:, 0] ** 2 - R[:, 1]) * (1 - 0.36 * R[:, 0] ** 2 - R[:, 1])
            T2 = 1.35 ** 2 - (R[:, 0] + 0.35) ** 2 - R[:, 1]
            T3 = 1.15 ** 2 - (R[:, 0] + 0.15) ** 2 - R[:, 1]
            invalid = torch.min(T1, T2 * T3) > 0
        rank = non_dominate_rank(R)
        R = R[rank == 0]
        return R

class MW10(MW):
    def __init__(self, d: int = 15, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        device = X.device
        z = 1 - torch.exp(
            -10 * (X[:, self.m-1:] - (torch.arange(self.m-1, self.d, device=device).unsqueeze(0) - 1) / self.d) ** 2)
        g = 1 + torch.sum((1.5 + (0.1 / self.d) * z ** 2 - 1.5 * torch.cos(2 * torch.pi * z)), dim=1, keepdim=True)
        PopObj = torch.zeros((X.size(0), 2), device=device)
        PopObj[:, 0] = g.view(-1) * (X[:, 0] ** self.d)
        PopObj[:, 1] = g.view(-1) * (1 - (PopObj[:, 0] / g.view(-1)) ** 2)
        PopCon = torch.zeros((X.size(0), 3), device=device)
        PopCon[:, 0] = -(2 - 4 * PopObj[:, 0] ** 2 - PopObj[:, 1]) * (2 - 8 * PopObj[:, 0] ** 2 - PopObj[:, 1])
        PopCon[:, 1] = (2 - 2 * PopObj[:, 0] ** 2 - PopObj[:, 1]) * (2 - 16 * PopObj[:, 0] ** 2 - PopObj[:, 1])
        PopCon[:, 2] = (1 - PopObj[:, 0] ** 2 - PopObj[:, 1]) * (1.2 - 1.2 * PopObj[:, 0] ** 2 - PopObj[:, 1])

        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        R = torch.zeros((N, 2))
        R[:, 0] = torch.arange(0, 1 + 1 / (N - 1), 1 / (N - 1), device=R.device)
        R[:, 1] = 1 - R[:, 0] ** 2
        c1 = (2 - 4 * R[:, 0] ** 2 - R[:, 1]) * (2 - 8 * R[:, 0] ** 2 - R[:, 1])
        c2 = (2 - 2 * R[:, 0] ** 2 - R[:, 1]) * (2 - 16 * R[:, 0] ** 2 - R[:, 1])
        c3 = (1 - R[:, 0] ** 2 - R[:, 1]) * (1.2 - 1.2 * R[:, 0] ** 2 - R[:, 1])
        invalid = (c1 < 0) | (c2 > 0) | (c3 > 0)
        while torch.any(invalid):
            R[invalid, :] *= 1.001
            R[torch.any(R > 1.3, dim=1), :] = torch.nan
            R = R[~torch.isnan(R).any(dim=1)]
            c1 = (2 - 4 * R[:, 0] ** 2 - R[:, 1]) * (2 - 8 * R[:, 0] ** 2 - R[:, 1])
            c2 = (2 - 2 * R[:, 0] ** 2 - R[:, 1]) * (2 - 16 * R[:, 0] ** 2 - R[:, 1])
            c3 = (1 - R[:, 0] ** 2 - R[:, 1]) * (1.2 - 1.2 * R[:, 0] ** 2 - R[:, 1])
            invalid = (c1 < 0) | (c2 > 0) | (c3 > 0)
        rank = non_dominate_rank(R)
        R = R[rank == 0]
        return R

class MW11(MW):
    def __init__(self, d: int = 15, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        device = X.device
        g = 1 + torch.sum(2 * (X[:, self.m - 1:] + (X[:, self.m - 2:-1] - 0.5) ** 2 - 1) ** 2, dim=1, keepdim=True)
        PopObj = torch.zeros((X.size(0), 2), device=device)
        PopObj[:, 0] = g.view(-1) * X[:, 0] * torch.sqrt(torch.tensor(1.9999, device=device))  # 第一个目标
        PopObj[:, 1] = g.view(-1) * torch.sqrt(2 - (PopObj[:, 0] / g.view(-1)) ** 2)  # 第二个目标
        PopCon = torch.zeros((X.size(0), 4), device=device)
        PopCon[:, 0] = -(3 - PopObj[:, 0] ** 2 - PopObj[:, 1]) * (3 - 2 * PopObj[:, 0] ** 2 - PopObj[:, 1])
        PopCon[:, 1] = (3 - 0.625 * PopObj[:, 0] ** 2 - PopObj[:, 1]) * (3 - 7 * PopObj[:, 0] ** 2 - PopObj[:, 1])
        PopCon[:, 2] = -(1.62 - 0.18 * PopObj[:, 0] ** 2 - PopObj[:, 1]) * (
                    1.125 - 0.125 * PopObj[:, 0] ** 2 - PopObj[:, 1])
        PopCon[:, 3] = (2.07 - 0.23 * PopObj[:, 0] ** 2 - PopObj[:, 1]) * (
                    0.63 - 0.07 * PopObj[:, 0] ** 2 - PopObj[:, 1])

        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        R = torch.zeros((N, 2))
        R[:, 0] = torch.arange(0, 1 + (1 / (N - 1)), 1 / (N - 1), device=R.device)
        R[:, 1] = 1 - R[:, 0]
        R /= torch.sqrt(torch.sum(R ** 2, dim=1, keepdim=True) / 2)
        c1 = (3 - R[:, 0] ** 2 - R[:, 1]) * (3 - 2 * R[:, 0] ** 2 - R[:, 1])
        c2 = (3 - 0.625 * R[:, 0] ** 2 - R[:, 1]) * (3 - 7 * R[:, 0] ** 2 - R[:, 1])
        c3 = (1.62 - 0.18 * R[:, 0] ** 2 - R[:, 1]) * (1.125 - 0.125 * R[:, 0] ** 2 - R[:, 1])
        c4 = (2.07 - 0.23 * R[:, 0] ** 2 - R[:, 1]) * (0.63 - 0.07 * R[:, 0] ** 2 - R[:, 1])
        invalid = (c1 < 0) | (c2 > 0) | (c3 < 0) | (c4 > 0)
        while torch.any(invalid):
            R[invalid, :] *= 1.001
            R[torch.any(R > 2.2, dim=1), :] = torch.nan
            R = R[~torch.isnan(R).any(dim=1)]
            c1 = (3 - R[:, 0] ** 2 - R[:, 1]) * (3 - 2 * R[:, 0] ** 2 - R[:, 1])
            c2 = (3 - 0.625 * R[:, 0] ** 2 - R[:, 1]) * (3 - 7 * R[:, 0] ** 2 - R[:, 1])
            c3 = (1.62 - 0.18 * R[:, 0] ** 2 - R[:, 1]) * (1.125 - 0.125 * R[:, 0] ** 2 - R[:, 1])
            c4 = (2.07 - 0.23 * R[:, 0] ** 2 - R[:, 1]) * (0.63 - 0.07 * R[:, 0] ** 2 - R[:, 1])
            invalid = (c1 < 0) | (c2 > 0) | (c3 < 0) | (c4 > 0)
        R = torch.cat((R, torch.tensor([[1.0, 1.0]], device=R.device)), dim=0)
        rank = non_dominate_rank(R)
        R = R[rank == 0]
        return R


class MW12(MW):
    def __init__(self, d: int = 15, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        device = X.device
        idx = torch.arange(self.m - 1, self.d, device=device).unsqueeze(0)  # 创建索引行
        g = 1 + torch.sum(
            1 - torch.exp(-10 * ((X[:, self.m - 1:] ** (self.d - self.m)) - 0.5 - (idx - 1) / (2 * self.d)) ** 2),
            dim=1, keepdim=True)
        PopObj = torch.zeros((X.size(0), 2), device=device)
        PopObj[:, 0] = g.squeeze() * X[:, 0]
        PopObj[:, 1] = g.squeeze() * (0.85 -
                                      0.8 * (PopObj[:, 0] / g.squeeze()) -
                                      0.08 * torch.abs(
                    torch.sin(3.2 * torch.pi * (PopObj[:, 0] / g.squeeze()))))
        PopCon = torch.zeros((X.size(0), 2), device=device)
        PopCon[:, 0] = (1 - 0.8 * PopObj[:, 0] - PopObj[:, 1] +
                        0.08 * torch.sin(2 * torch.pi * (PopObj[:, 1] - PopObj[:, 0] / 1.5))) * \
                       (1.8 - 1.125 * PopObj[:, 0] - PopObj[:, 1] +
                        0.08 * torch.sin(2 * torch.pi * (PopObj[:, 1] / 1.8 - PopObj[:, 0] / 1.6)))
        PopCon[:, 1] = -(1 - 0.625 * PopObj[:, 0] - PopObj[:, 1] +
                         0.08 * torch.sin(2 * torch.pi * (PopObj[:, 1] - PopObj[:, 0] / 1.6))) * \
                       (1.4 - 0.875 * PopObj[:, 0] - PopObj[:, 1] +
                        0.08 * torch.sin(2 * torch.pi * (PopObj[:, 1] / 1.4 - PopObj[:, 0] / 1.6)))
        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        R = torch.zeros((N, 2))  # 或 'cpu'，根据环境而定
        R[:, 0] = torch.arange(0, 1 + 1 / (N - 1), 1 / (N - 1), device=R.device)
        R[:, 1] = 0.85 - 0.8 * R[:, 0] - 0.08 * torch.abs(torch.sin(3.2 * torch.pi * R[:, 0]))

        c1 = (1 - 0.8 * R[:, 0] - R[:, 1] + 0.08 * torch.sin(2 * torch.pi * (R[:, 1] - R[:, 0] / 1.5))) * \
             (1.8 - 1.125 * R[:, 0] - R[:, 1] + 0.08 * torch.sin(2 * torch.pi * (R[:, 1] / 1.8 - R[:, 0] / 1.6)))

        invalid = c1 > 0
        while torch.any(invalid):
            R[invalid, :] = R[invalid, :] * 1.001
            c1 = (1 - 0.8 * R[:, 0] - R[:, 1] + 0.08 * torch.sin(2 * torch.pi * (R[:, 1] - R[:, 0] / 1.5))) * \
                 (1.8 - 1.125 * R[:, 0] - R[:, 1] + 0.08 * torch.sin(2 * torch.pi * (R[:, 1] / 1.8 - R[:, 0] / 1.6)))
            invalid = c1 > 0
        return R


class MW13(MW):
    def __init__(self, d: int = 15, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        device = X.device
        idx = torch.arange(self.m-1, self.d, device=device).unsqueeze(0)  # 创建行索引
        z = 1 - torch.exp(-10 * (X[:, self.m - 1:] - (idx - 1) / self.d) ** 2)

        g = 1 + torch.sum(1.5 + (0.1 / self.d) * z ** 2 - 1.5 * torch.cos(2 * torch.pi * z), dim=1, keepdim=True)

        PopObj = torch.zeros((X.size(0), 2), device=device)
        PopObj[:, 0] = g.squeeze() * X[:, 0] * 1.5  # 第一个目标
        PopObj[:, 1] = g.squeeze() * (5 - torch.exp(PopObj[:, 0] / g.squeeze()) -
                                      torch.abs(0.5 * torch.sin(3 * torch.pi * PopObj[:, 0] / g.squeeze())))  # 第二个目标

        PopCon = torch.zeros((X.size(0), 2), device=device)
        PopCon[:, 0] = (5 - torch.exp(PopObj[:, 0]) -
                        0.5 * torch.sin(3 * torch.pi * PopObj[:, 0]) - PopObj[:, 1]) * \
                       (5 - (1 + 0.4 * PopObj[:, 0]) -
                        0.5 * torch.sin(3 * torch.pi * PopObj[:, 0]) - PopObj[:, 1])
        PopCon[:, 1] = -(5 - (1 + PopObj[:, 0] + 0.5 * PopObj[:, 0] ** 2) -
                         0.5 * torch.sin(3 * torch.pi * PopObj[:, 0]) - PopObj[:, 1]) * \
                       (5 - (1 + 0.7 * PopObj[:, 0]) -
                        0.5 * torch.sin(3 * torch.pi * PopObj[:, 0]) - PopObj[:, 1])

        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        R = torch.zeros((N, 2))  # 或者使用 'cpu'，视环境而定
        R[:, 0] = torch.arange(0, 1.5 + 1.5 / (N - 1), 1.5 / (N - 1), device=R.device)
        R[:, 1] = 5 - torch.exp(R[:, 0]) - 0.5 * torch.abs(torch.sin(3 * torch.pi * R[:, 0]))
        c1 = (5 - torch.exp(R[:, 0]) -
              0.5 * torch.sin(3 * torch.pi * R[:, 0]) - R[:, 1]) * \
             (5 - (1 + 0.4 * R[:, 0]) -
              0.5 * torch.sin(3 * torch.pi * R[:, 0]) - R[:, 1])
        invalid = c1 > 0
        while torch.any(invalid):
            R[invalid, :] *= 1.001
            c1 = (5 - torch.exp(R[:, 0]) -
                  0.5 * torch.sin(3 * torch.pi * R[:, 0]) - R[:, 1]) * \
                 (5 - (1 + 0.4 * R[:, 0]) -
                  0.5 * torch.sin(3 * torch.pi * R[:, 0]) - R[:, 1])
            invalid = c1 > 0
        rank = non_dominate_rank(R)
        R = R[rank == 0]
        return R

class MW14(MW):
    def __init__(self, d: int = 15, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        X = 1.5 * X
        g = torch.sum(2 * (X[:, self.m - 1:] + (X[:, self.m - 2:-1] - 0.5) ** 2 - 1) ** 2, dim=1, keepdim=True)

        PopObj = torch.zeros((X.size(0), self.m), device=X.device)
        PopObj[:, :self.m - 1] = X[:, :self.m - 1]
        PopObj[:, self.m - 1] = ((1 + g.squeeze()) / (self.m - 1)) * torch.sum(
            6 - torch.exp(PopObj[:, :self.m - 1]) - 1.5 * torch.sin(1.1 * torch.pi * PopObj[:, :self.m - 1] ** 2),
            dim=1,
            keepdim=True
        ).squeeze()

        a = 1 + PopObj[:, :self.m - 1] + 0.5 * PopObj[:, :self.m - 1] ** 2 + 1.5 * torch.sin(
            1.1 * torch.pi * PopObj[:, :self.m - 1] ** 2)
        PopCon = PopObj[:, self.m - 1].unsqueeze(1) - (1 / (self.m - 1)) * torch.sum(6.1 - a, dim=1, keepdim=True)

        return PopObj, PopCon


    def grid(self, N: int, M: int) -> (torch.Tensor, int):
        gap = np.linspace(0, 1, int(np.ceil(N ** (1.0 / M))))
        mesh = np.meshgrid(*([gap] * M))
        W = np.vstack(list(map(np.ravel, mesh))).T
        N = W.shape[0]
        return torch.tensor(W, dtype=torch.float32), N

    def pf(self) -> torch.Tensor:
        N = 5000
        interval = torch.tensor([0.0, 0.731000, 1.331000, 1.500000])
        median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0])
        if self.m > 2:
            X, _ = self.grid(N, self.m-1)
        else:
            X = torch.linspace(0, 1, N)
        X[X <= median] = X[X <= median] * (interval[1] - interval[0]) / median + interval[0]
        X[X > median] = (X[X > median] - median) * (interval[3] - interval[2]) / (1 - median) + interval[2]
        target_value = (1 / (self.m - 1)) * torch.sum((6 - torch.exp(X) - 1.5 * torch.sin(1.1 * torch.pi * X ** 2)), dim=1, keepdim=True)
        R = torch.cat((X, target_value), dim=1)
        return R
