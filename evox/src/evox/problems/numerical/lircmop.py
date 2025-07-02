import torch

from evox.core import Problem
from evox.operators.sampling import grid_sampling, uniform_sampling
from evox.operators.selection import non_dominate_rank
import numpy as np
import  math

class LIRCMOP(Problem):
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

class LIRCMOP1(LIRCMOP):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> tuple:
        x_odd = X[:, 2::2]
        x_even = X[:, 1::2]
        g_1 = torch.sum((x_odd - torch.sin(0.5 * torch.pi * X[:, 0:1])) ** 2, dim=1, keepdim=True)
        g_2 = torch.sum((x_even - torch.cos(0.5 * torch.pi * X[:, 0:1])) ** 2, dim=1, keepdim=True)
        PopObj = torch.zeros((X.size(0), 2), device=X.device)
        PopObj[:, 0] = X[:, 0] + g_1.squeeze()
        PopObj[:, 1] = 1 - X[:, 0] ** 2 + g_2.squeeze()
        PopCon = torch.zeros((X.size(0), 2), device=X.device)
        PopCon[:, 0] = (0.5 - g_1.squeeze()).clamp(min=0) * (0.51 - g_1.squeeze()).clamp(min=0)
        PopCon[:, 1] = (0.5 - g_2.squeeze()).clamp(min=0) * (0.51 - g_2.squeeze()).clamp(min=0)
        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        R = torch.zeros((N, 2))
        R[:, 0] = torch.linspace(0, 1, N)
        R[:, 1] = 1 - R[:, 0] ** 2
        R += 0.5
        return R

class LIRCMOP2(LIRCMOP):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> tuple:
        x_odd = X[:, 2::2]
        x_even = X[:, 1::2]
        len_odd = x_odd.size(1)
        len_even = x_even.size(1)
        g_1 = torch.sum((x_odd - X[:, 0:1].expand(-1, len_odd)) ** 2, dim=1, keepdim=True)
        g_2 = torch.sum((x_even - X[:, 0:1].expand(-1, len_even)) ** 2, dim=1, keepdim=True)
        PopObj = torch.zeros((X.size(0), 2), device=X.device)
        PopObj[:, 0] = X[:, 0] + g_1.squeeze()
        PopObj[:, 1] = 1 - torch.sqrt(X[:, 0]) + g_2.squeeze()
        PopCon = torch.zeros((X.size(0), 2), device=X.device)
        constraint1 = (0.5 - g_1).clamp(min=0) * (0.51 - g_1).clamp(min=0)
        constraint2 = (0.5 - g_2).clamp(min=0) * (0.51 - g_2).clamp(min=0)
        PopCon[:, 0] = constraint1.squeeze()
        PopCon[:, 1] = constraint2.squeeze()
        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        R = torch.zeros((N, 2), device='cuda' if torch.cuda.is_available() else 'cpu')
        R[:, 0] = torch.linspace(0, 1, N)
        R[:, 1] = 1 - torch.sqrt(R[:, 0])
        R += 0.5
        return R

class LIRCMOP3(LIRCMOP):
    def __init__(self, d: int = 15, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> tuple:
        x_odd = X[:, 2::2]
        x_even = X[:, 1::2]
        len_odd = x_odd.size(1)
        len_even = x_even.size(1)
        g_1 = torch.sum((x_odd - X[:, 0:1].expand(-1, len_odd)) ** 2, dim=1, keepdim=True)  # (N, 1)
        g_2 = torch.sum((x_even - X[:, 0:1].expand(-1, len_even)) ** 2, dim=1, keepdim=True)  # (N, 1)
        PopObj = torch.zeros((X.size(0), 2), device=X.device)
        PopObj[:, 0] = X[:, 0] + g_1.squeeze()
        PopObj[:, 1] = 1 - X[:, 0] ** 2 + g_2.squeeze()
        PopCon = torch.zeros((X.size(0), 3), device=X.device)
        cons1 = (0.5 - g_1).clamp(min=0) * (0.51 - g_1).clamp(min=0)
        cons2 = (0.5 - g_2).clamp(min=0) * (0.51 - g_2).clamp(min=0)
        cons3 = (0.5 - torch.sin(20 * torch.pi * X[:, 0]))
        PopCon[:, 0] = cons1.squeeze()
        PopCon[:, 1] = cons2.squeeze()
        PopCon[:, 2] = cons3.squeeze()
        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        R = torch.zeros((N, 2), device='cuda' if torch.cuda.is_available() else 'cpu')
        R[:, 0] = torch.linspace(0, 1, N)
        R[:, 1] = 1 - R[:, 0] ** 2
        mask = torch.sin(20 * torch.pi * R[:, 0]) >= 0.5
        R = R[mask]
        R += 0.5
        return R

class LIRCMOP4(LIRCMOP):
    def __init__(self, d: int = 15, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> tuple:
        x_odd = X[:, 2::2]
        x_even = X[:, 1::2]
        len_odd = x_odd.size(1)
        len_even = x_even.size(1)
        g_1 = torch.sum((x_odd - X[:, 0:1].expand(-1, len_odd)) ** 2, dim=1, keepdim=True)  # (N, 1)
        g_2 = torch.sum((x_even - X[:, 0:1].expand(-1, len_even)) ** 2, dim=1, keepdim=True)  # (N, 1)
        PopObj = torch.zeros((X.size(0), 2), device=X.device)
        PopObj[:, 0] = X[:, 0] + g_1.squeeze()
        PopObj[:, 1] = 1 - torch.sqrt(X[:, 0]) + g_2.squeeze()
        PopCon = torch.zeros((X.size(0), 3), device=X.device)
        cons1 = (0.5 - g_1).clamp(min=0) * (0.51 - g_1).clamp(min=0)
        cons2 = (0.5 - g_2).clamp(min=0) * (0.51 - g_2).clamp(min=0)
        cons3 = 0.5 - torch.sin(20 * torch.pi * X[:, 0])
        PopCon[:, 0] = cons1.squeeze()
        PopCon[:, 1] = cons2.squeeze()
        PopCon[:, 2] = cons3.squeeze()
        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        R = torch.zeros((N, 2), device='cuda' if torch.cuda.is_available() else 'cpu')
        R[:, 0] = torch.linspace(0, 1, N)
        R[:, 1] = 1 - torch.sqrt(R[:, 0])
        mask = torch.sin(20 * torch.pi * R[:, 0]) >= 0.5
        R = R[mask]
        R += 0.5
        return R


class LIRCMOP5(LIRCMOP):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def constraint_lircmop5(self, PopObj: torch.Tensor) -> torch.Tensor:
        device = PopObj.device
        dtype = PopObj.dtype
        p = torch.tensor([1.6, 2.5], device=device, dtype=dtype)
        q = torch.tensor([1.6, 2.5], device=device, dtype=dtype)
        a = torch.tensor([2.0, 2.0], device=device, dtype=dtype)
        b = torch.tensor([4.0, 8.0], device=device, dtype=dtype)
        r = torch.tensor(0.1, device=device, dtype=dtype)
        theta = torch.tensor(-0.25 * torch.pi, device=device, dtype=dtype)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        dx = PopObj[:, 0:1] - p.unsqueeze(0)
        dy = PopObj[:, 1:2] - q.unsqueeze(0)
        x_rot = dx * cos_theta - dy * sin_theta
        y_rot = dx * sin_theta + dy * cos_theta
        a_squared = a.unsqueeze(0) ** 2
        b_squared = b.unsqueeze(0) ** 2
        ellipse_terms = (x_rot ** 2) / a_squared + (y_rot ** 2) / b_squared
        constraints = r - ellipse_terms
        PopCon = torch.clamp(constraints, min=0.0).max(dim=1, keepdim=True)[0]
        return PopCon

    def evaluate(self, X: torch.Tensor) -> tuple:
        popsize, variable_length = X.shape
        device = X.device
        j_matlab = torch.arange(2, variable_length + 1, device=device,
                                dtype=torch.float32)
        j_python = j_matlab - 1
        angles = 0.5 * j_matlab / variable_length * torch.pi
        x1 = X[:, 0:1]  # shape: (popsize, 1)
        angle_terms = angles.unsqueeze(0) * x1
        sin_terms = torch.sin(angle_terms)
        cos_terms = torch.cos(angle_terms)
        X_vars = X[:, 1:]
        diff_sin = (X_vars - sin_terms) ** 2
        diff_cos = (X_vars - cos_terms) ** 2
        odd_mask = (j_matlab % 2 == 1).unsqueeze(0)
        even_mask = (j_matlab % 2 == 0).unsqueeze(0)
        sum1 = torch.sum(diff_sin * odd_mask, dim=1, keepdim=True)
        sum2 = torch.sum(diff_cos * even_mask, dim=1, keepdim=True)
        gx = 0.7057
        PopObj1 = X[:, 0:1] + 10 * sum1 + gx
        PopObj2 = 1 - torch.pow(X[:, 0:1], 0.5) + 10 * sum2 + gx
        PopObj = torch.cat([PopObj1, PopObj2], dim=1)
        PopCon = self.constraint_lircmop5(PopObj)
        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        device = torch.device('cuda')
        dtype = torch.float32
        R1 = torch.linspace(0, 1, N, device=device, dtype=dtype).unsqueeze(1)
        R2 = 1 - torch.sqrt(R1)
        R = torch.cat([R1, R2], dim=1)
        R = R + 0.7057
        constraints = self.constraint_lircmop5(R)
        if constraints.dim() == 2 and constraints.size(1) > 1:
            constraint_violated = torch.any(constraints > 0, dim=1)
        else:
            constraint_violated = (constraints.squeeze() > 0)
        feasible_mask = ~constraint_violated
        R = R[feasible_mask]
        return R

class LIRCMOP6(LIRCMOP):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def constraint_lircmop6(self, PopObj: torch.Tensor) -> torch.Tensor:
        device = PopObj.device
        dtype = PopObj.dtype
        p = torch.tensor([1.8, 2.8], device=device, dtype=dtype)
        q = torch.tensor([1.8, 2.8], device=device, dtype=dtype)
        a = torch.tensor([2.0, 2.0], device=device, dtype=dtype)
        b = torch.tensor([8.0, 8.0], device=device, dtype=dtype)
        r = 0.1
        theta = torch.tensor(-0.25 * torch.pi, device=device, dtype=dtype)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        dx = PopObj[:, 0:1] - p.unsqueeze(0)
        dy = PopObj[:, 1:2] - q.unsqueeze(0)
        x_rot = dx * cos_theta - dy * sin_theta
        y_rot = dx * sin_theta + dy * cos_theta
        a_squared = a.unsqueeze(0) ** 2
        b_squared = b.unsqueeze(0) ** 2
        ellipse_terms = (x_rot ** 2) / a_squared + (y_rot ** 2) / b_squared
        constraints = r - ellipse_terms
        PopCon = torch.clamp(constraints, min=0.0).max(dim=1, keepdim=True)[0]
        return PopCon

    def evaluate(self, X: torch.Tensor) -> tuple:
        popsize, variable_length = X.shape
        device = X.device
        dtype = X.dtype
        j_matlab = torch.arange(2, variable_length + 1, device=device,
                                dtype=dtype)
        angles = 0.5 * j_matlab / variable_length * torch.pi
        x1 = X[:, 0:1]
        angle_terms = angles.unsqueeze(0) * x1
        sin_terms = torch.sin(angle_terms)
        cos_terms = torch.cos(angle_terms)
        X_vars = X[:, 1:]
        diff_sin = (X_vars - sin_terms) ** 2
        diff_cos = (X_vars - cos_terms) ** 2
        odd_mask = (j_matlab % 2 == 1).unsqueeze(0)
        even_mask = (j_matlab % 2 == 0).unsqueeze(0)
        sum1 = torch.sum(diff_sin * odd_mask, dim=1, keepdim=True)
        sum2 = torch.sum(diff_cos * even_mask, dim=1, keepdim=True)
        gx = 0.7057
        PopObj1 = X[:, 0:1] + 10 * sum1 + gx
        PopObj2 = 1 - torch.pow(X[:, 0:1], 2) + 10 * sum2 + gx
        PopObj = torch.cat([PopObj1, PopObj2], dim=1)
        PopCon = self.constraint_lircmop6(PopObj)
        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        device = self.device
        dtype = torch.float32
        R1 = torch.linspace(0, 1, N, device=device, dtype=dtype).reshape(-1, 1)  # shape: (N, 1)
        R2 = 1 - R1 ** 2  # shape: (N, 1)
        R = torch.cat([R1, R2], dim=1)  # shape: (N, 2)
        R = R + 0.7057
        constraints = self.constraint_lircmop6(R)
        if constraints.dim() == 2:
            if constraints.size(1) > 1:
                constraint_violated = torch.any(constraints > 0, dim=1)
            else:
                constraint_violated = constraints.squeeze(1) > 0
        else:
            constraint_violated = constraints > 0
        feasible_mask = ~constraint_violated
        R = R[feasible_mask]
        return R

class LIRCMOP7(LIRCMOP):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def constraint_lircmop7(self, PopObj: torch.Tensor) -> torch.Tensor:
        device = PopObj.device
        dtype = PopObj.dtype
        p = torch.tensor([1.2, 2.25, 3.5], device=device, dtype=dtype)
        q = torch.tensor([1.2, 2.25, 3.5], device=device, dtype=dtype)
        a = torch.tensor([2.0, 2.5, 2.5], device=device, dtype=dtype)
        b = torch.tensor([6.0, 12.0, 10.0], device=device, dtype=dtype)
        r = 0.1
        theta = torch.tensor(-0.25 * torch.pi, device=device, dtype=dtype)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        dx = PopObj[:, 0:1] - p.unsqueeze(0)
        dy = PopObj[:, 1:2] - q.unsqueeze(0)
        x_rot = dx * cos_theta - dy * sin_theta
        y_rot = dx * sin_theta + dy * cos_theta
        a_squared = a.unsqueeze(0) ** 2
        b_squared = b.unsqueeze(0) ** 2
        ellipse_terms = (x_rot ** 2) / a_squared + (y_rot ** 2) / b_squared
        constraints = r - ellipse_terms
        PopCon = torch.clamp(constraints, min=0.0).max(dim=1, keepdim=True)[0]
        return PopCon

    def evaluate(self, X: torch.Tensor) -> tuple:
        popsize, variable_length = X.shape
        device = X.device
        dtype = X.dtype
        j_matlab = torch.arange(2, variable_length + 1, device=device, dtype=dtype)
        angles = 0.5 * j_matlab / variable_length * torch.pi
        x1 = X[:, 0:1]
        angle_terms = angles.unsqueeze(0) * x1
        sin_terms = torch.sin(angle_terms)
        cos_terms = torch.cos(angle_terms)
        X_vars = X[:, 1:]
        diff_sin = (X_vars - sin_terms) ** 2
        diff_cos = (X_vars - cos_terms) ** 2
        odd_mask = (j_matlab % 2 == 1).unsqueeze(0)
        even_mask = (j_matlab % 2 == 0).unsqueeze(0)
        sum1 = torch.sum(diff_sin * odd_mask, dim=1, keepdim=True)
        sum2 = torch.sum(diff_cos * even_mask, dim=1, keepdim=True)
        gx = 0.7057
        PopObj1 = X[:, 0:1] + 10 * sum1 + gx
        PopObj2 = 1 - torch.sqrt(X[:, 0:1]) + 10 * sum2 + gx
        PopObj = torch.cat([PopObj1, PopObj2], dim=1)
        PopCon = self.constraint_lircmop7(PopObj)
        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        device = self.device
        dtype = torch.float32
        R1 = torch.linspace(0, 1, N, device=device, dtype=dtype).reshape(-1, 1)
        R2 = 1 - torch.sqrt(R1)
        R = torch.cat([R1, R2], dim=1) + 0.7057
        theta = torch.tensor(-0.25 * torch.pi, device=device, dtype=dtype)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        dx = R[:, 0] - 1.2
        dy = R[:, 1] - 1.2
        term1 = dx * cos_theta - dy * sin_theta
        term2 = dx * sin_theta + dy * cos_theta
        c1 = 0.1 - (term1 ** 2) / (2 ** 2) - (term2 ** 2) / (6 ** 2)
        invalid = c1 > 0
        max_iterations = 1000
        iteration = 0
        while invalid.any() and iteration < max_iterations:
            R[invalid] = (R[invalid] - 0.7057) * 1.001 + 0.7057
            dx = R[:, 0] - 1.2
            dy = R[:, 1] - 1.2
            term1 = dx * cos_theta - dy * sin_theta
            term2 = dx * sin_theta + dy * cos_theta
            c1 = 0.1 - (term1 ** 2) / (2 ** 2) - (term2 ** 2) / (6 ** 2)
            invalid = c1 > 0
            iteration += 1
        return R

class LIRCMOP8(LIRCMOP):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def constraint_lircmop7(self, PopObj: torch.Tensor) -> torch.Tensor:
        device = PopObj.device
        dtype = PopObj.dtype
        p = torch.tensor([1.2, 2.25, 3.5], device=device, dtype=dtype)
        q = torch.tensor([1.2, 2.25, 3.5], device=device, dtype=dtype)
        a = torch.tensor([2.0, 2.5, 2.5], device=device, dtype=dtype)
        b = torch.tensor([6.0, 12.0, 10.0], device=device, dtype=dtype)
        r = 0.1
        theta = torch.tensor(-0.25 * torch.pi, device=device, dtype=dtype)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        dx = PopObj[:, 0:1] - p.unsqueeze(0)
        dy = PopObj[:, 1:2] - q.unsqueeze(0)
        x_rot = dx * cos_theta - dy * sin_theta
        y_rot = dx * sin_theta + dy * cos_theta
        a_squared = a.unsqueeze(0) ** 2
        b_squared = b.unsqueeze(0) ** 2
        ellipse_terms = (x_rot ** 2) / a_squared + (y_rot ** 2) / b_squared
        constraints = r - ellipse_terms
        PopCon = torch.clamp(constraints, min=0.0).max(dim=1, keepdim=True)[0]
        return PopCon

    def evaluate(self, X: torch.Tensor) -> tuple:
        popsize, variable_length = X.shape
        device = X.device
        dtype = X.dtype
        j_matlab = torch.arange(2, variable_length + 1, device=device, dtype=dtype)
        angles = 0.5 * j_matlab / variable_length * torch.pi
        x1 = X[:, 0:1]
        angle_terms = angles.unsqueeze(0) * x1
        sin_terms = torch.sin(angle_terms)
        cos_terms = torch.cos(angle_terms)
        X_vars = X[:, 1:]
        diff_sin = (X_vars - sin_terms) ** 2
        diff_cos = (X_vars - cos_terms) ** 2
        odd_mask = (j_matlab % 2 == 1).unsqueeze(0)
        even_mask = (j_matlab % 2 == 0).unsqueeze(0)
        sum1 = torch.sum(diff_sin * odd_mask, dim=1, keepdim=True)
        sum2 = torch.sum(diff_cos * even_mask, dim=1, keepdim=True)
        gx = 0.7057
        PopObj1 = X[:, 0:1] + 10 * sum1 + gx
        PopObj2 = 1 - X[:, 0:1]**2 + 10 * sum2 + gx
        PopObj = torch.cat([PopObj1, PopObj2], dim=1)
        PopCon = self.constraint_lircmop7(PopObj)
        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        device = self.device
        dtype = torch.float32
        R1 = torch.linspace(0, 1, N, device=device, dtype=dtype).reshape(-1, 1)
        R2 = 1 - torch.sqrt(R1)
        R = torch.cat([R1, R2], dim=1) + 0.7057
        theta = torch.tensor(-0.25 * torch.pi, device=device, dtype=dtype)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        dx = R[:, 0] - 1.2
        dy = R[:, 1] - 1.2
        term1 = dx * cos_theta - dy * sin_theta
        term2 = dx * sin_theta + dy * cos_theta
        c1 = 0.1 - (term1 ** 2) / (2 ** 2) - (term2 ** 2) / (6 ** 2)
        invalid = c1 > 0
        max_iterations = 1000  # 防止无限循环
        iteration = 0
        while invalid.any() and iteration < max_iterations:
            R[invalid] = (R[invalid] - 0.7057) * 1.001 + 0.7057
            dx = R[:, 0] - 1.2
            dy = R[:, 1] - 1.2
            term1 = dx * cos_theta - dy * sin_theta
            term2 = dx * sin_theta + dy * cos_theta
            c1 = 0.1 - (term1 ** 2) / (2 ** 2) - (term2 ** 2) / (6 ** 2)
            invalid = c1 > 0
            iteration += 1
        return R

class LIRCMOP9(LIRCMOP):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def constraint_lircmop9(self, PopObj: torch.Tensor) -> torch.Tensor:
        device = PopObj.device
        dtype = PopObj.dtype
        p, q = 1.4, 1.4
        a, b = 1.5, 6.0
        r = 0.1
        theta = -0.25 * math.pi
        alpha = 0.25 * math.pi
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        cos_alpha = math.cos(alpha)
        sin_alpha = math.sin(alpha)
        dx = PopObj[:, 0] - p
        dy = PopObj[:, 1] - q
        x_rot = dx * cos_theta - dy * sin_theta
        y_rot = dx * sin_theta + dy * cos_theta
        constraint1 = r - (x_rot ** 2) / (a ** 2) - (y_rot ** 2) / (b ** 2)
        linear_term = PopObj[:, 0] * sin_alpha + PopObj[:, 1] * cos_alpha
        rotation_term = PopObj[:, 0] * cos_alpha - PopObj[:, 1] * sin_alpha
        constraint2 = 2 - linear_term + torch.sin(4 * math.pi * rotation_term)
        constraints = torch.stack([constraint1, constraint2], dim=1)  # (popsize, 2)
        PopCon = torch.clamp(constraints, min=0.0).max(dim=1, keepdim=True)[0]
        return PopCon

    def evaluate(self, X: torch.Tensor) -> tuple:
        popsize, variable_length = X.shape
        device = X.device
        dtype = X.dtype
        j_matlab = torch.arange(2, variable_length + 1, device=device, dtype=dtype)
        angles = 0.5 * j_matlab / variable_length * torch.pi
        x1 = X[:, 0:1]
        angle_terms = angles.unsqueeze(0) * x1
        sin_terms = torch.sin(angle_terms)
        cos_terms = torch.cos(angle_terms)
        X_vars = X[:, 1:]
        diff_sin = (X_vars - sin_terms) ** 2
        diff_cos = (X_vars - cos_terms) ** 2
        odd_mask = (j_matlab % 2 == 1).unsqueeze(0)
        even_mask = (j_matlab % 2 == 0).unsqueeze(0)
        sum1 = torch.sum(diff_sin * odd_mask, dim=1, keepdim=True)
        sum2 = torch.sum(diff_cos * even_mask, dim=1, keepdim=True)
        PopObj1 = 1.7057 * X[:, 0:1] * (10 * sum1 + 1)
        PopObj2 = 1.7057 * (1 - X[:, 0:1] ** 2) * (10 * sum2 + 1)
        PopObj = torch.cat([PopObj1, PopObj2], dim=1)
        PopCon = self.constraint_lircmop9(PopObj)
        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        device = self.device
        dtype = torch.float32
        R1 = torch.linspace(0, 1, N, device=device, dtype=dtype).reshape(-1, 1)
        R2 = 1 - R1 ** 2
        R = torch.cat([R1, R2], dim=1) * 1.7057
        constraints = self.constraint_lircmop9(R)
        if constraints.dim() == 2:
            if constraints.size(1) > 1:
                constraint_violated = torch.any(constraints > 0, dim=1)
            else:
                constraint_violated = constraints.squeeze(1) > 0
        else:
            constraint_violated = constraints > 0
        feasible_mask = ~constraint_violated
        R = R[feasible_mask]
        special_points = torch.tensor([[0, 2.182], [1.856, 0]], device=device, dtype=dtype)
        R = torch.cat([R, special_points], dim=0)

        return R

class LIRCMOP10(LIRCMOP):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def constraint_lircmop10(self, PopObj: torch.Tensor) -> torch.Tensor:
        device = PopObj.device
        dtype = PopObj.dtype
        p, q = 1.1, 1.2
        a, b = 2.0, 4.0
        r = 0.1
        theta = -0.25 * math.pi
        alpha = 0.25 * math.pi

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        cos_alpha = math.cos(alpha)
        sin_alpha = math.sin(alpha)
        dx = PopObj[:, 0] - p
        dy = PopObj[:, 1] - q
        x_rot = dx * cos_theta - dy * sin_theta
        y_rot = dx * sin_theta + dy * cos_theta
        constraint1 = r - (x_rot ** 2) / (a ** 2) - (y_rot ** 2) / (b ** 2)
        linear_term = PopObj[:, 0] * sin_alpha + PopObj[:, 1] * cos_alpha
        rotation_term = PopObj[:, 0] * cos_alpha - PopObj[:, 1] * sin_alpha
        constraint2 = 1 - linear_term + torch.sin(4 * math.pi * rotation_term)
        constraints = torch.stack([constraint1, constraint2], dim=1)  # (popsize, 2)
        PopCon = torch.clamp(constraints, min=0.0).max(dim=1, keepdim=True)[0]
        return PopCon

    def evaluate(self, X: torch.Tensor) -> tuple:
        """LIRCMOP9 评估函数"""
        popsize, variable_length = X.shape
        device = X.device
        dtype = X.dtype
        j_matlab = torch.arange(2, variable_length + 1, device=device, dtype=dtype)
        angles = 0.5 * j_matlab / variable_length * torch.pi
        x1 = X[:, 0:1]
        angle_terms = angles.unsqueeze(0) * x1
        sin_terms = torch.sin(angle_terms)
        cos_terms = torch.cos(angle_terms)
        X_vars = X[:, 1:]
        diff_sin = (X_vars - sin_terms) ** 2
        diff_cos = (X_vars - cos_terms) ** 2
        odd_mask = (j_matlab % 2 == 1).unsqueeze(0)
        even_mask = (j_matlab % 2 == 0).unsqueeze(0)
        sum1 = torch.sum(diff_sin * odd_mask, dim=1, keepdim=True)
        sum2 = torch.sum(diff_cos * even_mask, dim=1, keepdim=True)
        PopObj1 = 1.7057 * X[:, 0:1] * (10 * sum1 + 1)
        PopObj2 = 1.7057 * (1 - X[:, 0:1] ** 0.5) * (10 * sum2 + 1)
        PopObj = torch.cat([PopObj1, PopObj2], dim=1)
        PopCon = self.constraint_lircmop10(PopObj)
        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        device = self.device
        dtype = torch.float32
        R1 = torch.linspace(0, 1, N, device=device, dtype=dtype).reshape(-1, 1)
        R2 = 1 - torch.sqrt(R1)
        R = torch.cat([R1, R2], dim=1) * 1.7057
        constraints = self.constraint_lircmop10(R)
        if constraints.dim() == 2:
            if constraints.size(1) > 1:
                constraint_violated = torch.any(constraints > 0, dim=1)
            else:
                constraint_violated = constraints.squeeze(1) > 0
        else:
            constraint_violated = constraints > 0
        feasible_mask = ~constraint_violated
        R = R[feasible_mask]
        special_points = torch.tensor([[1.747, 0]], device=device, dtype=dtype)
        R = torch.cat([R, special_points], dim=0)
        return R

class LIRCMOP11(LIRCMOP):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def constraint_lircmop11(self, PopObj: torch.Tensor) -> torch.Tensor:
        p, q = 1.2, 1.2
        a, b = 1.5, 5.0
        r = 0.1
        theta = -0.25 * math.pi
        alpha = 0.25 * math.pi
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        cos_alpha = math.cos(alpha)
        sin_alpha = math.sin(alpha)
        dx = PopObj[:, 0] - p
        dy = PopObj[:, 1] - q
        x_rot = dx * cos_theta - dy * sin_theta
        y_rot = dx * sin_theta + dy * cos_theta
        constraint1 = r - (x_rot ** 2) / (a ** 2) - (y_rot ** 2) / (b ** 2)
        linear_term = PopObj[:, 0] * sin_alpha + PopObj[:, 1] * cos_alpha
        rotation_term = PopObj[:, 0] * cos_alpha - PopObj[:, 1] * sin_alpha
        constraint2 = 2.1 - linear_term + torch.sin(4 * math.pi * rotation_term)
        constraints = torch.stack([constraint1, constraint2], dim=1)
        PopCon = torch.clamp(constraints, min=0.0).max(dim=1, keepdim=True)[0]
        return PopCon

    def evaluate(self, X: torch.Tensor) -> tuple:
        popsize, variable_length = X.shape
        device = X.device
        dtype = X.dtype
        j_matlab = torch.arange(2, variable_length + 1, device=device, dtype=dtype)
        angles = 0.5 * j_matlab / variable_length * torch.pi
        x1 = X[:, 0:1]
        angle_terms = angles.unsqueeze(0) * x1
        sin_terms = torch.sin(angle_terms)
        cos_terms = torch.cos(angle_terms)
        X_vars = X[:, 1:]
        diff_sin = (X_vars - sin_terms) ** 2
        diff_cos = (X_vars - cos_terms) ** 2
        odd_mask = (j_matlab % 2 == 1).unsqueeze(0)
        even_mask = (j_matlab % 2 == 0).unsqueeze(0)
        sum1 = torch.sum(diff_sin * odd_mask, dim=1, keepdim=True)
        sum2 = torch.sum(diff_cos * even_mask, dim=1, keepdim=True)
        PopObj1 = 1.7057 * X[:, 0:1] * (10 * sum1 + 1)
        PopObj2 = 1.7057 * (1 - torch.sqrt(X[:, 0:1])) * (10 * sum2 + 1)
        PopObj = torch.cat([PopObj1, PopObj2], dim=1)
        PopCon = self.constraint_lircmop11(PopObj)
        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        device = self.device
        dtype = torch.float32
        R = torch.tensor([
            [1.3965, 0.1591],
            [1.0430, 0.5127],
            [0.6894, 0.8662],
            [0.3359, 1.2198],
            [0.0106, 1.6016],
            [0, 2.1910],
            [1.8730, 0]
        ], device=device, dtype=dtype)
        return R


class LIRCMOP12(LIRCMOP):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def constraint_lircmop12(self, PopObj: torch.Tensor) -> torch.Tensor:
        p, q = 1.6, 1.6
        a, b = 1.5, 6.0
        r = 0.1
        theta = -0.25 * math.pi
        alpha = 0.25 * math.pi
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        cos_alpha = math.cos(alpha)
        sin_alpha = math.sin(alpha)
        dx = PopObj[:, 0] - p
        dy = PopObj[:, 1] - q
        x_rot = dx * cos_theta - dy * sin_theta
        y_rot = dx * sin_theta + dy * cos_theta
        constraint1 = r - (x_rot ** 2) / (a ** 2) - (y_rot ** 2) / (b ** 2)
        linear_term = PopObj[:, 0] * sin_alpha + PopObj[:, 1] * cos_alpha
        rotation_term = PopObj[:, 0] * cos_alpha - PopObj[:, 1] * sin_alpha
        constraint2 = 2.5 - linear_term + torch.sin(4 * math.pi * rotation_term)
        constraints = torch.stack([constraint1, constraint2], dim=1)
        PopCon = torch.clamp(constraints, min=0.0).max(dim=1, keepdim=True)[0]
        return PopCon

    def evaluate(self, X: torch.Tensor) -> tuple:
        popsize, variable_length = X.shape
        device = X.device
        dtype = X.dtype
        j_matlab = torch.arange(2, variable_length + 1, device=device, dtype=dtype)
        angles = 0.5 * j_matlab / variable_length * torch.pi
        x1 = X[:, 0:1]
        angle_terms = angles.unsqueeze(0) * x1
        sin_terms = torch.sin(angle_terms)
        cos_terms = torch.cos(angle_terms)
        X_vars = X[:, 1:]
        diff_sin = (X_vars - sin_terms) ** 2
        diff_cos = (X_vars - cos_terms) ** 2
        odd_mask = (j_matlab % 2 == 1).unsqueeze(0)
        even_mask = (j_matlab % 2 == 0).unsqueeze(0)
        sum1 = torch.sum(diff_sin * odd_mask, dim=1, keepdim=True)
        sum2 = torch.sum(diff_cos * even_mask, dim=1, keepdim=True)
        PopObj1 = 1.7057 * X[:, 0:1] * (10 * sum1 + 1)
        PopObj2 = 1.7057 * (1 - torch.sqrt(X[:, 0:1])) * (10 * sum2 + 1)
        PopObj = torch.cat([PopObj1, PopObj2], dim=1)
        PopCon = self.constraint_lircmop12(PopObj)
        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        N = 5000
        device = self.device
        dtype = torch.float32
        R = torch.tensor([
            [1.6794, 0.4419],
            [1.3258, 0.7955],
            [0.9723, 1.1490],
            [2.0320, 0.0990],
            [0.6187, 1.5026],
            [0.2652, 1.8562],
            [0, 2.2580],
            [2.5690, 0]
        ], device=device, dtype=dtype)
        return R

class LIRCMOP13(LIRCMOP):
    def __init__(self, d: int = 30, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> tuple:
        popsize, variable_length = X.shape
        device = X.device
        dtype = X.dtype
        if variable_length > 2:
            sum1 = torch.sum(10 * (X[:, 2:] - 0.5) ** 2, dim=1, keepdim=True)
        else:
            sum1 = torch.zeros((popsize, 1), device=device, dtype=dtype)
        base_term = 1.7057 + sum1
        cos_x1 = torch.cos(0.5 * math.pi * X[:, 0:1])
        sin_x1 = torch.sin(0.5 * math.pi * X[:, 0:1])
        cos_x2 = torch.cos(0.5 * math.pi * X[:, 1:2])
        sin_x2 = torch.sin(0.5 * math.pi * X[:, 1:2])
        PopObj1 = base_term * cos_x1 * cos_x2
        PopObj2 = base_term * cos_x1 * sin_x2
        PopObj3 = base_term * sin_x1
        PopObj = torch.cat([PopObj1, PopObj2, PopObj3], dim=1)
        gx = torch.sum(PopObj ** 2, dim=1, keepdim=True)
        constraint1 = (gx - 9) * (4 - gx)
        constraint2 = (gx - 3.61) * (3.24 - gx)
        constraints = torch.cat([constraint1, constraint2], dim=1)
        PopCon = torch.clamp(constraints, min=0.0).max(dim=1, keepdim=True)[0]
        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        R = self.sample*2
        R_norm = torch.sqrt(torch.sum(R ** 2, dim=1, keepdim=True))
        R = R / R_norm
        R = 1.7057 * R
        return R

class LIRCMOP14(LIRCMOP):
    def __init__(self, d: int = 30, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> tuple:
        popsize, variable_length = X.shape
        device = X.device
        dtype = X.dtype
        if variable_length > 2:
            sum1 = torch.sum(10 * (X[:, 2:] - 0.5) ** 2, dim=1, keepdim=True)
        else:
            sum1 = torch.zeros((popsize, 1), device=device, dtype=dtype)
        base_term = 1.7057 + sum1
        cos_x1 = torch.cos(0.5 * math.pi * X[:, 0:1])
        sin_x1 = torch.sin(0.5 * math.pi * X[:, 0:1])
        cos_x2 = torch.cos(0.5 * math.pi * X[:, 1:2])
        sin_x2 = torch.sin(0.5 * math.pi * X[:, 1:2])
        PopObj1 = base_term * cos_x1 * cos_x2
        PopObj2 = base_term * cos_x1 * sin_x2
        PopObj3 = base_term * sin_x1
        PopObj = torch.cat([PopObj1, PopObj2, PopObj3], dim=1)
        gx = torch.sum(PopObj ** 2, dim=1, keepdim=True)
        constraint1 = (gx - 9) * (4 - gx)
        constraint2 = (gx - 3.61) * (3.24 - gx)
        constraint2 = (gx - 3.0625) * (2.56 - gx)
        constraints = torch.cat([constraint1, constraint2], dim=1)
        PopCon = torch.clamp(constraints, min=0.0).max(dim=1, keepdim=True)[0]
        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        R = self.sample*2
        R_norm = torch.sqrt(torch.sum(R ** 2, dim=1, keepdim=True))
        R = R / R_norm
        R = math.sqrt(3.0625) * R
        return R