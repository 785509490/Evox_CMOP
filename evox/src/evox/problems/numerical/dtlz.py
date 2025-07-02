import torch

from evox.core import Problem
from evox.operators.sampling import grid_sampling, uniform_sampling
import numpy as np

class DTLZ(Problem):
    """
    Base class for DTLZ test suite problems in multi-objective optimization.

    Inherit this class to implement specific DTLZ problem variants.

    :param d: Number of decision variables.
    :param m: Number of objectives.
    :param ref_num: Number of reference points used in the problem.
    """

    def __init__(self, d: int = None, m: int = None, ref_num: int = 1000):
        """Override the setup method to initialize the parameters"""
        super().__init__()
        self.d = d
        self.m = m
        self.ref_num = ref_num
        self.sample, _ = uniform_sampling(self.ref_num * self.m, self.m)  # Assuming UniformSampling is defined
        self.device = self.sample.device

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to evaluate the objective values for given decision variables.

        :param X: A tensor of shape (n, d), where n is the number of solutions and d is the number of decision variables.
        :return: A tensor of shape (n, m) representing the objective values for each solution.
        """
        raise NotImplementedError()

    def pf(self):
        """
        Return the Pareto front for the problem.

        :return: A tensor representing the Pareto front.
        """
        f = self.sample / 2
        return f


class DTLZ1(DTLZ):
    def __init__(self, d: int = 7, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.size()
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        flip_cumprod = torch.flip(
            torch.cumprod(
                torch.cat([torch.ones((n, 1), device=X.device), X[:, : m - 1]], dim=1),
                dim=1,
            ),
            dims=[1],
        )
        rest_part = torch.cat(
            [
                torch.ones((n, 1), device=X.device),
                1 - torch.flip(X[:, : m - 1], dims=[1]),
            ],
            dim=1,
        )
        f = 0.5 * (1 + g) * flip_cumprod * rest_part
        return f


class DTLZ2(DTLZ):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        g = torch.sum((X[:, m - 1 :] - 0.5) ** 2, dim=1, keepdim=True)
        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((X.size(0), 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((X.size(0), 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )

        return f

    def pf(self):
        f = self.sample
        f = f / torch.sqrt(f.pow(2).sum(dim=1, keepdim=True))
        return f


class DTLZ3(DTLZ2):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.size()
        m = self.m
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((n, 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((n, 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )
        return f


class DTLZ4(DTLZ2):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m

        Xfront = X[:, : m - 1].pow(100)
        Xrear = X[:, m - 1:].clone()
        # X[:, : m - 1] = X[:, : m - 1].pow(100)

        g = torch.sum((Xrear - 0.5) ** 2, dim=1, keepdim=True)

        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((g.size(0), 1), device=X.device),
                            torch.maximum(
                                torch.cos(Xfront * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((g.size(0), 1), device=X.device),
                    torch.sin(torch.flip(Xfront, dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )

        return f


class DTLZ5(DTLZ):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m

        g = torch.sum((X[:, m - 1 :] - 0.5) ** 2, dim=1, keepdim=True)
        temp = g.repeat(1, m - 2)

        Xfront = X[:, : m - 1].clone()
        Xfront[:, 1:] = (1 + 2 * temp * Xfront[:, 1:]) / (2 + 2 * temp)

        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((g.size(0), 1), device=X.device),
                            torch.maximum(
                                torch.cos(Xfront * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((g.size(0), 1), device=X.device),
                    torch.sin(torch.flip(Xfront, dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )

        return f

    def pf(self):
        n = self.ref_num * self.m

        f = torch.vstack(
            (
                torch.hstack(
                    (
                        torch.arange(0, 1, 1.0 / (n - 1), device=self.device),
                        torch.tensor(1.0, device=self.device),
                    )
                ),
                torch.hstack(
                    (
                        torch.arange(1, 0, -1.0 / (n - 1), device=self.device),
                        torch.tensor(0.0, device=self.device),
                    )
                ),
            )
        ).T

        f = f / torch.tile(torch.sqrt(torch.sum(f**2, dim=1, keepdim=True)), (1, f.size(1)))

        for i in range(self.m - 2):
            f = torch.cat((f[:, 0:1], f), dim=1)

        f = f / torch.sqrt(torch.tensor(2.0, device=self.device)) ** torch.tile(
            torch.hstack(
                (
                    torch.tensor(self.m - 2, device=self.device),
                    torch.arange(self.m - 2, -1, -1, device=self.device),
                )
            ),
            (f.size(0), 1),
        )
        return f


class DTLZ6(DTLZ):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        g = torch.sum((X[:, m - 1 :] ** 0.1), dim=1, keepdim=True)
        temp = torch.tile(g, (1, m - 2))
        Xfront = X[:, : m - 1].clone()
        Xfront[:, 1:] = (1 + 2 * temp * Xfront[:, 1:]) / (2 + 2 * temp)

        f = (
            torch.tile(1 + g, (1, m))
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((X.size(0), 1), device=X.device),
                            torch.maximum(
                                torch.cos(Xfront * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((X.size(0), 1), device=X.device),
                    torch.sin(torch.flip(Xfront, dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )
        return f

    def pf(self):
        n = self.ref_num * self.m

        # Ensure the tensor is created on the same device (use X.device if needed)
        f = torch.vstack(
            (
                torch.hstack(
                    (
                        torch.arange(0, 1, 1.0 / (n - 1), device=self.device),
                        torch.tensor(1.0, device=self.device),
                    )
                ),
                torch.hstack(
                    (
                        torch.arange(1, 0, -1.0 / (n - 1), device=self.device),
                        torch.tensor(0.0, device=self.device),
                    )
                ),
            )
        ).T

        f = f / torch.tile(torch.sqrt(torch.sum(f**2, dim=1, keepdim=True)), (1, f.size(1)))

        for i in range(self.m - 2):
            f = torch.cat((f[:, 0:1], f), dim=1)

        f = f / torch.sqrt(torch.tensor(2.0, device=self.device)) ** torch.tile(
            torch.hstack(
                (
                    torch.tensor(self.m - 2, device=self.device),
                    torch.arange(self.m - 2, -1, -1, device=self.device),
                )
            ),
            (f.size(0), 1),
        )
        return f


class DTLZ7(DTLZ):
    def __init__(self, d: int = 21, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)
        self.sample, _ = grid_sampling(self.ref_num * self.m, self.m - 1)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.size()
        m = self.m
        f = torch.zeros((n, m), device=X.device)
        g = 1 + 9 * torch.mean(X[:, m - 1 :], dim=1, keepdim=True)

        term = torch.sum(
            X[:, : m - 1] / (1 + torch.tile(g, (1, m - 1))) * (1 + torch.sin(3 * torch.pi * X[:, : m - 1])),
            dim=1,
            keepdim=True,
        )
        f = torch.cat([X[:, : m - 1].clone(), (1 + g) * (m - term)], dim=1)

        return f

    def pf(self):
        interval = torch.tensor([0.0, 0.251412, 0.631627, 0.859401], dtype=torch.float, device=self.device)
        median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0]).to(self.device)

        x = self.sample.to(self.device)

        mask_less_equal_median = x <= median
        mask_greater_median = x > median

        x = torch.where(
            mask_less_equal_median,
            x * (interval[1] - interval[0]) / median + interval[0],
            x,
        )
        x = torch.where(
            mask_greater_median,
            (x - median) * (interval[3] - interval[2]) / (1 - median) + interval[2],
            x,
        )

        last_col = 2 * (self.m - torch.sum(x / 2 * (1 + torch.sin(3 * torch.pi * x)), dim=1, keepdim=True))

        pf = torch.cat([x, last_col], dim=1)
        return pf


class C1_DTLZ1(DTLZ):
    def __init__(self, d: int = 7, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> tuple:
        m = self.m
        n, d = X.size()
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        flip_cumprod = torch.flip(
            torch.cumprod(
                torch.cat([torch.ones((n, 1), device=X.device), X[:, : m - 1]], dim=1),
                dim=1,
            ),
            dims=[1],
        )
        rest_part = torch.cat(
            [
                torch.ones((n, 1), device=X.device),
                1 - torch.flip(X[:, : m - 1], dims=[1]),
            ],
            dim=1,
        )
        f = 0.5 * (1 + g) * flip_cumprod * rest_part
        # Calculate constraints
        PopCon = (f[:, -1].unsqueeze(1) / 0.6) + (torch.sum(f[:, :-1] / 0.5, dim=1, keepdim=True)) - 1
        return f, PopCon


class C2_DTLZ2(DTLZ):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        g = torch.sum((X[:, m - 1 :] - 0.5) ** 2, dim=1, keepdim=True)
        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((X.size(0), 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((X.size(0), 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )
        PopObj = f
        # 约束计算
        if m == 3:
            r = 0.4
        else:
            r = 0.5

        # 约束条件1：(f - 1)^2 + sum(f^2) - f^2 <= r^2
        constraint1 = torch.min(
            (PopObj - 1) ** 2 + torch.sum(PopObj ** 2, dim=1, keepdim=True) - PopObj ** 2 - r ** 2,
            dim=1,
            keepdim=True
        )[0]

        # 约束条件2：sum((f - 1/sqrt(m))^2) <= r^2
        constraint2 = torch.sum((PopObj - 1 / torch.sqrt(torch.tensor(m))) ** 2, dim=1, keepdim=True) - r ** 2

        # 最终约束值：取两个约束条件的最小值
        PopCon = torch.min(constraint1, constraint2)
        return PopObj, PopCon

    def GetOptimum(self):
        R = self.sample
        R = R / np.sqrt(np.sum(R ** 2, axis=1, keepdims=True))

        if self.m == 3:
            r = 0.4
        else:
            r = 0.5

        mask = (np.min((R - 1) ** 2 + np.sum(R ** 2, axis=1, keepdims=True) - R ** 2 - r ** 2, axis=1) > 0) & \
               (np.sum((R - 1 / np.sqrt(self.M)) ** 2, axis=1) - r ** 2 > 0)

        # 删除满足条件的行
        R = R[~mask]

        return R

    def pf(self):
        if self.m == 2:
            f = self.sample
            f = f / torch.sqrt(f.pow(2).sum(dim=1, keepdim=True))
            return f

        elif self.m == 3:
            a = np.linspace(0, np.pi / 2, 60)
            x = np.outer(np.sin(a), np.cos(a))  # Shape (50, 50)
            y = np.outer(np.sin(a), np.sin(a))  # Shape (50, 50)
            z = np.outer(np.cos(a), np.ones_like(a))  # Shape (50, 50)

            # 将 x, y, z 合并成 R
            R = np.column_stack((x.flatten(), y.flatten(), z.flatten()))  # Merge into shape (2500, 3)

            # 计算 fes 符合条件的布尔数组
            fes = (np.min((R - 1) ** 2 + np.sum(R ** 2, axis=1, keepdims=True) - R ** 2 - 0.4 ** 2, axis=1) <= 0) | \
                  (np.sum((R - 1 / np.sqrt(3)) ** 2, axis=1) - 0.4 ** 2 <= 0)

            # 根据 fes 选择有效的索引
            valid_indices = np.where(fes)[0]

            # 选择符合条件的 x, y, z 值
            x_filtered = x.flatten()[valid_indices]
            y_filtered = y.flatten()[valid_indices]
            z_filtered = z.flatten()[valid_indices]

            # 合并并返回结果，确保没有 NaN
            result = np.column_stack((x_filtered, y_filtered, z_filtered))
            return torch.tensor(result, dtype=torch.float32)

        else:
            return torch.tensor([])  # 返回一个空的 Tensor

class C1_DTLZ3(DTLZ2):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.size()
        m = self.m
        g = 10 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((n, 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((n, 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )
        PopObj = f
        # 计算约束 r 的值
        if self.m == 2:
            r = 6
        elif self.m <= 3:
            r = 9
        elif self.m <= 8:
            r = 12.5
        else:
            r = 15
        # 计算约束条件
        PopCon = -(torch.sum(PopObj ** 2, dim=1, keepdim=True) - 16) * (torch.sum(PopObj ** 2, dim=1, keepdim=True) - r ** 2)

        return PopObj, PopCon

class C3_DTLZ4(DTLZ2):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m

        Xfront = X[:, : m - 1].pow(100)
        Xrear = X[:, m - 1:].clone()
        # X[:, : m - 1] = X[:, : m - 1].pow(100)

        g = torch.sum((Xrear - 0.5) ** 2, dim=1, keepdim=True)

        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((g.size(0), 1), device=X.device),
                            torch.maximum(
                                torch.cos(Xfront * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((g.size(0), 1), device=X.device),
                    torch.sin(torch.flip(Xfront, dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )
        PopObj = f
        # 计算约束条件
        PopCon = 1 - PopObj ** 2 / 4 - (torch.sum(PopObj ** 2, dim=1, keepdim=True) - PopObj ** 2)

        return PopObj, PopCon

    def GetOptimum(self):
        f = self.sample
        f = f / torch.sqrt(f.pow(2).sum(dim=1, keepdim=True))
        return f

    def pf(self):
        # 处理不同的目标数
        if self.m == 2:
            R = self.GetOptimum(100)
            return R  # 返回 (N, 2)

        elif self.m == 3:
            a = np.linspace(0, np.pi / 2, 60)  # 生成 10 个点
            x = np.outer(np.sin(a), np.cos(a))
            y = np.outer(np.sin(a), np.sin(a))
            z = np.outer(np.cos(a), np.ones_like(a))

            # 将 x, y, z 合并成 R
            R = np.column_stack((x.flatten(), y.flatten(), z.flatten()))  # Shape (100, 3)

            # 正规化
            R = R / np.sqrt(np.sum(R ** 2, axis=1, keepdims=True) - 3 / 4 * np.max(R ** 2, axis=1, keepdims=True))

            # reshape 为 (N, 3)
            result = [R[:, 0].reshape(x.shape), R[:, 1].reshape(x.shape), R[:, 2].reshape(x.shape)]

            # 应确保返回的形状为 (N, 3)
            return torch.tensor(R, dtype=torch.float32)  # 返回 torch.Tensor 并确保适当的形状

        else:
            return torch.tensor([])  # 返回一个空的 Tensor

class DC1_DTLZ1(DTLZ):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.size()
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        flip_cumprod = torch.flip(
            torch.cumprod(
                torch.cat([torch.ones((n, 1), device=X.device), X[:, : m - 1]], dim=1),
                dim=1,
            ),
            dims=[1],
        )
        rest_part = torch.cat(
            [
                torch.ones((n, 1), device=X.device),
                1 - torch.flip(X[:, : m - 1], dims=[1]),
            ],
            dim=1,
        )
        f = 0.5 * (1 + g) * flip_cumprod * rest_part

        pop_con = 0.5 - torch.cos(3 * torch.pi * X[:, 0])
        return f, pop_con.unsqueeze(1)

    def pf(self) -> torch.Tensor:
        if self.m == 2:
            # 生成 100 个点在 [0, 1] 之间
            R = torch.zeros((100, 2), dtype=torch.float32)
            R[:, 0] = torch.linspace(0, 1, 100)
            R[:, 1] = 1 - R[:, 0]

            # 使用 nan 处理约束条件
            R[torch.cos(3 * torch.pi * R[:, 0]) < 0.5, :] = float('nan')
            R /= 2
            # 删除包含 nan 值的行
            R = R[~torch.isnan(R).any(dim=1)]
            return R  # 返回有效点的张量

        elif self.m == 3:
            a = torch.linspace(0, 1, 60).view(-1, 1)  # 生成 10 个点并转为列向量
            x = a @ a.T  # 形状 (10, 10)
            y = a * (1 - a.T)  # 形状 (10, 10)
            z = (1 - a) * torch.ones_like(x)  # 确保 z 也是 (10, 10)

            # 创建 mask
            mask = torch.cos(3 * torch.pi * (a @ torch.ones((1, 10)))) < 0.5  # 扩展 a 以匹配形状
            z[mask] = float('nan')  # 使用 mask 进行条件筛选

            # 合并并返回结果
            # Flatten in a way to keep the tensors as 2D
            R = torch.cat([(x / 2).view(-1, 1), (y / 2).view(-1, 1), (z / 2).view(-1, 1)], dim=1)
            R = R[~torch.isnan(R).any(dim=1)]  # 删除所有行含有 na 的点
            return R  # 返回的 R 是一个 (100, 3) 的张量

        else:
            return torch.tensor([])  # 返回一个空的 Tensor

class DC1_DTLZ3(DTLZ2):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.size()
        m = self.m
        g = 10 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((n, 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((n, 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )
        pop_con = 0.5 - torch.cos(3 * torch.pi * X[:, 0])
        return f, pop_con.unsqueeze(1)

    def pf(self) -> torch.Tensor:
        if self.m == 2:
            x = torch.linspace(0, torch.pi / 2, 100).view(-1, 1)  # 生成列向量
            R = torch.zeros((100, 2), dtype=torch.float32)  # 初始化结果 Tensor
            R[:, 0] = torch.cos(x).squeeze()  # 第一维
            R[:, 1] = torch.sin(x).squeeze()  # 第二维

            # 使用有效性掩码处理约束条件
            R[torch.cos(6 * x) < 0.5, :] = float('nan')  # 将不符合条件的点设为 nan

            # 删除包含 nan 的行
            R = R[~torch.isnan(R).any(dim=1)]  # 删除所有行含有 nan 的点
            return R  # 返回有效点的张量

        elif self.m == 3:
            a = torch.linspace(0, torch.pi / 2, 50).view(-1, 1)  # 生成 10 个点并转为列向量
            x = torch.cos(a) @ torch.cos(a.T)  # 形状 (10, 10)
            y = torch.cos(a) @ torch.sin(a.T)  # 形状 (10, 10)
            z = torch.sin(a) * torch.ones_like(x)  # 形状 (10, 10)

            # 创建掩码，将不符合条件的点设为 nan
            mask = torch.cos(6 * a) < 0.5
            z[mask.expand_as(z)] = float('nan')  # 扩展 mask 的形状以匹配 z

            # 合并并返回结果
            R = torch.cat((x.view(-1, 1), y.view(-1, 1), z.view(-1, 1)), dim=1)
            R = R[~torch.isnan(R).any(dim=1)]  # 删除所有行含有 nan 的点
            return R  # 返回有效点的张量

        else:
            return torch.tensor([])  # 返回一个空 Tensor


class DC2_DTLZ1(DTLZ):
    def __init__(self, d: int = 7, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.size()
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        flip_cumprod = torch.flip(
            torch.cumprod(
                torch.cat([torch.ones((n, 1), device=X.device), X[:, : m - 1]], dim=1),
                dim=1,
            ),
            dims=[1],
        )
        rest_part = torch.cat(
            [
                torch.ones((n, 1), device=X.device),
                1 - torch.flip(X[:, : m - 1], dims=[1]),
            ],
            dim=1,
        )
        f = 0.5 * (1 + g) * flip_cumprod * rest_part

        # 计算约束值
        g = 100 * (self.d - self.m + 1 + torch.sum((X[:, self.m - 1:] - 0.5) ** 2 -
                                                   torch.cos(20 * torch.pi * (X[:, self.m - 1:] - 0.5)), dim=1,
                                                   keepdim=True))

        # 计算约束条件
        pop_con = torch.zeros((X.size(0), 2), device=X.device)
        pop_con[:, 0] = 0.5 - torch.cos(3 * torch.pi * g.squeeze())
        pop_con[:, 1] = 0.5 - torch.exp(-g.squeeze())
        return f, pop_con

class DC2_DTLZ3(DTLZ2):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.size()
        m = self.m
        g = 10 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((n, 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((n, 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )
        # 计算约束值
        g = 10 * (d - self.m + 1 + torch.sum((X[:, self.m - 1:] - 0.5) ** 2 -
                                                   torch.cos(20 * torch.pi * (X[:, self.m - 1:] - 0.5)), dim=1,
                                                   keepdim=True))

        # 计算约束条件
        pop_con = torch.zeros((X.size(0), 2), device=X.device)
        pop_con[:, 0] = 0.5 - torch.cos(3 * torch.pi * g.squeeze())
        pop_con[:, 1] = 0.5 - torch.exp(-g.squeeze())
        return f, pop_con

class DC3_DTLZ1(DTLZ):
    def __init__(self, d: int = 7, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.size()
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        flip_cumprod = torch.flip(
            torch.cumprod(
                torch.cat([torch.ones((n, 1), device=X.device), X[:, : m - 1]], dim=1),
                dim=1,
            ),
            dims=[1],
        )
        rest_part = torch.cat(
            [
                torch.ones((n, 1), device=X.device),
                1 - torch.flip(X[:, : m - 1], dims=[1]),
            ],
            dim=1,
        )
        f = 0.5 * (1 + g) * flip_cumprod * rest_part

        # 计算约束值
        pop_con = torch.zeros((X.size(0), self.m), device=X.device)

        # 计算约束条件 1
        pop_con[:, :self.m - 1] = 0.5 - torch.cos(3 * torch.pi * X[:, :self.m - 1])

        # 计算 g 并获得约束条件 2
        g = 100 * (self.d - self.m + 1 + torch.sum((X[:, self.m - 1 :] - 0.5) ** 2 -
                                                    torch.cos(20 * torch.pi * (X[:, self.m - 1 :] - 0.5)), dim=1, keepdim=True))
        pop_con[:, self.m - 1] = 0.5 - torch.cos(3 * torch.pi * g.squeeze())

        return f, pop_con

    def pf(self) -> torch.Tensor:
        if self.m == 2:
            R = torch.zeros((100, 2), dtype=torch.float32)  # 初始化结果张量
            R[:, 0] = torch.linspace(0, 1, 100)  # 第一维
            R[:, 1] = 1 - R[:, 0]  # 第二维

            R[torch.cos(3 * torch.pi * R[:, 0]) < 0.5, :] = float('nan')  # 应用约束
            R /= 2  # 缩放

            R = R[~torch.isnan(R).any(dim=1)]  # 删除含有 nan 的行

        elif self.m == 3:
            a = torch.linspace(0, 1, 40).view(-1, 1)  # 生成 40 个点
            x = a @ a.t()  # 生成 x
            y = a * (1 - a.t())  # 生成 y
            z = (1 - a) * torch.ones_like(a.t())  # 生成 z

            mask1 = torch.cos(3 * torch.pi * a @ torch.ones((1, 40))) < 0.5
            mask2 = torch.cos(3 * torch.pi * torch.ones((40, 1)) @ a.t()) < 0.5
            z[mask1 | mask2] = float('nan')  # 设置 z 的约束

            R = torch.cat((x.view(-1,1) / 2, y.view(-1,1) / 2, z.view(-1,1) / 2), dim=1)
            R = R[~torch.isnan(R).any(dim=1)]  # 删除含有 nan 的行

        else:
            return torch.tensor([])  # 返回空张量

        return R  # 返回 Pareto 前沿的张量

class DC3_DTLZ3(DTLZ2):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.size()
        m = self.m
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((n, 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((n, 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )

        # 计算约束值
        pop_con = torch.zeros((X.size(0), self.m), device=X.device)

        # 计算约束条件 1
        pop_con[:, :self.m - 1] = 0.5 - torch.cos(3 * torch.pi * X[:, :self.m - 1])

        # 计算 g 并获得约束条件 2
        g = 100 * (self.d - self.m + 1 + torch.sum((X[:, self.m - 1 :] - 0.5) ** 2 -
                                                    torch.cos(20 * torch.pi * (X[:, self.m - 1 :] - 0.5)), dim=1, keepdim=True))
        pop_con[:, self.m - 1] = 0.5 - torch.cos(3 * torch.pi * g.squeeze())

        return f, pop_con

    def pf(self) -> torch.Tensor:
        if self.m == 2:
            # 当有 2 个目标时
            x = torch.linspace(0, torch.pi / 2, 100).view(-1, 1)  # 生成 100 个点（列向量）
            R = torch.zeros((100, 2), dtype=torch.float32)  # 初始化结果张量
            R[:, 0] = torch.cos(x)  # 第一维
            R[:, 1] = torch.sin(x)  # 第二维

            # 根据约束条件设置为 nan
            R[torch.cos(6 * x) < 0.5, :] = float('nan')

        elif self.m == 3:
            # 当有 3 个目标时
            a = torch.linspace(0, torch.pi / 2, 40).view(-1, 1)  # 生成 40 个点（列向量）
            x = torch.cos(a) @ torch.cos(a.t())  # 生成 x
            y = torch.cos(a) @ torch.sin(a.t())  # 生成 y
            z = torch.sin(a) @ torch.ones((1, 40))  # 生成 z

            # 创建 mask 以满足约束条件
            mask1 = torch.cos(6 * a @ torch.ones((1, 40))) < 0.5
            mask2 = torch.cos(6 * torch.ones((40, 1)) @ a.t()) < 0.5

            z[mask1 | mask2] = float('nan')  # 满足任一约束条件的 z 设置为 nan

            # 将 x、y 和 z 组合成结果
            R = torch.cat((x.view(-1, 1), y.view(-1, 1), z.view(-1, 1)), dim=1)  # 合并为 (40*40, 3) 的张量

        else:
            return torch.tensor([])  # 返回一个空 Tensor

        # 返回最终的 Pareto 前沿的张量
        return R[~torch.isnan(R).any(dim=1)]  # 删除所有行含有 nan 的点并返回