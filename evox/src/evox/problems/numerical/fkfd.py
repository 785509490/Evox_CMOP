import torch

from evox.core import Problem
from evox.fkfd.Convert import SetParameter
from evox.utils import clamp

class FKFDTestSuit(Problem):
    """
    Base class for FKFD test suite problems in single-objective optimization.

    Inherit this class to implement specific FKFD problem variants.
    """

    def __init__(self, scenario_idx: int = None):
        """Override the setup method to initialize the parameters"""
        super().__init__()
        self.scenario_idx = scenario_idx

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to evaluate the objective values for given decision variables.
        :param X: A tensor of shape (n, d), where n is the number of solutions and d is the number of decision variables.
        :return: A tensor of shape (n, m) representing the objective values for each solution.
        """
        return self._true_evaluate(X)


def batch_update_radar(radar_channel, launch_missile, valid_mask,
                       start_idx, hit_idx, cur_radar, radar_usage,
                       radar_n_channel, radar_type, cur_launch):
    """
    参数说明：
    radar_channel: [n, time_steps, radar_count] 雷达通道状态
    launch_missile: [n, missile_types] 剩余导弹数量
    valid_mask: [n] 有效样本掩码
    start_idx: [n] 开始时间索引
    hit_idx: [n] 结束时间索引
    cur_radar: [n] 当前雷达索引 (0-based)
    radar_usage: [n] 雷达使用系数
    radar_n_channel: [max_radar_type] 各类型雷达通道数
    radar_type: [radar_count] 雷达类型映射表
    cur_launch: [n] 当前发射类型索引
    """
    # 步骤1: 筛选有效样本
    valid_indices = torch.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return radar_channel, launch_missile, valid_mask

    # 步骤2: 二次校验（防御性编程）
    valid_radar = cur_radar[valid_indices]
    assert torch.all(valid_radar >= 0), "存在负雷达索引"
    assert torch.all(valid_radar < len(radar_type)), \
        f"雷达索引超限 ({valid_radar.max()} >= {len(radar_type)})"

    # 步骤3: 保存原始状态用于回滚
    original_radar = radar_channel[valid_indices].clone()  # [V,T,R]

    # 步骤4: 张量化三维散射更新 -------------------------------
    # 生成时间步张量 [1, T]
    time_steps_tensor = torch.arange(radar_channel.size(1), device=radar_channel.device).view(1, -1)

    # 扩展 start_idx 和 hit_idx 为列向量 [n, 1]
    start_expanded = start_idx.view(-1, 1)
    hit_expanded = hit_idx.view(-1, 1)

    # 钳位 hit_idx 不超过最大时间步
    hit_expanded = torch.clamp(hit_expanded, max=radar_channel.size(1) - 1)

    # 生成时间掩码 [n, T]
    time_mask = (time_steps_tensor >= start_expanded) & (time_steps_tensor <= hit_expanded)
    time_mask = time_mask & valid_mask.view(-1, 1)  # 应用有效样本掩码

    # 生成 usage 矩阵 [n, T]
    usage = time_mask * radar_usage.view(-1, 1)

    # 生成雷达掩码 [n, R]
    radar_mask = torch.zeros((radar_channel.size(0), radar_channel.size(2)), device=radar_channel.device)
    radar_mask[torch.arange(radar_channel.size(0)), cur_radar] = 1  # 当前雷达位置为1

    # 扩展 usage 到三维 [n, T, R]
    usage_expanded = usage.unsqueeze(2) * radar_mask.unsqueeze(1)  # [n, T, R]

    # 将更新值累加到 radar_channel
    radar_channel += usage_expanded
    # ---------------------------------------------------------

    # 步骤5: 通道数约束检查
    radar_limits = radar_n_channel[radar_type[valid_radar] - 1]  # [V]
    updated_usage = radar_channel[valid_indices, :, valid_radar]  # [V,T]
    overflow_mask = (updated_usage > radar_limits.unsqueeze(1)).any(dim=1)  # [V]

    # 步骤6: 处理回滚
    if overflow_mask.any():
        rollback_idx = valid_indices[overflow_mask]  # [K], K为超限样本数
        radar_channel[rollback_idx] = original_radar[overflow_mask]
        valid_mask[rollback_idx] = False

    # 步骤7: 更新导弹数量
    success_mask = ~overflow_mask
    success_idx = valid_indices[success_mask]
    launch_missile[success_idx, cur_launch[success_idx]] -= 1

    return radar_channel, launch_missile, valid_mask

def fkfd_func(self, X: torch.Tensor) -> torch.Tensor:
    """
    输入:
    X: shape [n, d] 的决策变量张量
    返回:
    包含目标函数和约束的字典
    """
    device = X.device
    Missile = self.Prob.Missile
    Radar = self.Prob.Radar
    Target = self.Prob.Target
    Prob = self.Prob

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(X)
    time_scatter = Prob.dT
    n_target = len(Prob.targetInd)
    max_missile_num = Target.MaxMissiles

    # 统一转换所有参数为PyTorch张量
    Prob.data = [torch.tensor(target_data, dtype=torch.float32, device=device)
                 for target_data in Prob.data]
    Prob.valid_dec_counts = [t.shape[0] for t in Prob.data]

    radar_base = torch.from_numpy(Radar.radar.copy()).float().to(device)
    radar_n_channel = torch.tensor(Radar.nChannel, dtype=torch.float32, device=device)
    radar_coefficient = torch.tensor(Radar.coefficient, dtype=torch.float32, device=device)
    launch_num = torch.tensor(Missile.num_type, dtype=torch.int32, device=device)
    radar_type = torch.tensor(Radar.type, dtype=torch.long, device=device)
    enemy_type = torch.tensor(Target.type, dtype=torch.long, device=device)

    # 初始化雷达通道（带批次维度）
    radar_channel = radar_base.unsqueeze(0).expand(n, -1, -1).clone()  # [n, time, radar]
    launch_missile = launch_num.unsqueeze(0).expand(n, -1).clone()  # [n, missile_types]

    # 预处理所有个体的决策
    all_decs = X  # [n, n_target]
    f = torch.zeros(n, device=device)

    # 遍历每个目标
    for j in range(n_target):
        cur_target = Prob.targetInd[j] - 1  # 0-based
        cur_decs = all_decs[:, j].long()  # [n]

        # 获取当前目标的所有方案参数
        target_data = Prob.data[cur_target]  # [valid_decs, params]
        cur_scheme = target_data[cur_decs]

        # 提取方案参数
        start_time = cur_scheme[:, 0].long()  # [n]
        hit_time = cur_scheme[:, 3].long()  # [n]
        cur_radar = cur_scheme[:, 2].long() - 1  # [n]
        cur_launch = cur_scheme[:, 1].long() - 1  # [n]
        cur_hit_p = cur_scheme[:, 8]  # [n]

        # 约束检查
        time_diff = hit_time - start_time
        time_valid = (torch.ceil(time_diff.float() / time_scatter) > 0).float()
        missile_valid = (launch_missile[torch.arange(n), cur_launch] >= 1).float()

        # 计算雷达使用
        start_idx = torch.ceil(start_time.float() / time_scatter).long()  # [n]
        hit_idx = torch.ceil(hit_time.float() / time_scatter).long() - 1  # [n]

        # 获取雷达使用系数
        radar_usage = radar_coefficient[radar_type[cur_radar] - 1, enemy_type[cur_target] - 1]  # [n]

        # 生成有效掩码 (基于之前的约束检查)
        valid_mask = (time_valid * missile_valid * cur_hit_p) > 0

        # 调用批量更新函数
        radar_channel, launch_missile, valid_mask = batch_update_radar(
                radar_channel, launch_missile, valid_mask,
                start_idx, hit_idx, cur_radar, radar_usage,
                radar_n_channel, radar_type, cur_launch
        )
        valid_hits = cur_hit_p * valid_mask.float()  # 有效命中值 [n]
        f += valid_hits
    # 汇总目标函数
    f = 1 / (f + 1e-4)
    return f

class FKFD(FKFDTestSuit):
    def __init__(self, scenario_idx: int = None):
        self.Prob = SetParameter(scenario_idx)
        super().__init__()

        # 将关键参数转换为张量
        self.time_scatter = torch.tensor(self.Prob.dT, dtype=torch.float32)
        radar_type_indices = self.Prob.Radar.type.astype(int) - 1  # 类型索引从0开始
        self.radar_n_channel = torch.tensor(
            [self.Prob.Radar.nChannel[idx] for idx in radar_type_indices],
            dtype=torch.float32
        )
        self.radar_coefficient = torch.tensor(
            self.Prob.Radar.coefficient, dtype=torch.float32
        )
        self.launch_num = torch.tensor(
            self.Prob.Missile.num_type, dtype=torch.int32
        )
        self.Prob.Target.type = torch.from_numpy(self.Prob.Target.type.astype(int))
        self.Prob.Radar.type = torch.from_numpy(self.Prob.Radar.type.astype(int))

    def _true_evaluate(self, X: torch.Tensor) -> torch.Tensor:
        return fkfd_func(self, X)
