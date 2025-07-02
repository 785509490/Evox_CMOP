import numpy as np
from collections import defaultdict


def cal_obj(population, Prob):
    Missile = Prob.Missile
    Radar = Prob.Radar
    Target = Prob.Target

    """计算种群中每个个体的目标函数值"""
    n = len(population)
    time_scatter = Prob.dT
    n_target = len(Prob.targetInd)
    max_missile_num = Target.MaxMissiles
    radar_n_channel = Radar.nChannel
    radar_coefficient = Radar.coefficient
    launch_num = Missile.num_type
    radar_type = Radar.type
    enemy_type = Target.type

    for i in range(n):
        radar_channel = Radar.radar.copy()   # 雷达通道记录
        launch_missile = launch_num.copy()   # 导弹数量记录
        tmp_obj = []
        intercept_t = np.zeros((max_missile_num, n_target))

        for j in range(n_target):
            cur_target = int(Prob.targetInd[j])
            cur_decs = int(population[i].decs[j])
            cur_flag = np.zeros((1, max_missile_num))
            cur_flag[0] = 1  # 初始第一个位置为可行
            cur_scheme = Prob.data[cur_target - 1][cur_decs]  # 当前目标的发射方案
            # former_radar = radar_channel

            for n_missile in range(1):  # 原MATLAB代码中 n=1:1 的特殊情况
                if not cur_flag[n_missile]:
                    continue

                # 提取当前导弹信息
                start_time = int(cur_scheme[0])
                hit_time = int(cur_scheme[3])
                cur_radar = int(cur_scheme[2])  # 转换为0-based索引
                cur_launch = int(cur_scheme[1])
                cur_hit_p = cur_scheme[8]
                cur_flag[n_missile] = cur_hit_p

                # 约束检查
                time_diff = hit_time - start_time
                if np.ceil(time_diff / time_scatter) <= 2:
                    cur_flag[n_missile] = 0
                    continue

                if launch_missile[cur_launch - 1] < 1:
                    cur_flag[n_missile] = 0
                    continue

                # 雷达通道计算
                start_idx = int(np.ceil(start_time / time_scatter))
                hit_idx = int(np.ceil(hit_time / time_scatter)) - 1
                radar_usage = radar_coefficient[radar_type[cur_radar - 1] - 1][enemy_type[cur_target - 1] - 1]

                # 临时保存原始通道状态
                tmp_radar = radar_channel
                radar_channel[start_idx - 1:hit_idx - 1, cur_radar - 1] += radar_usage

                # 检查通道约束
                if np.any(radar_channel[:, cur_radar - 1] > radar_n_channel[radar_type[cur_radar - 1] - 1]):
                    radar_channel = tmp_radar
                    cur_flag[n_missile] = 0
                    continue

                # 更新导弹数量
                launch_missile[cur_launch - 1] -= 1
                target_used_radar = radar_channel
                if cur_flag[n_missile] != 0:
                    population[i].TargetUsedRadar[j] = target_used_radar


            # 保存目标信息
            if cur_flag[0] not in (0, 1):
                tmp_obj.append(cur_flag[0])

        # 计算目标函数
        population[i].everyobjs = tmp_obj
        population[i].objs = np.sum(tmp_obj)
        population[i].radar = radar_channel
        population[i].missile = launch_missile

    return population
