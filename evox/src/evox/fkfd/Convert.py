from types import SimpleNamespace
import torch
import numpy as np
from mpmath import zeros

from evox.fkfd.utilsmat import load_mat_file


class ProbClass:
    pass


class MissileClass:
    pass


class TargetClass:
    pass


class targetClass:
    pass


class RadarClass:
    pass


def SetParameter(j: int = None, case=None):
    parameter = load_mat_file(r'E:\codePY\newevox-main\newevox-main\evox\src\evox\fkfd\data\parameter_{}.mat'.format(j + 1))["parameter"]
    parameter = SimpleNamespace(**parameter)
    Prob = ProbClass()
    Missile = MissileClass()
    Target = TargetClass()
    Radar = RadarClass()

    Prob.nRadarType = 6

    Prob.nTargetType = 3

    Prob.nMissileType = 6

    ## 雷达参数
    Radar.R = parameter.radarR[parameter.radar_type - 1]
    Radar.accuracy = parameter.radar_accuracy
    Radar.nChannel = parameter.radar_n_channel
    Radar.coefficient = np.ones((Prob.nRadarType, Prob.nTargetType))
    # 部署参数
    Radar.N = parameter.N_radar
    Radar.type = parameter.radar_type
    Radar.location = parameter.radar_location
    Radar.angle = parameter.radar_angle
    ## 导弹参数

    Missile.rKill = np.empty((Prob.nRadarType, Prob.nMissileType), dtype=object)
    Missile.pKill = np.empty((Prob.nRadarType, Prob.nMissileType), dtype=object)

    Missile.rKill[0, 0] = parameter.r
    Missile.pKill[0, 0] = parameter.p

    for i in range(Prob.nRadarType):
        for j in range(Prob.nMissileType):
            Missile.rKill[i, j] = Missile.rKill[0, 0]
            Missile.pKill[i, j] = Missile.pKill[0, 0]

    Missile.cost = parameter.cost_our[parameter.launch_type - 1]

    Missile.v = parameter.v_our[parameter.launch_type - 1]

    Missile.launchProbability = parameter.launch_probability

    Missile.R = parameter.R

    Missile.Theta = parameter.Theta

    # 部署参数
    Missile.N = parameter.N_missile

    Missile.type = parameter.launch_type

    Missile.location = parameter.launch_location

    Missile.accuracy = parameter.launch_to_radar

    Missile.num_type = parameter.num_type

    ## 目标参数

    Target.v = parameter.v_enemy[parameter.launch_type - 1]

    Target.cost = parameter.cost_enemy[parameter.launch_type - 1]

    # 部署参数

    Target.startPOS = parameter.enemy_start
    Target.type = parameter.enemyType
    Target.angle = parameter.enemy_angle
    Target.order = parameter.Target_order

    Target.N = len(parameter.enemyType)

    Target.x0 = np.zeros(Target.N)

    Target.length = np.zeros(Target.N)

    for i in range(Target.N):
        Target.length[i] = Target.startPOS[i, 1] / np.sin(Target.angle[i])
        Target.x0[i] = Target.startPOS[i, 0] - Target.length[i] * np.cos(Target.angle[i])

    ## 总体参数设置
    Target.MaxMissiles = 1

    Prob.preT = 15
    Prob.dT = 5
    Prob.preNT = int(np.ceil(Prob.preT / Prob.dT))
    Prob.modeFire = 2
    if Prob.modeFire == 1:
        Prob.deltaT1Lower = 0
        Prob.deltaT1Upper = 5
    else:
        Prob.deltaT1Lower = 10
        Prob.deltaT1Upper = 15

    Prob.deltaT2 = 10
    Prob.deltaT3 = 15

    target = []
    for i in range(Target.N):
        target_ = targetClass()
        # target_.type = Target.type[i]
        target_.type = Target.type[min(i, len(Target.type) - 1)]
        target_.startPOS = Target.startPOS[i]
        target_.angle = Target.angle[i]
        target_.length = Target.length[i]
        target_.v = Target.v[target_.type - 1]
        target_.cost = Target.cost[target_.type - 1]
        target_.T = np.floor(target_.length / target_.v)
        target.append(target_)

    flyT = [tmp.T for tmp in target]
    maxT = max(flyT)

    Prob.nS = int(np.floor(maxT / Prob.dT))
    Radar.radar = np.zeros((Prob.nS, Radar.N))

    ## 封装数据
    Prob.nRadar = Radar.N
    Prob.nMissile = Missile.N
    Prob.nTarget = Target.N
    Prob.Missile = Missile
    Prob.Target = Target
    Prob.Radar = Radar
    Prob.target = target

    target = Prob.target
    Prob.data = parameter.data
    data_len = np.zeros((1, Target.N))
    for i in range(Target.N):
        cur_target = parameter.Target_order[i]
        data = parameter.data[cur_target - 1]
        data_len[0, i] = len(data)
    for i in range(Target.N):
        data = parameter.data[i]
        pKill = []
        for j in range(Prob.nMissile):
            pKillm = []
            for k in range(Prob.nRadar):
                selected_data = data[(data[:, 1] == (j + 1)) & (data[:, 2] == (k + 1))][:, [-1, 0, 3]]
                sort_indices = np.lexsort((-selected_data[:, 0], selected_data[:, 1]))
                sorted_data = selected_data[sort_indices]
                pKillm.append(sorted_data)
            pKill.append(pKillm)
        target[i].pKill = pKill
        combination = np.unique(data[:, 1:3], axis=0)
        max_values = np.zeros(len(combination))
        # 遍历每个唯一的组合
        for idx, comb in enumerate(combination):
            # 找到所有匹配当前组合的行的索引
            matching_indices = np.all(data[:, 1:3] == comb, axis=1)
            # 提取这些行的最后一列并计算最大值
            max_values[idx] = np.max(data[matching_indices, -1])
        target[i].combination_p = max_values
        target[i].combination = (combination - 1).astype(int)
    data_len = torch.tensor(data_len)
    Prob.data_len = data_len
    # Prob.targetInd = np.argsort(flyT, kind='mergesort')
    Prob.targetInd = parameter.Target_order
    Prob.target = target
    return Prob

