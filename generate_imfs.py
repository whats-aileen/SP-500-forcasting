import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import csv
from scipy import stats

# 近似为零的标准
zero_standard = 0.1
# 计算单个IMF的最高循环次数
max_times = 10


def is_max(i_, list_):
    if i_ == 0 or i_ == len(list_) - 1:
        return False
    if list_[i_ - 1] >= list_[i_]:
        return False
    j_ = i_ + 1
    while j_ < len(list_):
        if list_[j_] < list_[i_]:
            break
        elif list_[j_] > list_[i_]:
            return False
        else:
            j_ += 1
    if j_ == len(list_):
        return False
    return True


def is_min(i_, list_):
    if i_ == 0 or i_ == len(list_) - 1:
        return False
    if list_[i_ - 1] <= list_[i_]:
        return False
    j_ = i_ + 1
    while j_ < len(list_):
        if list_[j_] > list_[i_]:
            break
        elif list_[j_] < list_[i_]:
            return False
        else:
            j_ += 1
    if j_ == len(list_):
        return False
    return True


# 判断是否是IMF分量
def is_imf(a, t):
    num_m = 0
    num_zero = 0
    avg = 0
    for i_ in range(1, len(a) - 1):
        if is_max(i_, a):
            num_m += 1
        elif is_min(i_, a):
            num_m += 1
    for i_ in range(len(a) - 1):
        avg = avg + a[i_]
        if a[i_] * a[i_ + 1] < 0:
            num_zero += 1
        elif a[i_] == 0:
            num_zero += 1
    delta = num_m - num_zero
    if delta > 1 or delta < -1:
        return False
    avg += a[-1]
    avg = avg / len(a)
    if -zero_standard < avg < zero_standard or t > max_times:
        return True
    else:
        return False


def is_final_residual(a):
    compare = []
    in_point = 0
    for i in range(len(a) - 1):
        if a[i] > a[i + 1]:
            compare.append(1)
        elif a[i] < a[i + 1]:
            compare.append(-1)
        else:
            compare.append(0)
    for i in range(len(compare) - 1):
        if compare[i] * compare[i + 1] < 0:
            in_point = in_point + 1
        elif compare[i] * compare[i + 1] == 0 and a[i] != 0 and i != len(compare) - 2:
            j = i + 1
            while compare[j] == 0 and j < len(compare):
                j = j + 1
            if compare[i] * compare[j] < 0:
                in_point = in_point + 1
    if in_point > 1:
        return False
    else:
        return True


def generate_imf(series_):
    imf_list = []
    s_t = series_
    while True:
        y_t_ = s_t
        tms = 1
        while True:
            p_t = generate_p_t(y_t_)
            if is_imf(p_t, tms):
                plt.clf()
                break
            else:
                y_t_ = p_t
                tms = tms + 1
        imf_list.append(p_t)
        res = []
        for i_ in range(len(s_t)):
            res.append(s_t[i_] - p_t[i_])
        if is_final_residual(res):
            break
        else:
            s_t = res
    return [imf_list, res]


def generate_p_t(y):
    high = []  # h_t是上包络线，其元素为多个[x, y]数对
    low = []
    length = len(y)
    # if y[0] > y[1]:
    high.append([0, y[0]])
    # elif y[0] < y[1]:
    low.append([0, y[0]])

    for i_ in range(1, length - 1):
        if y[i_] > y[i_ + 1] and y[i_] > y[i_ - 1]:
            high.append([i_, y[i_]])
        elif y[i_] < y[i_ + 1] and y[i_] < y[i_ - 1]:
            low.append([i_, y[i_]])
    # if y[length - 1] > y[length - 2]:
    high.append([length - 1, y[length - 1]])
    # elif y[length - 1] < y[length - 2]:
    low.append([length - 1, y[length - 1]])

    x_h = []
    y_h = []
    x_l = []
    y_l = []
    for i_ in range(len(high)):
        x_h.append(high[i_][0])
        y_h.append(high[i_][1])
    for i_ in range(len(low)):
        x_l.append(low[i_][0])
        y_l.append(low[i_][1])
    x_h = np.array(x_h)
    y_h = np.array(y_h)
    x_l = np.array(x_l)
    y_l = np.array(y_l)
    #样条插值
    cs_h = CubicSpline(x_h, y_h)
    cs_l = CubicSpline(x_l, y_l)
    x_new = np.linspace(0, length - 1, length)
    y_h_new = cs_h(x_new)
    y_l_new = cs_l(x_new)
    p = []
    for i_ in range(length):
        p.append(y[i_] - ((y_h_new[i_] + y_l_new[i_]) / 2))
    return p


def draw(IMFs_, res_):
    num = len(IMFs)
    x = range(1, len(IMFs_[0]) + 1)
    fig, axes = plt.subplots(num + 1, 1, figsize=(6, 9))
    for i_ in range(num):
        print(f'type of axes[{i_}] is {type(axes[i_])}')
        axes[i_].plot(x, IMFs_[i_], label=f'IMFs_emd. IMF {i_ + 1}')
        axes[i_].legend()
    axes[num].plot(x, res_, label='IMFs_emd. residual')
    axes[num].legend()
    plt.tight_layout()
    plt.savefig('CEEMDAN_result.png')
    plt.show()


def calculate_p(x):
    x = np.array(x)
    t_stat, p_value = stats.ttest_1samp(x, popmean=0)
    return p_value


# 数据读取
data_pd = pd.read_csv("data.csv", header=None)

# 数据类型转换, data是一维数组
data = []
for i in range(len(data_pd)):
    data.append(data_pd.iloc[i, 1])
flag = True
times = 0

result = generate_imf(data)
IMFs = result[0]
residual = result[1]
print('IMFs generated. ')

draw(IMFs, residual)
print('fig saved. ')

# IMFs参数
amount = len(IMFs)
days = len(IMFs[0])

p_list = []
for i in range(amount):
    p_list.append(float(calculate_p(IMFs[i])))

# 频率分类列表，第一个元素为高频，第二个为低频，第三个为趋势
fre_list = [[0 for i in range(days)], [0 for i in range(days)], residual]

for i in range(amount):
    if p_list[i] >= 0.01:
        for j in range(days):
            fre_list[0][j] += IMFs[i][j]
    else:
        for j in range(days):
            fre_list[1][j] += IMFs[i][j]

for i in range(3):
    series = []
    for j in range(days):
        series.append([float(fre_list[i][j])])
    with open(f'frequency_list/fre{i}.csv', 'w') as f:
        csv.writer(f).writerows(series)

print('data saved. ')
