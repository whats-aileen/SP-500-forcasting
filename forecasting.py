import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random

Train_Window = 30
Epochs = 50
Input_Size = 1
Hidden_Size = 5
Num_Layers = 3
Output_Size = 1
Batch_Size_Train = 72
Batch_Size_Val = 1
Learning_Rate = 0.025
Train_Ratio = 0.7


class LSTMModel(nn.Module):
    def __init__(self, input_size=Input_Size, hidden_size=Hidden_Size, num_layers=Num_Layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, Output_Size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class SeriesDataSet(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(-1)
        self.y = torch.tensor(y).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(data_):
    xs, ys = [], []
    for i in range(len(data_) - Train_Window):
        xs.append(data_[i:i + Train_Window])
        ys.append(data_[i + Train_Window])
    return xs, ys


# 评价x与y序列的差值
def evaluate(p, a):
    MAE = mean_absolute_error(a, p)
    MSE = mean_squared_error(a, p)
    print(f'mean absolute error is: {MAE}, mean squared error is: {MSE}')


# 需要输入列数为1的向量
def normalize(data_, i_):
    return scaler[i_].fit_transform(data_)


def denormalize(data_, i_):
    return scaler[i_].inverse_transform(data_)


print(f'cuda available: {torch.cuda.is_available()}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = [MinMaxScaler(feature_range=(-1, 1)) for i in range(3)]
model = [LSTMModel().to(device) for i in range(3)]
name = ['high_frequency', 'medium_frequency', 'trend']
all_data = []
# 验证集预测结果
forecasting_val = []
# 整体数据预测结果
forecasting_all = []

days_list = []
for i in range(3):
    data = pd.read_csv(f'frequency_list/fre{i}.csv', header=None)
    days_list.append(len(data))
if days_list[0] != days_list[1] or days_list[1] != days_list[2]:
    print(f'[error]: days are {days_list[0]}, {days_list[1]} and {days_list[2]}, not equal')
days = days_list[0]
train_size = int(days * Train_Ratio)
next_day = 0

for i in range(3):
    data = pd.read_csv(f'frequency_list/fre{i}.csv', header=None)
    data = data[0].tolist()
    all_data.append(data)
    data = np.array(data).reshape(-1, 1)
    data = normalize(data, i)

    # 划分测试集
    X, y = create_sequences(data)
    X_v = torch.tensor(X[train_size:], dtype=torch.float32)
    y_v = torch.tensor(y[train_size:], dtype=torch.float32)
    X_last = torch.tensor(X[-1][None, :, :], dtype=torch.float32)

    # 打乱数据
    dataset = []
    for j in range(train_size):
        dataset.append([X[j], y[j]])
    random.shuffle(dataset)
    # 划分训练集
    X_t = []
    y_t = []
    for j in range(train_size):
        X_t.append(dataset[j][0])
        y_t.append(dataset[j][1])
    X_t = torch.tensor(X_t, dtype=torch.float32)
    y_t = torch.tensor(y_t, dtype=torch.float32)

    # 整体数据张量化
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    LSTMModel().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model[i].parameters(), lr=Learning_Rate)

    print(f'[adv]: {name[i]} data divided, begin training model')

    # 模型训练部分
    for epoch in range(Epochs):
        model[i].train()
        total_loss = 0
        for j in range(int(train_size / Batch_Size_Train) + 1):
            # 将训练数据划分为batch
            if (j + 1) * Batch_Size_Train > train_size:
                X_batch = X_t[j*Batch_Size_Train: train_size].to(device)
                y_batch = y_t[j*Batch_Size_Train: train_size].to(device)
            else:
                X_batch = X_t[j * Batch_Size_Train: (j + 1) * Batch_Size_Train].to(device)
                y_batch = y_t[j * Batch_Size_Train: (j + 1) * Batch_Size_Train].to(device)
            output = model[i](X_batch)
            loss = loss_function(output, y_batch)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 1:
            print(f'Epoch is {epoch + 1} / {Epochs}, loss is {total_loss}')

    print(f'[adv]: model for {name[i]} trained')

    # 对验证集的测试
    model[i].eval()
    prediction = []
    actual = []
    for j in range(days - train_size - Train_Window):
        X_batch = X_v[j][None, :, :].to(device)
        output = model[i](X_batch)
        prediction.append(output.item())
        actual.append(y_v[j].item())

    # 模型评价
    if i == 0:
        print('evaluation of high_frequency series is: ')
    elif i == 1:
        print('evaluation of medium_frequency series is: ')
    else:
        print('evaluation of trend is: ')
    evaluate(prediction, actual)

    # 对全部数据的测试
    prediction_all = []
    for j in range(len(X)):
        X_batch = X[j][None, :, :].to(device)
        output = model[i](X_batch)
        prediction_all.append(output.item())

    # 预测下一天的数据
    output = model[i](X_last.to(device))

    # 数据去归一化
    '''
    prediction_all = scaler[i].inverse_transform(np.array(prediction_all).reshape(1, -1))
    '''
    prediction_all = denormalize(np.array(prediction_all).reshape(-1, 1), i)
    forecasting_all.append(prediction_all.tolist())
    next_day += denormalize(np.array(output.item()).reshape(-1, 1), i)[0][0]

print(f'[predict]: Expecting index of the next day is {next_day}')

t = range(len(all_data[0]))
plt.clf()
#   0 1
# 0 1 2
# 1 3 4
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for i in range(3):
    xx = int(i/2)
    yy = i - 1 - xx
    axes[xx][yy].plot(t, all_data[i], color='black', label='actual', linewidth=2)
    axes[xx][yy].plot(t[-len(forecasting_all[i]):], forecasting_all[i], color='green', label='forecasting for all data')
    axes[xx][yy].plot(train_size, forecasting_all[i][train_size], color='red', label='dividing line', marker='+')
    axes[xx][yy].set_title(f'{name[i]}')
    axes[xx][yy].set_xlabel('time')
    axes[xx][yy].set_ylabel('index value')
    axes[xx][yy].legend()

plus_a = []
plus_p = []
for i in range(len(all_data[0])):
    plus_a.append(float(all_data[0][i]) + float(all_data[1][i]) + float(all_data[2][i]))
for i in range(len(forecasting_all[0])):
    plus_p.append(forecasting_all[0][i][0] + forecasting_all[1][i][0] + forecasting_all[2][i][0])
axes[1][1].plot(range(len(plus_a)), plus_a, color='black', label='actual')
axes[1][1].plot(range(len(plus_a))[-len(plus_p):], plus_p, color='green', label='prediction')
axes[1][1].set_title('summary')
axes[1][1].set_xlabel('time')
axes[1][1].set_ylabel('index value')
axes[1][1].legend()

plt.tight_layout()
plt.savefig('prediction.png')
plt.show()
