import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim

import torchbnn as bnn

import matplotlib.pyplot as plt

# 导入数据集
data = pd.read_csv('winequality-red.csv', sep=';')  # 确保分隔符正确
X = data.iloc[:, 0:11].values  # 选择前11个特征
Y = data.iloc[:, 11].values - 1  # 选择'quality'列作为标签，并减一

# 数据预处理
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
X_train, x_val, Y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=132)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

x, y = torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long()
x_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).long()
x_val, y_val = torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long()

train_acc = []
train_loss = []
val_acc = []
val_loss = []

# 定义模型

model = nn.Sequential(
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=11, out_features=100), # 第一层，输入维度为11，输出维度为50
    nn.ReLU(), # 激活函数
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=50), # 第二层，输入维度为50，输出维度为20
    nn.ReLU(), # 激活函数
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=50, out_features=10), # 输出层，输入维度为10，输出维度为10
)


# 定义损失函数
ce_loss = nn.CrossEntropyLoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.005

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for step in range(5000):
    pre = model(x)
    ce = ce_loss(pre, y)
    kl = kl_loss(model)
    cost = ce + kl_weight * kl

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 输出训练的准确率和损失值
    _, predicted = torch.max(pre.data, 1)
    total = y.size(0)
    correct = (predicted == y).sum()
    print('- Accuracy: %f %%' % (100 * float(correct) / total))
    print('- CE : %2.2f, KL : %2.2f' % (ce.item(), kl.item()))
    # 将准确率和损失值添加到列表中
    train_acc.append(100 * float(correct) / total)
    train_loss.append(ce.item() + kl_weight * kl.item())

    # 使用验证集计算验证的准确率和损失值
    pre_val = model(x_val)
    ce_val = ce_loss(pre_val, y_val)
    kl_val = kl_loss(model)
    cost_val = ce_val + kl_weight * kl_val
    _, predicted_val = torch.max(pre_val.data, 1)
    total_val = y_val.size(0)
    correct_val = (predicted_val == y_val).sum()

    # 将验证的准确率和损失值添加到列表中
    val_acc.append(100 * float(correct_val) / total_val)
    val_loss.append(cost_val.item())

# 绘制训练和验证的准确率曲线
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()

# 绘制训练和验证的损失曲线
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.show()
# 评估模型
pre = model(x_test)
_, predicted = torch.max(pre.data, 1)
total = y_test.size(0)
correct = (predicted == y_test).sum()
print('- Test Accuracy: %f %%' % (100 * float(correct) / total))


# 绘制预测结果图
def draw_plot(X, Y, predicted):
    fig = plt.figure(figsize=(16, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    z1_plot = ax1.scatter(X[:, 0], X[:, 1], c=Y)
    z2_plot = ax2.scatter(X[:, 0], X[:, 1], c=predicted)

    plt.colorbar(z1_plot, ax=ax1)
    plt.colorbar(z2_plot, ax=ax2)

    ax1.set_title("REAL")
    ax2.set_title("PREDICT")

    plt.show()

draw_plot(X_test, Y_test, predicted.numpy())

y_true = y_test.numpy().flatten()
y_pred = predicted.numpy()
labels = list(range(10))
C = confusion_matrix(y_true, y_pred, labels=labels)

plt.imshow(C, cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(labels)
plt.yticks(labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion matrix')

# 在每个单元格里添加数值
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, C[i, j], ha='center', va='center', color='red')

plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 计算各个类别的Precision
precision = precision_score(y_true, y_pred, labels=labels, average=None)
print('Precision by class:', precision)

# 计算各个类别的Recall
recall = recall_score(y_true, y_pred, labels=labels, average=None)
print('Recall by class:', recall)

# 计算各个类别的F1-score
f1 = f1_score(y_true, y_pred, labels=labels, average=None)
print('F1-score by class:', f1)

from tabulate import tabulate

# 将指标结果转换为列表
metrics = [precision, recall, f1]
metrics_list = []
for metric in metrics:
    metrics_list.append(list(metric))

# 使用tabulate函数输出为表格
headers = ['Class ' + str(i) for i in labels]
rows = ['Precision', 'Recall', 'F1-score']
print(tabulate(metrics_list, headers, showindex=rows))
