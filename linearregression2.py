# author:zhang ming yi
# two dimmensions
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
# %matplotlib inline


# 生成数据集
num_inputs = 2 # x的维度，features的特征数
num_examples = 1000 # 样本数，features的长度
true_w = torch.tensor([2, -3.4]) # 给定的w和b的真实值
true_b = torch.tensor([4.2])
# 随机生成一个num_examples*num_inputs形状的numpy，再转成tensor，符合正态分布
features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs)))
# 用随机生成的features生成labels标签，也就是真实值
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# 加上高斯噪声模拟随机性
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))

# 作图函数
# 用矢量图显示
def use_svg_display():
    display.set_matplotlib_formats('svg')

# 设置图的尺寸
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show() # 画图

# 读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 打乱索引顺序
    for i in range(0, num_examples, batch_size):
        # 用j提取出一个batch_size(或者小一些)的索引链表转成tensor
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        # 生成(返回)j索引的数据，index_select()的第一个参数：0表示按行索引，1表示按列索引
        yield features.index_select(0, j), labels.index_select(0, j)

# mini_batch的大小
batch_size = 10

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.double)
b = torch.zeros(1,dtype=torch.double)
w.requires_grad_(True)
b.requires_grad_(True)

# 定义模型
def linreg(X, w, b):
    return torch.mm(X, w) + b # X和w的数据类型要相同

# 定义损失函数
def square_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2 # 返回的是一个标量

# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


lr = 0.03
num_epochs = 10 # 迭代周期
linear = linreg
loss = square_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(linear(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(linear(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)



