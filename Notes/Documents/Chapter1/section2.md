## section2 神经网络设计过程

> 以鸢尾花分类为例

[TOC]



## 背景知识

鸢尾花有三种类型

- 0狗尾草鸢尾
- 1杂色鸢尾
- 2弗吉尼亚鸢尾

人们通过经验总结出了规律:通过测量花的花曹长、花尊宽、花瓣长、花瓣宽，可以得出鸢尾花的类别。

### 方法1 专家系统

if语句case语句——专家系统，把专家的经验告知计算机，计算机执行逻辑判别(理性计算)，给出分类。

> (如:花萼长>花尊宽且花瓣长花瓣宽>2则为1杂色鸢尾)

### 方法2 神经网络

1. 采集大量(输入特征：花萼长、花夢宽、花瓣长、花瓣宽，标签(需人工 标定)：对应的类别)数据对构成数据集。
2. 把数据集限入搭建好的神经网络结构，网络优化参 数得到模型，模型读入新输入特征，输出识别

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20200601175024207.png" style="zoom:80%;" />

## 用神经网络实现鸢尾花分类

主要流程

1. 网络参数 初始化
2. 喂入数据，前向传播
3. 损失函数(初体会)
4. 梯度下降(初体会)
5. 学习率(初体会)
6. 反向传播更新参数

在本次学习中，并不要求你从头到尾写代码，而给你代码运行。自己改变参数学习率，对流程有个大概了解。



### 1 初始化参数

1. 随机初始化网络参数
2. 打印参数的形状

代码示例：

```python
import tensorflow as tf

x1 = tf.constant([[5.8, 4.0, 1.2, 0.2]])  # 5.8,4.0,1.2,0.2（0）
w1 = tf.constant([[-0.8, -0.34, -1.4],
                  [0.6, 1.3, 0.25],
                  [0.5, 1.45, 0.9],
                  [0.65, 0.7, -1.2]])
b1 = tf.constant([2.52, -3.1, 5.62])
y = tf.matmul(x1, w1) + b1
print("x1.shape:", x1.shape)
print("w1.shape:", w1.shape)
print("b1.shape:", b1.shape)
print("y.shape:", y.shape)
print("y:", y)

#####以下代码可将输出结果y转化为概率值#####
y_dim = tf.squeeze(y)  # 去掉y中纬度1（观察y_dim与 y 效果对比）
y_pro = tf.nn.softmax(y_dim)  # 使y_dim符合概率分布，输出为概率值了
print("y_dim:", y_dim)
print("y_pro:", y_pro)

#请观察打印出的shape
```

运行结果

```python
x1.shape: (1, 4)
w1.shape: (4, 3)
b1.shape: (3,)
y.shape: (1, 3)
y: tf.Tensor([[ 1.0099998  2.008     -0.6600003]], shape=(1, 3), dtype=float32)
y_dim: tf.Tensor([ 1.0099998  2.008     -0.6600003], shape=(3,), dtype=float32)
y_pro: tf.Tensor([0.2563381  0.69540703 0.04825489], shape=(3,), dtype=float32)
```





<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20200601175159359.png" style="zoom:80%;" />



### 2 前向传播

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20200601175118615.png" style="zoom:80%;" />

图片用张量的形式表示，喂入神经网络之前，把张量拉成一维向量 $\vec{x}$
$$
\vec{y }= \vec{x} * \hat{w} + b
$$
从左往右每一层神经网络都是这个代数过程

### 3 损失函数

损失函数(loss function) ：预测值(y)与标准答案(y_ )的差距。

目的：损失函数可以定量判断W、b的优劣，当损失函数输出最小时，参数w、b会出现最优值。

本次使用的损失函数是均方误差：
$$
\operatorname{MSE}\left(y, y_{-}\right)=\frac{\sum_{k=0}^{n}(y-y)^{2}}{n}
$$

### 4 梯度下降

目的：想找到一组参数w和b,使得损失函数最小。

梯度：函数对各参数求偏导后的向量。函数梯度下降方向是函数减小方向。

梯度下降法：沿损失函数梯度下降的方向，寻找损失函数的最小值，得到最优参数的方法。
$$
\begin{array}{l}
w_{t+1}=w_{t}-l r * \frac{\partial l o s s}{\partial w_{t}} \\
b_{t+1}=b-l r * \frac{\partial l o s s}{\partial b_{t}} \\
w_{t+1} * x+b_{t+1} \rightarrow y
\end{array}
$$
学习率(learning rate, Ir) ：当学习率设置的过小时，收敛过程将变得十分缓慢。而当学习率设置的过大时，梯度可能会在最小值附近来回震荡，甚至可能无法收敛。

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20200601175413420.png" style="zoom:80%;" />

#### 反向传播

反向传播：从后向前，逐层求损失函数对每层神经元参数的偏导数，迭代更新所有参数。
$$
w_{t+1}=w_{t}-l r * \frac{\partial l o s s}{\partial w_{t}}
$$
例如损失函数为：
$$
\operatorname{loss}=(w+1)^{2}
$$
求偏导
$$
\frac{\partial \operatorname{loss}}{\partial w}=2 w+2
$$
参数w初始化为5，学习率为0.2则

| 次数 | 参数w | 结果                          |
| ---- | ----- | ----------------------------- |
| 1    | 5     | `5-0.2*（2*4+2）=2.6`         |
| 2    | 2.6   | `2.6-0.2*（2*2.6+2）=1.16`    |
| 3    | 1.16  | `1.16-0.2*（2*1.16+2）=0.296` |
| 4    | 0.296 | .......                       |

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20200601175508405.png" style="zoom:80%;" />

求出w的最佳值，使得损失函数最小，代码示例：

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息

import tensorflow as tf

w = tf.Variable(tf.constant(5, dtype=tf.float32))
lr = 0.2
epoch = 40

for i in range(epoch):  # for epoch 定义顶层循环，表示对数据集循环epoch次，此例数据集数据仅有1个w,初始化时候constant赋值为5，循环40次迭代。
    # 用with结构让损失函数loss对参数w求梯度
    with tf.GradientTape() as tape:  # with结构到grads框起了梯度的计算过程。
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)  # .gradient函数告知谁对谁求导,此处为loss函数对w求偏导

    w.assign_sub(lr * grads)  # .assign_sub 对变量做自减 即：w -= lr*grads 即 w = w - lr*grads
    print("After %s epoch,w is %f,loss is %f" % (i, w.numpy(), loss))

# lr初始值：0.2   请自改学习率  0.001  0.999 看收敛过程
# 最终目的：找到 loss 最小 即 w = -1 的最优参数w
```

运行结果

```python
After 0 epoch,w is 2.600000,loss is 36.000000
After 1 epoch,w is 1.160000,loss is 12.959999
After 2 epoch,w is 0.296000,loss is 4.665599
After 3 epoch,w is -0.222400,loss is 1.679616
After 4 epoch,w is -0.533440,loss is 0.604662
After 5 epoch,w is -0.720064,loss is 0.217678
After 6 epoch,w is -0.832038,loss is 0.078364
After 7 epoch,w is -0.899223,loss is 0.028211
After 8 epoch,w is -0.939534,loss is 0.010156
After 9 epoch,w is -0.963720,loss is 0.003656
After 10 epoch,w is -0.978232,loss is 0.001316
After 11 epoch,w is -0.986939,loss is 0.000474
After 12 epoch,w is -0.992164,loss is 0.000171
After 13 epoch,w is -0.995298,loss is 0.000061
After 14 epoch,w is -0.997179,loss is 0.000022
After 15 epoch,w is -0.998307,loss is 0.000008
After 16 epoch,w is -0.998984,loss is 0.000003
After 17 epoch,w is -0.999391,loss is 0.000001
After 18 epoch,w is -0.999634,loss is 0.000000
After 19 epoch,w is -0.999781,loss is 0.000000
After 20 epoch,w is -0.999868,loss is 0.000000
After 21 epoch,w is -0.999921,loss is 0.000000
After 22 epoch,w is -0.999953,loss is 0.000000
After 23 epoch,w is -0.999972,loss is 0.000000
After 24 epoch,w is -0.999983,loss is 0.000000
After 25 epoch,w is -0.999990,loss is 0.000000
After 26 epoch,w is -0.999994,loss is 0.000000
After 27 epoch,w is -0.999996,loss is 0.000000
After 28 epoch,w is -0.999998,loss is 0.000000
After 29 epoch,w is -0.999999,loss is 0.000000
After 30 epoch,w is -0.999999,loss is 0.000000
After 31 epoch,w is -1.000000,loss is 0.000000
After 32 epoch,w is -1.000000,loss is 0.000000
After 33 epoch,w is -1.000000,loss is 0.000000
After 34 epoch,w is -1.000000,loss is 0.000000
After 35 epoch,w is -1.000000,loss is 0.000000
After 36 epoch,w is -1.000000,loss is 0.000000
After 37 epoch,w is -1.000000,loss is 0.000000
After 38 epoch,w is -1.000000,loss is 0.000000
After 39 epoch,w is -1.000000,loss is 0.000000
```

## 









