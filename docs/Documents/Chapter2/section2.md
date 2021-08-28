复杂度和学习率

## 神经网络复杂度

神经网络(NN)复杂度：多用神经网络层数 和神经网络参数的个数表示

1. 空间复杂度
2. 时间复杂度

下面以图为例子计算神经网络的复杂度

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20200601183556407.png" alt="image-20200601183556407" style="zoom:67%;" />



#### **空间复杂度:**

层数=隐藏层的层数+ 1个输出层

图中为：2层NN

总参数=总w+总b

- 第1层：3x4+4
- 第2层：4x2+2

图中共计：3x4+4 +4x2+2 = 26

#### **时间复杂度:**

乘加运算次数

- 第1层：3x4
- 第2层：4x2

图中共计：3x4 + 4x2 = 20



## 学习率

学习率($lr$)决定了神经网络更新参数的步伐大小
$$
w_{t+1}=w_{t}-l r * \frac{\partial L o s s}{\partial w_{t}}
$$

参数说明

- $w_{t+1}$ 更新后的参数
- $w_{t}$ 当前参数
- $lr$ 学习率
- $\frac{\partial \text { Loss }}{\partial w_{t}}$ 损失函数的梯度（偏导数）

### 指数衰减学习率

- 较大的学习率可以比较快速接近最优解，但是精确度比较低
- 较小的学习率迭代比较慢，但是精确度比较高

所以我们可以先用较大的学习率，快速得到较优解，然后逐步减小学习率，使模型在训练后期稳定。
$$
\text { 指数衰减学习率=初始学习率＊学习率衰减率 }^{(当 前 轮 数 / 多 少 轮 衰 减 一 次) ~}
$$
代码示例

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息

import tensorflow as tf

w = tf.Variable(tf.constant(5, dtype=tf.float32))

epoch = 40
LR_BASE = 0.2  # 最初学习率
LR_DECAY = 0.99  # 学习率衰减率
LR_STEP = 1  # 喂入多少轮BATCH_SIZE后，更新一次学习率

for epoch in range(epoch):  
# for epoch 定义顶层循环，表示对数据集循环epoch次，此例数据集数据仅有1个w,初始化时候constant赋值为5，循环40次迭代。
    lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)
    with tf.GradientTape() as tape:  # with结构到grads框起了梯度的计算过程。
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)  # .gradient函数告知谁对谁求导

    w.assign_sub(lr * grads)  
    # .assign_sub 对变量做自减 即：w -= lr*grads 即 w = w - lr*grads
    print("After %s epoch,w is %f,loss is %f,lr is %f" % (epoch, w.numpy(), loss, lr))

```

运行结果，学习率lr在指数衰减

```python
After 0 epoch,w is 2.600000,loss is 36.000000,lr is 0.200000
After 1 epoch,w is 1.174400,loss is 12.959999,lr is 0.198000
After 2 epoch,w is 0.321948,loss is 4.728015,lr is 0.196020
After 3 epoch,w is -0.191126,loss is 1.747547,lr is 0.194060
After 4 epoch,w is -0.501926,loss is 0.654277,lr is 0.192119
After 5 epoch,w is -0.691392,loss is 0.248077,lr is 0.190198
After 6 epoch,w is -0.807611,loss is 0.095239,lr is 0.188296
After 7 epoch,w is -0.879339,loss is 0.037014,lr is 0.186413
After 8 epoch,w is -0.923874,loss is 0.014559,lr is 0.184549
After 9 epoch,w is -0.951691,loss is 0.005795,lr is 0.182703
After 10 epoch,w is -0.969167,loss is 0.002334,lr is 0.180876
After 11 epoch,w is -0.980209,loss is 0.000951,lr is 0.179068
After 12 epoch,w is -0.987226,loss is 0.000392,lr is 0.177277
After 13 epoch,w is -0.991710,loss is 0.000163,lr is 0.175504
After 14 epoch,w is -0.994591,loss is 0.000069,lr is 0.173749
After 15 epoch,w is -0.996452,loss is 0.000029,lr is 0.172012
After 16 epoch,w is -0.997660,loss is 0.000013,lr is 0.170292
After 17 epoch,w is -0.998449,loss is 0.000005,lr is 0.168589
After 18 epoch,w is -0.998967,loss is 0.000002,lr is 0.166903
After 19 epoch,w is -0.999308,loss is 0.000001,lr is 0.165234
After 20 epoch,w is -0.999535,loss is 0.000000,lr is 0.163581
After 21 epoch,w is -0.999685,loss is 0.000000,lr is 0.161946
After 22 epoch,w is -0.999786,loss is 0.000000,lr is 0.160326
After 23 epoch,w is -0.999854,loss is 0.000000,lr is 0.158723
After 24 epoch,w is -0.999900,loss is 0.000000,lr is 0.157136
After 25 epoch,w is -0.999931,loss is 0.000000,lr is 0.155564
After 26 epoch,w is -0.999952,loss is 0.000000,lr is 0.154009
After 27 epoch,w is -0.999967,loss is 0.000000,lr is 0.152469
After 28 epoch,w is -0.999977,loss is 0.000000,lr is 0.150944
After 29 epoch,w is -0.999984,loss is 0.000000,lr is 0.149434
After 30 epoch,w is -0.999989,loss is 0.000000,lr is 0.147940
After 31 epoch,w is -0.999992,loss is 0.000000,lr is 0.146461
After 32 epoch,w is -0.999994,loss is 0.000000,lr is 0.144996
After 33 epoch,w is -0.999996,loss is 0.000000,lr is 0.143546
After 34 epoch,w is -0.999997,loss is 0.000000,lr is 0.142111
After 35 epoch,w is -0.999998,loss is 0.000000,lr is 0.140690
After 36 epoch,w is -0.999999,loss is 0.000000,lr is 0.139283
After 37 epoch,w is -0.999999,loss is 0.000000,lr is 0.137890
After 38 epoch,w is -0.999999,loss is 0.000000,lr is 0.136511
After 39 epoch,w is -0.999999,loss is 0.000000,lr is 0.135146
```

## 

















