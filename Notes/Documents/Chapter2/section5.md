## 过拟合与欠拟合

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20200601183952319.png" style="zoom:80%;" />

欠拟合的解决方法:

- 增加输入特征项
- 增加网络参数
- 减少正则化参数

过拟合的解决方法:

- 数据清洗
- 增大训练集
- 采用正则化
- 增大正则化参数

## 正则化缓解过拟合

正则化在损失函数中引入模型复杂度指标，利用给W加权值，弱化了训练数据的噪声(一般不正则化b)
$$
\operatorname{loss}=\operatorname{loss}\left(\mathrm{y}-{y}_{-}\right)+\mathrm{REGULARIZER}{*} \operatorname{loss}(\mathrm{w})
$$
式中含义：

- `loss(y-y_)`：模型中所有参数的损失函数。如:交叉熵、均方误差

- `REGULARIZER`：用超参数REGULARIZER给出参数w在总loss中的比例，即正则化的权重

`loss(w)`：需要正则化的参数。计算方式有两种
$$
\operatorname{loss}_{L_{1}}(w)=\sum_{i}\left|w_{i}\right|
$$

$$
\operatorname{loss}_{L 2}(w)=\sum_{i}\left|w_{i}^{2}\right|
$$

正则化的选择

- L1正则化大概率会使很多参数变为零，因此该方法可通过稀疏参数，即减少参数的数量，降低复杂度。
- L2正则化会使参数很接近零但不为零，因此该方法可通过减小参数值的大小降低复杂度。
  - `tf.nn.l2_loss(w)`



### 未用正则化

代码示例，未采用正则化`p29_regularizationfree.py`

```python
# 导入所需模块
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# 读入数据/标签 生成x_train y_train
df = pd.read_csv('dot.csv')
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])

# reshape(-1,x) -1是将一维数组转换为二维的矩阵，并且第二个参数是表示分成几列，
# 但是在reshape的时候必须让数组里面的个数和shape的函数做取余时值为零才能转换
x_train = np.vstack(x_data).reshape(-1,2)
y_train = np.vstack(y_data).reshape(-1,1)  #将y_data转换为二维数组


Y_c = [['red' if y else 'blue'] for y in y_train]  # 三元运算

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型问题报错
x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

# from_tensor_slices函数切分传入的张量的第一个维度，生成相应的数据集，使输入特征和标签值一一对应
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

# 生成神经网络的参数，输入层为2个神经元，隐藏层为11个神经元，1层隐藏层，输出层为1个神经元
# 隐藏层11个神经元为人为指定
# 用tf.Variable()保证参数可训练
w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)  # 隐藏层2个输入，11个输出
b1 = tf.Variable(tf.constant(0.01, shape=[11]))  # b的个数与w个数相同

w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)  # 输出层接收11个，输出1个
b2 = tf.Variable(tf.constant(0.01, shape=[1]))

lr = 0.01  # 学习率
epoch = 400  # 循环轮数

# 训练部分
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:  # 记录梯度信息

            h1 = tf.matmul(x_train, w1) + b1  # 记录神经网络乘加运算
            h1 = tf.nn.relu(h1)  # relu激活函数
            y = tf.matmul(h1, w2) + b2

            # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss = tf.reduce_mean(tf.square(y_train - y))

        # 计算loss对各个参数的梯度
        variables = [w1, b1, w2, b2]
        grads = tape.gradient(loss, variables)

        # 实现梯度更新
        # w1 = w1 - lr * w1_grad tape.gradient是自动求导结果与[w1, b1, w2, b2] 索引为0，1，2，3 
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    # 每20个epoch，打印loss信息
    if epoch % 20 == 0:
        print('epoch:', epoch, 'loss:', float(loss))

# 预测部分
print("*******predict*******")
# xx在-3到3之间以步长为0.01，yy在-3到3之间以步长0.01,生成间隔数值点
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
# 将xx , yy拉直，并合并配对为二维张量，生成二维坐标点
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)
# 将网格坐标点喂入神经网络，进行预测，probs为输出
probs = []
for x_test in grid:
    # 使用训练好的参数进行预测
    h1 = tf.matmul([x_test], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2  # y为预测结果
    probs.append(y)

# 取第0列给x1，取第1列给x2
x1 = x_data[:, 0]
x2 = x_data[:, 1]
# probs的shape调整成xx的样子
probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1, x2, color=np.squeeze(Y_c)) # squeeze去掉纬度是1的纬度,相当于去掉[['red'],['blue']],内层括号变为['red','blue']
# 把坐标xx yy和对应的值probs放入contour<[‘kɑntʊr]>函数，给probs值为0.5的所有点上色  plt点show后 显示的是红蓝点的分界线
plt.contour(xx, yy, probs, levels=[.5])  # 画出probs值为0.5轮廓线,levels:这个参数用于显示具体哪几条登高线
plt.show()

# 读入红蓝点，画出分割线，不包含正则化
# 不清楚的数据，建议print出来查看
```

运行结果

```
epoch: 0 loss: 1.6901788711547852
epoch: 20 loss: 0.06456395983695984
epoch: 40 loss: 0.0639718547463417
epoch: 60 loss: 0.054891664534807205
epoch: 80 loss: 0.037164993584156036
epoch: 100 loss: 0.0290686022490263
epoch: 120 loss: 0.026631897315382957
epoch: 140 loss: 0.025654718279838562
epoch: 160 loss: 0.025450214743614197
epoch: 180 loss: 0.02445397339761257
epoch: 200 loss: 0.02315516769886017
epoch: 220 loss: 0.02262507937848568
epoch: 240 loss: 0.02210732363164425
epoch: 260 loss: 0.02202308177947998
epoch: 280 loss: 0.022013641893863678
epoch: 300 loss: 0.02216213382780552
epoch: 320 loss: 0.02226211130619049
epoch: 340 loss: 0.022413412109017372
epoch: 360 loss: 0.022659024223685265
epoch: 380 loss: 0.02281317301094532
*******predict*******
```

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/Figure_1.png" alt="Figure_1" style="zoom:80%;" />

### L2正则化

代码示例，在训练部分采用L2正则化

```python
# 导入所需模块
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# 读入数据/标签 生成x_train y_train
df = pd.read_csv('dot.csv')
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])

x_train = x_data
y_train = y_data.reshape(-1, 1)

Y_c = [['red' if y else 'blue'] for y in y_train]

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型问题报错
x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

# from_tensor_slices函数切分传入的张量的第一个维度，生成相应的数据集，使输入特征和标签值一一对应
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

# 生成神经网络的参数，输入层为4个神经元，隐藏层为32个神经元，2层隐藏层，输出层为3个神经元
# 用tf.Variable()保证参数可训练
w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01, shape=[11]))

w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=[1]))

lr = 0.01  # 学习率为
epoch = 400  # 循环轮数

# 训练部分
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:  # 记录梯度信息

            h1 = tf.matmul(x_train, w1) + b1  # 记录神经网络乘加运算
            h1 = tf.nn.relu(h1)
            y = tf.matmul(h1, w2) + b2

            # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss_mse = tf.reduce_mean(tf.square(y_train - y))
            # 添加l2正则化
            loss_regularization = []
            # tf.nn.l2_loss(w)=sum(w ** 2) / 2
            loss_regularization.append(tf.nn.l2_loss(w1))
            loss_regularization.append(tf.nn.l2_loss(w2))
            # 求和
            # 例：x=tf.constant(([1,1,1],[1,1,1]))
            #   tf.reduce_sum(x)
            # >>>6
            # loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
            loss_regularization = tf.reduce_sum(loss_regularization)
            loss = loss_mse + 0.03 * loss_regularization # REGULARIZER = 0.03

        # 计算loss对各个参数的梯度
        variables = [w1, b1, w2, b2]
        grads = tape.gradient(loss, variables)

        # 实现梯度更新
        # w1 = w1 - lr * w1_grad
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    # 每200个epoch，打印loss信息
    if epoch % 20 == 0:
        print('epoch:', epoch, 'loss:', float(loss))

# 预测部分
print("*******predict*******")
# xx在-3到3之间以步长为0.01，yy在-3到3之间以步长0.01,生成间隔数值点
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
# 将xx, yy拉直，并合并配对为二维张量，生成二维坐标点
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)
# 将网格坐标点喂入神经网络，进行预测，probs为输出
probs = []
for x_predict in grid:
    # 使用训练好的参数进行预测
    h1 = tf.matmul([x_predict], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2  # y为预测结果
    probs.append(y)

# 取第0列给x1，取第1列给x2
x1 = x_data[:, 0]
x2 = x_data[:, 1]
# probs的shape调整成xx的样子
probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1, x2, color=np.squeeze(Y_c))
# 把坐标xx yy和对应的值probs放入contour<[‘kɑntʊr]>函数，给probs值为0.5的所有点上色  plt点show后 显示的是红蓝点的分界线
plt.contour(xx, yy, probs, levels=[.5])
plt.show()

# 读入红蓝点，画出分割线，包含正则化
# 不清楚的数据，建议print出来查看 
```

运行结果

```python
epoch: 0 loss: 1.530280351638794
epoch: 20 loss: 0.7782743573188782
epoch: 40 loss: 0.6781619191169739
epoch: 60 loss: 0.5953636765480042
epoch: 80 loss: 0.5263288617134094
epoch: 100 loss: 0.4674427807331085
epoch: 120 loss: 0.41659849882125854
epoch: 140 loss: 0.37269479036331177
epoch: 160 loss: 0.3337797522544861
epoch: 180 loss: 0.3002385199069977
epoch: 200 loss: 0.27038004994392395
epoch: 220 loss: 0.24350212514400482
epoch: 240 loss: 0.22041508555412292
epoch: 260 loss: 0.20032131671905518
epoch: 280 loss: 0.1829461306333542
epoch: 300 loss: 0.16758175194263458
epoch: 320 loss: 0.15422624349594116
epoch: 340 loss: 0.14259028434753418
epoch: 360 loss: 0.13238896429538727
epoch: 380 loss: 0.12349270284175873
*******predict*******
```

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/Figure_2.png" alt="Figure_2" style="zoom:80%;" />

补充内容：python matplotlib contour画等高线图

https://blog.csdn.net/Mr_Cat123/article/details/80677525