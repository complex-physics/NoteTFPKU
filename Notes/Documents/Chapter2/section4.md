### 损失函数

损失函数(loss)的定义 ：预测值(y) 与已知答案(y_) 的差距

NN优化目标：loss最小

常见的三种损失函数

- mse (Mean Squared Error)均方误差
- 自定义
- ce (Cross Entropy)交叉熵

### 均方误差mes

$$
\operatorname{MSE}\left(y_{-}, y\right)=\frac{\sum_{i=1}^{n}\left(y-y_{-}\right)^{2}}{n}
$$

```
loss_mse = tf.reduce_mean(tf.square(y_ - y))
```

#### 例子

> 预测酸奶日销量y, x1、 x2是影响日销量的因素。

1. 建模前，应预先采集的数据有:每日x1、x2和销量y_ (即已知答案，最佳情况:产量=销量) 拟造数据集X，Y_ : y_ =x1 + x2，噪声: -0.05~ +0.05
2. 拟合可以预测销量的函数

代码示例

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息

import tensorflow as tf
import numpy as np

SEED = 23455

rdm = np.random.RandomState(seed=SEED)  # 生成[0,1)之间的随机数
x = rdm.rand(32, 2)
y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in x]  # 生成噪声[0,1)/10=[0,0.1); [0,0.1)-0.05=[-0.05,0.05)
x = tf.cast(x, dtype=tf.float32)

w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))

epoch = 15000
lr = 0.002

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)
        loss_mse = tf.reduce_mean(tf.square(y_ - y))

    grads = tape.gradient(loss_mse, w1)
    w1.assign_sub(lr * grads)

    if epoch % 500 == 0:
        print("After %d training steps,w1 is " % (epoch))
        print(w1.numpy(), "\n")
print("Final w1 is: ", w1.numpy())
```

运行结果

```python
......
After 14000 training steps,w1 is 
[[0.9993659]
 [0.999166 ]] 

After 14500 training steps,w1 is 
[[1.0002553 ]
 [0.99838644]] 

Final w1 is:  [[1.0009792]
 [0.9977485]]
```

### 自定义损失函数

如预测商品销量，预测多了，损失成本;预测少了，损失利润

若利润≠成本，则mse产生的loss无法利益最大化。

自定义损失函数，y_：标准答案数据集的，y：预测答案计算出的
$$
\operatorname{loss}\left(y_{-} y\right)=\sum_{n} f\left(y_， y\right)
$$

$$
f\left(y_{-}, y\right)=\left\{\begin{array}{lll}\text { PROFIT* }\left(y_{-}-y\right) & y<y_{-} & \text {预测的 } y\text { 少了, 损失利高(PROFIT) } \\ \text { COST } *\left(y-y_{-}\right) & y>=y_{-} & \text {预测的 } y \text { 多了，损失成本(COST) }\end{array}\right.
$$

如:预测酸奶销量，酸奶成本(COST) 1元，酸奶利润(PROFIT) 99元。

预测少了损失利润99元，大于预测多了损失成本1元。预测少了损失大，希望生成的预测函数往多了预测。

则损失函数为

```python
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * COST, (y_ - y) * PROFIT))
```

代码示例

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息

import tensorflow as tf
import numpy as np

SEED = 23455
COST = 1
PROFIT = 99

rdm = np.random.RandomState(SEED)
x = rdm.rand(32, 2)
y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in x]  # 生成噪声[0,1)/10=[0,0.1); [0,0.1)-0.05=[-0.05,0.05)
x = tf.cast(x, dtype=tf.float32)

w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))

epoch = 10000
lr = 0.002

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)
        loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * COST, (y_ - y) * PROFIT))

    grads = tape.gradient(loss, w1)
    w1.assign_sub(lr * grads)

    if epoch % 500 == 0:
        print("After %d training steps,w1 is " % (epoch))
        print(w1.numpy(), "\n")
print("Final w1 is: ", w1.numpy())
```

自定义损失函数，酸奶成本1元， 酸奶利润99元，成本很低，利润很高，人们希望多预测些，生成模型系数大于1，往多了预测。运行结果

```python
......
After 9000 training steps,w1 is 
[[1.1707956]
 [1.0338644]] 

After 9500 training steps,w1 is 
[[1.1611756]
 [1.0651482]] 

Final w1 is:  [[1.1626335]
 [1.1191947]]
```

自定义损失函数，酸奶成本99元， 酸奶利润1元，成本很高，利润很低，人们希望多少预测，生成模型系数小于1，往少了预测。运行结果

```python
......
After 9000 training steps,w1 is 
[[0.8042901]
 [0.7632218]] 

After 9500 training steps,w1 is 
[[0.8998995 ]
 [0.82509875]] 

Final w1 is:  [[0.9205433]
 [0.9186459]]
```

### 交叉熵损失函数

交义熵损失函数CE (Cross Entropy)：表征两个概率分布之间的距离
$$
\mathrm{H}\left(\mathrm{y}_{-}, \mathrm{y}\right)=-\sum y_{-} * \ln y
$$
eg.二分类已知答案y_ =(1, 0)，预测y1=(0.6, 0.4) y2=(0.8, 0.2)哪个更接近标准答案?
$$
\begin{aligned}
&\mathrm{H}_{1}((1,0),(0.6,0.4))=-(1 * \ln 0.6+0 * \ln 0.4) \approx-(-0.511+0)=0.511\\
&\mathrm{H}_{2}((1,0),(0.8,0.2))=-(1 * \ln 0.8+0 * \ln 0.2) \approx-(-0.223+0)=0.223
\end{aligned}
$$
因为H> H2，所以y2预测更准

```python
tf.losses.categorical crossentropy(y_ ，y)
```

代码示例

```python
import tensorflow as tf

loss_ce1 = tf.losses.categorical_crossentropy([1, 0], [0.6, 0.4])
loss_ce2 = tf.losses.categorical_crossentropy([1, 0], [0.8, 0.2])
print("loss_ce1:", loss_ce1)
print("loss_ce2:", loss_ce2)

# 交叉熵损失函数
```

运行结果

```python
loss_ce1: tf.Tensor(0.5108256, shape=(), dtype=float32)
loss_ce2: tf.Tensor(0.22314353, shape=(), dtype=float32)
```

### 交叉熵损失函数与softmax结合

输出先过softmax函数，再计算y与y_ 的交叉熵损失函数。

```python
tf.nn.softmax_cross_entropy_with_logits(y_, y)
```

代码示例

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息

# softmax与交叉熵损失函数的结合
import tensorflow as tf
import numpy as np

y_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
y = np.array([[12, 3, 2], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]])
y_pro = tf.nn.softmax(y)
loss_ce1 = tf.losses.categorical_crossentropy(y_,y_pro)
loss_ce2 = tf.nn.softmax_cross_entropy_with_logits(y_, y)

print('分步计算的结果:\n', loss_ce1)
print('结合计算的结果:\n', loss_ce2)


# 输出的结果相同
```

运行结果

```python
分步计算的结果:
 tf.Tensor(
[1.68795487e-04 1.03475622e-03 6.58839038e-02 2.58349207e+00
 5.49852354e-02], shape=(5,), dtype=float64)
结合计算的结果:
 tf.Tensor(
[1.68795487e-04 1.03475622e-03 6.58839038e-02 2.58349207e+00
 5.49852354e-02], shape=(5,), dtype=float64)
```

