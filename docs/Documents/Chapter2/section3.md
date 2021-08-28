激活函数

> 为什么要用激活函数

在神经网络中，如果不对上一层结点的输出做非线性转换的话，两个线性函数的组合还是线性函数。所以再深的网络也是线性模型，只能把输入线性组合再输出，不能学习到复杂的映射关系，因此需要使用激活函数这个非线性函数做转换。

> 常见的激活函数

1. Sigmoid函数
2. Tanh函数
3. Relu函数
4. Leaky Relu函数



## Sigmoid函数

$$
\begin{aligned}
\operatorname{sigmod}(x)=\frac{1}{1+e^{-x}} \in(0,1)
\end{aligned}
$$

求导
$$
\begin{aligned}
\operatorname{sigmod}^{\prime}(x)=&\operatorname{sigmod}(x)\times(1-\operatorname{sigmod}(x))
\\=&\frac{1}{1+e^{-x}}\times \frac{e^{-x}}{1+e^{-x}}
\\=&\frac{e^{-x}}{\left(1+e^{-x}\right)^{2}} \in(0,0.25)
\end{aligned}
$$
代码

```python
tf.nn.sigmoid(x)
```

sigmoid函数图像

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/v2-15ef91c7563ef2a046de444a904f1ff8_720w.jpg" style="zoom:80%;" />

sigmoid导数图像

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/v2-4b322e9c5d48a434c8a400d96a1de5fd_720w.jpg" style="zoom:80%;" />

目前使用sigmoid函数为激活函数的神经网络已经很少了

### 特点

(1)易造成梯度消失

​		深层神经网络更新参数时，需要从输入层到输出层，逐层进行链式求导，而sigmoid函数的导数输出为[0,0.25]间的小数，链式求导需要多层导数连续相乘，这样会出现多个[0,0.25]间的小数连续相乘，从而造成结果趋于0，产生梯度消失，使得参数无法继续更新。

(2)输出非0均值，收敛慢

​		希望输入每层神经网络的特征是以0为均值的小数值，但sigmoid函数激活后的数据都时整数，使得收敛变慢。

(3)幂运算复杂，训练时间长

​		sigmoid函数存在幂运算，计算复杂度大。

## Tanh函数

$$
\begin{array}{l}
\tanh (x)=\frac{1-e^{-2 x}}{1+e^{-2 x}} \in(-1,1) \\
\tanh ^{\prime}(x)=1-(\tanh (x))^{2}=\frac{4 e^{-2 x}}{\left(1+e^{-2 x}\right)^{2}} \in(0,1]
\end{array}
$$

```python
tf.math.tanh(x)
```

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20200601183826652.png" style="zoom:80%;" />



### 特点

(1)输出是0均值

(2)易造成梯度消失

(3)幂运算复杂，训练时间长

## Relu函数

$$
\begin{array}{l}
r e l u(x)=\max (x, 0)=\left\{\begin{array}{l}
x, \quad x \geq 0 \\
0, \quad x<0
\end{array} \in[0,+\infty)\right. \\
r e l u^{\prime}(x)=\left\{\begin{array}{ll}
1, & x \geq 0 \\
0, & x<0
\end{array} \in\{0,1\}\right.
\end{array}
$$

```
tf.nn.relu(x)
```



<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20200601183848839.png" style="zoom:80%;" />

**优点:**

1. 解决了**梯度消失**问题(在正区间)
2. 只 需判断输入是否大于0，计算速度快
3. 收敛速度远快于sigmoid和tanh

**缺点:**

1. 输出非0均值，收敛慢
2. Dead ReIU问题:某些神经元可能永远不会被激活，导致相应的参数永远不能被更新。

## Leaky Relu函数

$$
\begin{aligned}
&\text { LeakyReLU }(x)=\left\{\begin{array}{ll}
x, & x \geq 0 \\
a x, & x<0
\end{array} \in R\right.\\
&\text { LeakyReL } U^{\prime}(x)=\left\{\begin{array}{ll}
1, & x \geq 0 \\
a, & x<0
\end{array} \in\{a, 1\}\right.
\end{aligned}
$$

```python
tf.nn.leaky_relu(x)
```



<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20200601183910439.png" style="zoom:80%;" />

理论上来讲，Leaky Relu有Relu的所有优点，外加不会有Dead Relu问题，但是在实际操作当中，并没有完全证明Leaky Relu总是好于Relu。

## 总结

- 首选relu激活函数;
- 学习率设置较小值;
- 输入特征标准化，即让输入特征满足以0为均值，1为标准差的正态分布;
- 初始参数中心化，即让随机生成的参数满足以0为均值，下式为标准差的正态分布

$$
\sqrt{\frac{2}{\text { 当前层输入特征个数 }}}
$$







































