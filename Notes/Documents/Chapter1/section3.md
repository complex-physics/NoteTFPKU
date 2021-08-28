## section 张量生成

> 了解张量的概念，以及用tensorflow创建张量，和张量的基本运算

张量(Tensor) ：多维数组(列表)

阶：张量的维数

| 维数 | 阶   | 名字        | 例子                              |
| ---- | ---- | ----------- | --------------------------------- |
| 0-D  | 0    | 标量 scalar | s=1 2 3                           |
| 1-D  | 1    | 向量 vector | v=[1, 2, 3]                       |
| 2-D  | 2    | 矩阵 matrix | m=`[[1,2,3],[4,5,6][7,8,9]]`      |
| N-D  | N    | 张量 tensor | t=[[[    有几个中括号就是几阶张量 |

## 数据类型

- `tf.int`, `tf.float` 浮点型
  - `tf.int 32`，`tf.float 32`, `tf.float 64`
- `tf.bool` 布尔型
  - `tf.constant([True, False])`
- `tf.string` 字符串
  - `tf.constant("Hello, world!")`

## 如何创建一个Tensor

用tensorflow

```
tf.constant(张量内容, dtype=数据类型（可选）)
```

例子

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息

import tensorflow as tf

a = tf.constant([1, 5], dtype=tf.int64)
print("a:", a) # 给出张量a的所有信息，
print("a.dtype:", a.dtype)
print("a.shape:", a.shape)

# 本机默认 tf.int32  可去掉dtype试一下 查看默认值
```

运行结果

```python
a: tf.Tensor([1 5], shape=(2,), dtype=int32)
a.dtype: <dtype: 'int32'>
a.shape: (2,)
```

### 与numpy数据类型转换

将numpy的数据类型转换为Tensor数据类型`tf. convert to_tensor(数据名，dtype=数 据类型(可选))`

代码示例

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息

import tensorflow as tf
import numpy as np

a = np.arange(0, 5)
b = tf.convert_to_tensor(a, dtype=tf.int64)
print("a:", a)
print("b:", b)
```

运行结果

```python
a: [0 1 2 3 4]
b: tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int64)
```

- **创建全为0的张量**
  - `tf. zeros(维度)`
- **创建全为1的张量**
  - `tf. ones(维度)`
- **创建全为指定值的张量**
  - `tf. fil(维度，指定值)`

维度：

- 一维直接写个数
- 二维用[行，列]
- 多维用[n,m,.K....]

代码示例：

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息

import tensorflow as tf

a = tf.zeros([2, 3])
b = tf.ones(4)
c = tf.fill([2, 2], 9)
print("a:", a)
print("b:", b)
print("c:", c)
```

运行结果

```python
a: tf.Tensor(
[[0. 0. 0.]
 [0. 0. 0.]], shape=(2, 3), dtype=float32)
b: tf.Tensor([1. 1. 1. 1.], shape=(4,), dtype=float32)
c: tf.Tensor(
[[9 9]
 [9 9]], shape=(2, 2), dtype=int32)
```

- **生成正态分布的随机数，默认均值为0，标准差为1**
  - `tf. random.normal (维度，mean=均值，stddev=标准差)`
- **生成截断式正态分布的随机数**
  - `tf. random.truncated normal (维度，mean=均值，stddev=标准差)`

在`tf.truncated normal`中如果随机生成数据的取值在(μ-2σ， μ+2σ) 之外,则重新进行生成，保证了生成值在均值附近。(μ:均值，σ:标准差）

标准差计算公式：
$$
\sigma=\sqrt{\frac{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}}{n}}
$$
代码示例：

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息

import tensorflow as tf

d = tf.random.normal([2, 2], mean=0.5, stddev=1)
print("d:", d)
e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print("e:", e)

```

运行结果

```python
d: tf.Tensor(
[[0.21293578 1.3744814 ]
 [0.57955813 0.21024475]], shape=(2, 2), dtype=float32)
e: tf.Tensor(
[[ 1.0601948 -0.290159 ]
 [ 1.4136508  1.1021997]], shape=(2, 2), dtype=float32)
```

**生成均匀分布随机数( minval, maxval )**
`tf. random. uniform(维度，minval=最小值，maxval=最大值)`

代码示例

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息

import tensorflow as tf

f = tf.random.uniform([2, 2], minval=0, maxval=1)
print("f:", f)
```

运行结果

```python
f: tf.Tensor(
[[0.8941778  0.1506716 ]
 [0.61614406 0.6378647 ]], shape=(2, 2), dtype=float32)
```

