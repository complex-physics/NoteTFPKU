## 神经网络参数优化器

**优化器**：是引导神经网络更新参数的工具

**作用**：用来更新和计算影响模型训练和模型输出的网络参数，使其逼近或达到最优值，从而最小化(或最大化)损失函数

- 待优化参数w
- 损失函数loss
- 学习率lr
- 每次迭代个batch（每个batch包含 $2^n$ 组数据）
- t表示当前batch迭代的总次数

更新网络参数的一般流程

1. 计算t时刻损失函数关于当前参数的梯度

$$
g_{t}=\nabla \operatorname{loss}=\frac{\partial \operatorname{loss}}{\partial\left(w_{t}\right)}
$$

2. 计算t时刻一阶动量 $m_{ t }$ 和二阶动量 $\sqrt{V_{ t }}$

- 一阶动量:与梯度相关的函数
- 二阶动量:与梯度平方相关的函数

3. 计算t时刻下降梯度:

$$
\eta_{\mathrm{t}}=l r \cdot m_{\mathrm{t}} / \sqrt{V_{\mathrm{t}}}
$$

4. 计算t+1时刻参数

$$
w_{\mathrm{t}+1}=w_{t}-\eta_{t}=w_{t}-l r \cdot m_{t} / \sqrt{V_{t}}
$$



> 不同的优化器实质上只是定义了不同的一阶动量和二阶动量公式



## SGD 随机梯度下降

SGD (无momentum)，常用的梯度下降法。

- $m_{ t }=g_{ t }$ 一阶动量设为梯度
- $V_{ t }=1$ 二阶动量

$$
\boldsymbol{\eta}_{\mathrm{t}}=\boldsymbol{l} \boldsymbol{r} \cdot \boldsymbol{m}_{\mathrm{t}} / \sqrt{\boldsymbol{V}_{t}}=\boldsymbol{l} \boldsymbol{r} \cdot \boldsymbol{g}_{t}
$$

$$
\begin{aligned}
w_{t+1}=& w_{t}-\eta_{t} \\
&=w_{t}-l r \cdot m_{t} / \sqrt{v_{t}}=w_{t}-lr \cdot g_{t}
\end{aligned}
$$

即为
$$
\mathrm{w}_{\mathrm{t}+1}=w_{t}-l r * \frac{\partial l o s s}{\partial w_{t}}
$$

## SGDM

( SGDM (含momentum的SGD)，在SGD基础上 $m_{ t }=g_{ t }$ 增加一阶动量。
$$
m_{\mathrm{t}}=\beta \cdot m_{t-1}+(1-\beta) \cdot g_{t}
$$
其中

- $m_{ t }$ ：表示各时刻梯度方向的指数滑动平均值

- $\beta$ ：超参数，趋近于1，经验值为0.9
- $V_{ t }=1$ 二阶动量

$$
\begin{aligned}
\eta_{\mathrm{t}}=& \operatorname{lr} \cdot m_{\mathrm{t}} / \sqrt{V_{\mathrm{t}}}=\operatorname{lr} \cdot m_{\mathrm{t}} \\
&=\operatorname{lr} \cdot\left(\beta \cdot m_{\mathrm{t}-1}+(1-\beta) \cdot g_{\mathrm{t}}\right)
\end{aligned}
$$

$$
\begin{aligned}
w_{\mathrm{t}+1}=& w_{\mathrm{t}}-\eta_{\mathrm{t}} \\
=&w_{\mathrm{t}}-l r \cdot\left(\beta \cdot m_{\mathrm{t}-1}+(1-\beta) \cdot g_{\mathrm{t}}\right)
\end{aligned}
$$

## Adagrad

Adagrad, 在SGD基础上增加二阶动量
$$
m_{\mathrm{t}}=g_{\mathrm{t}}
$$
二阶动量是从开始到现在梯度平方的累计和:
$$
V_{t}=\sum_{\tau=1}^{t} g_{\tau}^{2}
$$

$$
\begin{array}{l}
\eta_{\mathrm{t}}=lr \cdot m_{t} /(\sqrt{V_{t}}) \\
\quad=lr \cdot g_{t} /(\sqrt{\sum_{\tau=1}^{t} g_{t}^{2}})
\end{array}
$$

$$
\begin{aligned}
w_{t+1}=& w_{t}-\eta_{t} \\
&=w_{t}-lr \cdot g_{t} /(\sqrt{\sum_{\tau=1}^{t} g_{t}^{2}})
\end{aligned}
$$

## RMSProp

RMSProp, SGD基础上增加二 阶动量
$$
m_{\mathrm{t}}=g_{\mathrm{t}}
$$
二阶动量v使用指数滑动平均值计算，表征的是过去一段时间的平均值
$$
V_{t}=\beta \cdot V_{t-1}+(1-\beta) \cdot g_{t}^{2}
$$

$$
\begin{aligned}
\eta_{t}=& l r \cdot m_{\mathrm{t}} / \sqrt{V_{\mathrm{t}}} \\
&=lr \cdot g_{t} /(\sqrt{\beta \cdot V_{t-1}+(1-\beta) \cdot g_{t}^{2}})
\end{aligned}
$$

$$
\begin{aligned}
w_{t+1} &=w_{t}-\eta_{t} \\
&=w_{t}-lr \cdot g_{t} /(\sqrt{\beta \cdot V_{t-1}+(1-\beta) \cdot g_{t}^{2}})
\end{aligned}
$$

## Adam

Adam，同时结合SGDM一阶动量和RMSProp二阶动量

一阶动量：
$$
m_{\mathrm{t}}=\beta_{1} \cdot m_{t-1}+\left(1-\beta_{1}\right) \cdot g_{t}
$$
修正一阶动量的偏差，t为从训练开始到当前时刻所经历的总batch数::
$$
\widehat{m}_{\mathrm{t}}=\frac{m_{\mathrm{t}}}{1-\beta_{1}^{t}}
$$
二阶动量：
$$
V_{t}=\beta_{2} \cdot V_{s t e p-1}+\left(1-\beta_{2}\right) \cdot g_{t}^{2}
$$
修正二阶动量的偏差，t为从训练开始到当前时刻所经历的总batch数:
$$
\widehat{V_{t}}=\frac{V_{t}}{1-\beta_{2}^{t}}
$$

$$
\begin{aligned}
\eta_{t}=& lr \cdot \widehat{m}_{\mathrm{t}} / \sqrt{\widehat{V}_{t}} \\
&=\operatorname{lr} \cdot \frac{m_{\mathrm{t}}}{1-\beta_{1}^{t}} / \sqrt{\frac{V_{t}}{1-\beta_{2}^{t}}}
\end{aligned}
$$

$$
\begin{aligned}
w_{t+1} &=w_{t}-\eta_{t} \\
&=w_{t}-l r \cdot \frac{m_{t}}{1-\beta_{1}^{t}} / \sqrt{\frac{v_{t}}{1-\beta_{2}^{t}}}
\end{aligned}
$$

## 优化器对比

> class2中代码p32-p40

#### SGD

loss图像

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/SGD_loss.png" alt="SGD_loss" style="zoom:67%;" />

acc图像

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/SGD_acc.png" alt="SGD_acc" style="zoom:67%;" />

耗时：12.678699254989624

#### SGDM

loss图像

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/sgdm_loss.png" alt="sgdm_loss" style="zoom:67%;" />

acc图像

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/sgdm_acc.png" alt="sgdm_acc" style="zoom:67%;" />

耗时：17.32265305519104

#### Adagrad

loss图像

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/Adagrad_loss.png" alt="Adagrad_loss" style="zoom:67%;" />

acc图像

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/Adagrad_acc.png" alt="Adagrad_acc" style="zoom:67%;" />

耗时：13.080469131469727

#### RMSProp

loss图像

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/RMSProp_loss.png" alt="RMSProp_loss" style="zoom:67%;" />

acc图像

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/RMSProp_acc.png" alt="RMSProp_acc" style="zoom:67%;" />

耗时：16.42955780029297

#### Adam

loss图像

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/adam_loss.png" alt="adam_loss" style="zoom:67%;" />

acc图像

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/adam_acc.png" alt="adam_acc" style="zoom:67%;" />

耗时：22.04225492477417  