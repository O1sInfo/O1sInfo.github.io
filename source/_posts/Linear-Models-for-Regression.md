---
title: Linear Models for Regression
date: 2018-09-21 20:10:45
tags: 回归
categories: 机器学习
mathjax: true
---
## 1. Linear Basis Function Models

对于回归任务最简单的线性模型是：输入变量的线性组合。

$$y(\vec x, \vec w) = w_0+ w_1x_1 + ... + w_Dx_D \tag{3.1}$$

这就是我们所说的线性回归。

这个模型既是参数的线性函数也是输入的线性函数，然而这会带来很多限制。因此我们对输入变量进行一个非线性处理。

$$y(\vec x, \vec w) = w_0 + \sum_{j=1}^{M-1}w_j\phi_j(\vec x) = \sum_{j=0}^{M-1}w_j\phi_j(\vec x) = \vec w^T \phi(\vec x) \tag{3.2}$$

这里的 $\phi_j(\vec x)$ 就是所说 **基函数（basis function）** , 其中 $\phi_0(\vec x) = 1$, 注意 $\vec w, \vec \phi$ 均为列向量。

在一些实际的模式识别应用中，我们将会对原始输入变量应用一些形式的预处理或者特征提取，例如这里的 $\{\phi_j(\vec x)\}$。

基函数的选择有很多种形式，比如：

$$\phi_j(x) = exp\{-\frac{(x - \mu_j)^2}{2s^2}\} \tag{3.3}$$

$\mu_j$ 决定了基函数在输入空间的位置，$s$决定了空间的范围. 但这些参数都不是重要的，因为它们还要乘以一个自适应的系数 $w_j$.

$$\phi_j = \sigma(\frac{x- \mu_j}{s}) \tag{3.4}\\ \sigma(a) = \frac{1}{1 + exp(-a)}$$

除此之外还有傅里叶基函数，比如对sin函数的扩展。每一个基函数表示一个具体的频率，并且在空间上是无限的。相对地，被限定在有限的输入空间上的基函数由多个频率组合而成。在信号处理领域，考虑在空间和频率上都是有限的基函数是很有用的，它们被称为 **小波(waveles)**。

### 1.1 Maximum likelihood and least squares

我们假设目标变量t由下式得到

$$t = y(\vec x, \vec w) + \epsilon \tag{3.5}$$

$\epsilon$ 是一个零均值的高斯随机变量，其精度为 $\beta = \frac{1}{\sigma ^2}$
因此

$$p(t|\vec x, \vec w, \beta) = \mathcal N(t|y(\vec x, \vec w), \beta^{-1}) \tag{3.6}$$

如果我们的损失函数是平方损失，那么最优的预测就是目标变量的条件期望。在（3.6）式的高斯条件分布下，它的条件期望为

$$\mathbb E[t|\vec x] = \int tp(t|\vec x)dt = y(\vec x, \vec w) \tag{3.7}$$

注意，高斯噪声这样的假设暗示着给定x下t的条件期望是单峰的，这可能在一些应用上不适用。

现在假设我们由N个观察到的输入数据 $\mathbf X = \{\vec x_1, ..., \vec x_N\}$ 相对应的目标值是 $t_1, ..., t_N$. 这里把（3.6）写成矩阵形式得到的似然函数为：

$$p(t|\mathbf x, \vec w, \beta) = \prod_{n=1}^{N}\mathcal N(t_n|\vec W^T \phi(\vec x_n), \beta^{-1}) \tag{3.8}$$

注意在监督学习问题中（分类和回归），我们并不要求对输入变量的分布建模。因此x可能不会出现在条件变量上。例如 $p(t|\vec w, \beta)$

对式（3.8）取对数有：

$$\ln p(t|w, \beta) = \sum_{n=1}^{N}\mathcal N(t_n|\vec W^T \phi(\vec x_n), \beta^{-1})\\= \frac{N}{2}\ln \beta - \frac{N}{2}\ln(2\pi) - \beta E_D(\vec w) \tag{3.8}$$

这里平方和误差为

$$E_D(\vec w) = \frac{1}{2} \sum_{n=1}^{N} \{t_n - \vec w^T \phi(\vec x_n)\}^2 \tag{3.9}$$

最大化似然函数可通过对似然函数求导：

$$\frac{\partial \ln p(\vec t |\vec w, \beta)}{\partial \vec w} = \sum_{n=1}^{N} \{t_n - \vec w^T \phi(\vec x_n)\} \phi(\vec x_n)^T \tag{3.10}$$

令导数为0得到：

$$\vec w_{ML} = (\Phi ^T \Phi)^{-1} \Phi ^T \vec t \tag{3.11}$$

这个方程被称为 **正则方程(normal equations)** .

$$\Phi = \left [ \begin{matrix} \phi_0(\vec x_1) & \phi_1(\vec x_1) ... \phi_{M-1}(\vec x_1)\\\phi_0(\vec x_2) & \phi_1(\vec x_2) ... \phi_{M-1}(\vec x_2)\\... & ...\\\phi_0(\vec x_N) & \phi_1(\vec x_N) ... \phi_{M-1}(\vec x_N)\end{matrix}\right] \tag{3.12}$$

$\Phi ^+ = ((\Phi ^T \Phi)^{-1} \Phi ^T)$ 被称为矩阵 $\Phi$ 的 **Moor-Penrose pseudo-inverse**

同样我们也能得到 $\beta$ 的最大似然估计量。
$$\frac{1}{\beta_{ML}} = \frac{1}{N}\sum_{n=1}^{N}\{t_n - \vec w_{ML}^T \phi(\vec x_n)\}^2$$

### 1.2 Geometry of least squares

考虑这样一个N维空间，其以$t_n$为坐标轴，因此 $\vec t = (t_1, ..., t_N)^T$ 是该空间的一个向量。在N个数据上的每一个基函数 $\phi_j(\vec x_n)$ 也能表示为相同空间中的向量。我们定义 $\vec y$ 是一个N维向量，它的第n个元素是 $y(\vec x_n, \vec w)$. 由于 $\vec y$ 是 $\phi_j(\vec x_n)$ 在M维空间的任意线性组合。误差平方和等价于 $\vec y$和 $t$ 之间的欧几里得距离。 因此最小二乘解就是选择在子空间中最接近 $\vec t$的 $\vec y$.
