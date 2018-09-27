---
title: Linear Models for Regression
date: 2018-09-21 20:10:45
tags: 回归
categories: 机器学习
mathjax: true
---
## Linear Basis Function Models

对于回归任务最简单的线性模型是：输入变量的线性组合。

$$y(\vec x, \vec w) = w_0+ w_1x_1 + ... + w_Dx_D \tag{3.1}$$

这就是我们所说的线性回归。

这个模型既是参数的线性函数也是输入的线性函数，然而这会带来很多限制。因此我们对输入变量进行一个非线性处理。

$$y(\vec x, \vec w) = w_0 + \sum_{j=1}^{M-1}w_j\phi_j(\vec x) = \sum_{j=0}^{M-1}w_j\phi_j(\vec x) = \vec w^T \phi(\vec x) \tag{3.2}$$

这里的 $\phi_j(\vec x)$ 就是所说 **基函数（basis function）** , 其中 $\phi_0(\vec x) = 1$, 注意 $\vec w, \vec \phi$ 均为列向量。

在许多模式识别的实际应⽤中，我们会对原始的数据变量进⾏某种固定形式的预处理或者特征抽取。如果原始变量由向量 x 组成，那么特征可以⽤基函数 $\{\phi_j(\vec x)\}$ 来表示。

基函数的选择有很多种形式，比如：

$$\phi_j(x) = exp\{-\frac{(x - \mu_j)^2}{2s^2}\} \tag{3.3}$$

$\mu_j$ 决定了基函数在输入空间的位置，$s$决定了空间的范围. 但这些参数都不是重要的，因为它们还要乘以一个自适应的系数 $w_j$.

$$\phi_j = \sigma(\frac{x- \mu_j}{s}) \tag{3.4}\\ \sigma(a) = \frac{1}{1 + exp(-a)}$$

除此之外还有傅里叶基函数，比如对sin函数的扩展。每一个基函数表示一个具体的频率，并且在空间上是无限的。相对地，被限定在有限的输入空间上的基函数由多个频率组合而成。在信号处理领域，考虑在空间和频率上都是有限的基函数是很有用的，它们被称为 **小波(waveles)**。

### Maximum likelihood and least squares

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

$$\ln p(t|w, \beta) = \sum_{n=1}^{N}\mathcal N(t_n|\vec W^T \phi(\vec x_n), \beta^{-1})\\ = \frac{N}{2}\ln \beta - \frac{N}{2}\ln(2\pi) - \beta E_D(\vec w) \tag{3.9}$$

这里平方和误差为

$$E_D(\vec w) = \frac{1}{2} \sum_{n=1}^{N} \{t_n - \vec w^T \phi(\vec x_n)\}^2 \tag{3.10}$$

最大化似然函数可通过对似然函数求导：

$$\frac{\partial \ln p(\vec t |\vec w, \beta)}{\partial \vec w} = \sum_{n=1}^{N} \{t_n - \vec w^T \phi(\vec x_n)\} \phi(\vec x_n)^T \tag{3.11}$$

令导数为0得到：

$$\vec w_{ML} = (\Phi ^T \Phi)^{-1} \Phi ^T \vec t \tag{3.12}$$

这个方程被称为 **正则方程(normal equations)** .

$$\Phi = \left [ \begin{matrix} \phi_0(\vec x_1) & \phi_1(\vec x_1) ... \phi_{M-1}(\vec x_1)\\ \phi_0(\vec x_2) & \phi_1(\vec x_2) ... \phi_{M-1}(\vec x_2)\\ ... & ...\\ \phi_0(\vec x_N) & \phi_1(\vec x_N) ... \phi_{M-1}(\vec x_N\end{matrix}\right] \tag{3.13}$$

$\Phi ^+ = ((\Phi ^T \Phi)^{-1} \Phi ^T)$ 被称为矩阵 $\Phi$ 的 **Moor-Penrose pseudo-inverse**

同样我们也能得到 $\beta$ 的最大似然估计量。
$$\frac{1}{\beta_{ML}} = \frac{1}{N}\sum_{n=1}^{N}\{t_n - \vec w_{ML}^T \phi(\vec x_n)\}^2$$

### Geometry of least squares

考虑这样一个N维空间，其以$t_n$为坐标轴，因此 $\vec t = (t_1, ..., t_N)^T$ 是该空间的一个向量。在N个数据上的每一个基函数 $\phi_j(\vec x_n)$ 也能表示为相同空间中的向量。我们定义 $\vec y$ 是一个N维向量，它的第n个元素是 $y(\vec x_n, \vec w)$. 由于 $\vec y$ 是 $\phi_j(\vec x_n)$ 在M维空间的任意线性组合。误差平方和等价于 $\vec y$和 $t$ 之间的欧几里得距离。 因此最小二乘解就是选择在子空间中最接近 $\vec t$的 $\vec y$.

![](/images/lmfr.PNG)

### Sequential learning

批处理技术使得可以一次处理很大的训练数据集。如果数据集是足够大，此时使用序列算法可能更有用，序列算法也被称作 **在线（on-line）算法**。这种算法在每一次表示后都更新模型的参数。序列学习适用于一些观察数据是连续产生的实时应用，这些应用必须利用目前所有观测到的数据来预测。

我们能应用随机梯度下降技术来获得序列学习算法。

$$w^{\tau + 1} = w^\tau - \eta \nabla E_n \tag{3.22}$$

这里 $\tau$ 是迭代的次数，$\eta$ 是学习率。对于（3.10）那样的平方和误差函数，上式可写为：

$$w^{\tau + 1} = w^\tau + \eta (t_n - w^{(\tau)T}\Phi_n)\Phi_n \tag{3.23}$$

$\Phi_n = \Phi(x_n)$ 这就是 **least-mean-squares(LMS)** 算法。

### Regularized learst squares

我们给误差函数增加一个正则项来控制模型的过拟合。新的误差函数为。

$$E_D(W) + \lambda E_w(w) \tag{3.24}$$

$\lambda$ 是正则系数用来控制由数据决定的误差和由正则项引入的误差的相对重要程度。一个最简单的正则形式为权重向量的平方和：

$$E_w(\vec w) = \frac{1}{2}\vec w^T \vec w \tag{3.25}$$

如果我们考虑平方和误差函数，那么总的误差为：

$$\frac{1}{2}\sum_{n=1}^{N}\{t_n - \vec w^T \Phi(\vec x_n)\}^2 + \frac{\lambda}{2}\vec w^T \vec w \tag{3.27}$$

在机器学习文献中正则的一个解释是 **权重衰减**，因为在序列学习中，它鼓励权重向朝着0的方向变化。
在统计学中它提供一种参数收缩的方法。它的优点是误差函数保持为w的二次型，因此可以得到精确的最小化解。

$$\vec w = (\lambda \mathbf I + \Phi^T\Phi)^{-1}\Phi^T \vec t \tag{3.28}$$

有时也会使用更一般的正则化形式如：

$$\frac{1}{2}\sum_{n=1}^{N}\{t_n - \vec w^T \Phi(\vec x_n)\}^2 + \frac{\lambda}{2}\sum_{j=1}^M |w_j|^q \tag{3.29}$$

$q=1$ 时为统计学中所说的lasso。它有这样的特点：如果 $\lambda$ 足够大，一些系数就被学习为0，与之对应的基函数就没有起作用。这学得一个稀疏的模型。

![](/images/lmfr_f3.PNG)

正则化允许在大小有限的数据集上训练复杂的模型而不导致严重的过拟合，这实质上是通过限制有效的模型复杂度来实现的。然而，决定最优模型复杂度的问题从寻找合适基函数的数量转变为决定一个合适的正则化系数 $\lambda$。

![](/images/lmfr_f4.PNG)

### Multiple outputs

到目前为止我们已经考虑了单个目标变量的例子，在某些应用中我们可能希望预测多个目标变量，我们将它们定义为一个目标向量 $\vec t$。这可能通过分别对每一个目标变量定义一组基函数，依照单变量回归那样做。然而一个更合适更常用的方法是，使用同样的一组基函数来对所有的目标变量建模。

$$y(\vec x, \vec w) = \mathbf W^T \phi(\vec x) \tag{3.31}$$

这里 $\vec y$ 是一个K维的列向量，$\mathbf W$ 是一个$M \times K$的参数矩阵。$\phi(\vec x)$ 是一个M维的列向量其中 $\phi_0(\vec x) = 0$.假设目标向量的条件分布是一个各向同性的高斯分布。

$$p(\vec t|\vec x, \mathbf W, \beta) = \mathcal N(\vec t|\mathbf W^T \phi(\vec x),\beta^{-1}\mathbf I) \tag{3.32}$$

如果我们有一组观察集 $\vec t_1, ..., \vec t_N$, 把它们组合进一个大小 $N\times K$ 的矩阵$T$. 则对数似然函数为：

$$\ln p(\mathbf T|\mathbf X, \mathbf W, \beta) = \sum_{n=1}^{N}\mathcal N(\vec t_n|\mathbf W^T \phi(\vec x_n), \beta^{-1}\mathbf I)\\ = \frac{NK}{2}\ln(\frac{\beta}{2\pi}) - \frac{\beta}{2}\sum_{n=1}^{N}||\vec t_n - \mathbf W^T\phi(\vec x_n)||^2 \tag{3.33}$$

我们最大化该似然函数得到：

$$\mathbf W_{ML} = (\Phi^T\Phi)^{-1}\Phi^T \mathbf T \tag{3.34}$$

如果我们测试每一个目标的结果我们有：

$$\vec w_k = (\Phi^T\Phi)^{-1}\Phi^T \vec t_k$$

$\vec t_k$ 是一个N维的列向量。

## The Bias-Variance Decomposition

在我们目前考虑的线性回归模型中，我们假定了基函数的数量和形式都是固定的。如果使⽤有限规模的数据集来训练复杂的模型，那么使⽤最⼤似然⽅法（或最⼩平⽅⽅法），会导致严重的过拟合问题。然⽽，通过限制基函数的数量来避免过拟合问题有⼀个负作⽤，即限制了模型描述数据中有趣且重要的规律的灵活性。虽然引⼊正则化项可以控制具有多个参数的模型的过拟合问题，但是这就产⽣了⼀个问题：如何确定正则化系数 $\lambda$ 的合适的值。同时关于权值 $w$ 和正则化系数 $\lambda$ 来最小化正则化的误差函数显然不是⼀个正确的⽅法，因为这样做会使得 $\lambda = 0$ ，从⽽产⽣非正则化的解。

从频率学家的观点考虑⼀下模型的复杂度问题被称为 **偏置-⽅差折中（ bias-variance trade-off ）**

当我们讨论回归问题的决策论时，我们考虑了不同的损失函数。⼀旦我们知道了
条件概率分布 $p(t | x)$ ，每⼀种损失函数都能够给出对应的最优预测结果。使⽤最多的⼀个选择是平方损失函数，此时最优的预测由条件期望给出，即：

$$h(\vec x) = \mathbb E[t | \vec x] = \int tp(t|\vec x)dt \tag{3.36}$$

现在，有必要区分决策论中出现的平⽅损失函数以及模型参数的最⼤似然估计中出现的平⽅和误差函数。我们可以使⽤⽐最⼩平⽅更复杂的⽅法，例如正则化或者纯粹的贝叶斯⽅法，来确定条件概率分布 $p(t | x)$ 。为了进⾏预测，这些⽅法都可以与平⽅损失函数相结合。

平方损失函数的期望可以写成：

$$\mathbb E[L] = \int\{y(\vec x) - h(\vec x)\}^2p(\vec x)d\vec x + \int \int \{h(\vec x - t)\}^2p(\vec x, t)d\vec x dt \tag{3.37}$$

回忆⼀下，与 y(x) ⽆关的第⼆项，是由数据本⾝的噪声造成的，表⽰期望损失能够达到的最⼩值。第⼀项与我们对函数 y(x) 的选择有关，我们要找⼀个 y(x) 的解，使得这⼀项最⼩。由于它是⾮负的，因此我们希望能够让这⼀项的最⼩值等于零。如果我们有⽆限多的数据（以及⽆限多的计算资源），那么原则上我们能够以任意的精度寻找回归函数 h(x) ，这会给出 y(x) 的最优解。然⽽，在实际应⽤中，我们的数据集 D 只有有限的 N 个数据点，从⽽我们不能够精确地知道回归函数 h(x) 。

如果我们使⽤由参数向量 w 控制的函数 y(x,w) 对 h(x) 建模，那么从贝叶斯的观点来看我们模型的不确定性是通过 w 的后验概率分布来表⽰的。但是，频率学家的⽅法涉及到根据数据集 D 对 w 进⾏点估计，然后试着通过下⾯的思想实验来表⽰估计的不确定性。假设我们有许多数据集，每个数据集的⼤⼩为 N ，并且每个数据集都独⽴地从分布 p(t,x) 中抽取。对于任意给定的数据集 D ，我们可以运⾏我们的学习算法，得到⼀个预测函数 y(x;D) 。不同的数据集给出不同的函数，从⽽给出不同的平⽅损失的值。这样，特定的学习算法的表现就可以通过取各个数据集上的表现的平均值来进⾏评估。

考虑公式（3.37）的第⼀项的被积函数，对于⼀个特定的数据集 D ，它的形式为

$${y(\vec x;D) − h(\vec x)}^2 \tag{3.38}$$

由于这个量与特定的数据集 D 相关，因此我们对所有的数据集取平均.我们有:

$$\{y(\vec x;D) − \mathbb E_D[y(\vec x;D)] + \mathbb E_D[y(\vec x;D)] − h(x)\}^2 \\ = \{y(\vec x;D) − \mathbb E_D[y(\vec x;D)]\}^2 + \{\mathbb E_D[y(\vec x;D)] − h(x)\}^2 + \\ 2\{y(\vec x;D) − \mathbb E_D[y(\vec x;D)]\}\{\mathbb E_D[y(\vec x;D)] − h(\vec x)\} \tag{3.39}$$

关于D求期望得到, 注意最后一项为0：

$$\mathbb E_D[\{y(\vec x; D) - h(\vec x)\}^2] \\ = \{\mathbb E_D[y(\vec x; D)] - h(\vec x)\}^2 + \mathbb E_D[\{y(\vec x; D) - \mathbb E_D[y(\vec x; D)]\}^2] \tag{3.40}$$

$y(x;D)$ 与回归函数 $h(x)$ 的差的平⽅的期望可以表⽰为两项的和。第⼀项，被称为平⽅偏置（ **bias** ），表⽰所有数据集的平均预测与预期的回归函数之间的差异。第⼆项，被称为⽅差（ **variance** ），度量了对于单独的数据集，模型所给出的解在平均值附近波动的情况，此也就度量了函数 $y(x;D)$ 对于特定的数据集的选择的敏感程度。

如果我们把这个展开式带回到公式（3.37）中，那么我们就得到了下⾯的对于期望平⽅损失的分解

$$expected = (bias)^2 + variance + noise \tag{3.41}$$

$$ (bias)^2 = \int \{\mathbb E_D[y(x;D)] − h(x)\}^2p(x)dx \\
    variance = \int \mathbb E_D[\{y(\vec x; D) - \mathbb E_D[y(\vec x; D)]\}^2] p(x)dx \\
    noise = \int \{h(x) - t\}^2p(x,t)dxdt
$$

现在，偏置和⽅差指的是积分后的量。我们的⽬标是最⼩化期望损失，它可以分解为（平⽅）偏置、⽅差和⼀个常数噪声项的和。正如我们将看到的那样，在偏置和⽅差之间有⼀个折中。对于⾮常灵活的模型来说，偏置较⼩，⽅差较⼤。对于相对固定的模型来说，偏置较⼤，⽅差较⼩。有着最优预测能⼒的模型时在偏置和⽅差之间取得最优的平衡的模型。

将多个解加权平均是贝叶斯⽅法的核⼼，虽然这种求平均针对的是参数的后验分布，⽽不是针对多个数据集。
