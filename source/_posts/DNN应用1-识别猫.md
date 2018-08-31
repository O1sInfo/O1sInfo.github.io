---
title: DNN应用1--识别猫
date: 2018-08-03 08:52:05
tags: DNN
categories: 深度学习
---
## 实验目的

使用深层全连接神经网络识别一副图片是否为猫，并将网络层数及每层单元数设为超参数。

## 实验方案

- 使用python自行编码各运算单元，主要借助numpy库的数据结构和运算函数。
- 各个隐藏层采用Relu激活函数，输出层采用Sigmod激活函数，隐藏层使用dropout处理
- 损失函数采用交叉熵，并使用L2正则化
- 网络架构
 ![](/images/LlayerNN.png)

## 详细设计

### 数据预处理

#### 加载数据

```python
import numpy as np
import h5py

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
```

```python
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
```

#### 数据集形状

>Number of training examples: 209
Number of testing examples: 50
Each image is of size: (64, 64, 3)
train_x_orig shape: (209, 64, 64, 3)
train_y shape: (1, 209)
test_x_orig shape: (50, 64, 64, 3)
test_y shape: (1, 50)

#### 展示数据图片

```python
import matplotlib.pyplot as plt

index = 7
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
plt.show()
```

#### 图像矩阵向量化

```python
# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
```

#### 数据集最终形状

>train_x's shape: (12288, 209)
test_x's shape: (12288, 50)

### 网络设计

1. 初始化参数 / 定义超参数
2. 迭代循环:
    a. 前向传播
    b. 计算代价函数
    c. 反向传播
    d. 更新参数
3. 使用训练的参数去预测新的数据标签

网络主框架代码，其他细节函数参见“神经网络中的通用函数代码”

```python
def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    costs = []                         # keep track of cost
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        # Compute cost.
        cost = compute_cost(AL, Y)
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate=0.0075)
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters
```

## 实验结果

### 训练集结果

```python
layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
```

![](/images/res1.PNG)

```python
pred_train = predict(train_x, train_y, parameters)
```

>Accuracy: 0.9856459330143539

### 测试集结果

```python
pred_test = predict(test_x, test_y, parameters)
```

>Accuracy: 0.8

### 数据集外结果

```python
from scipy import ndimage
import scipy.misc

my_image = "my_image.jpg"
my_label_y = [0]
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((num_px * num_px * 3, 1))
my_predicted_image = predict(my_image, my_label_y, parameters)
plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)), ].decode("utf-8") + "\" picture.")
```

>Accuracy: 1.0
>y = 1.0, your L-layer model predicts a "cat" picture.

![](/images/my_image.jpg)
