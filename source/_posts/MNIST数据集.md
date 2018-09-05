---
title: MNIST数据集
date: 2018-09-04 21:21:44
tags: 数据集
categories: 深度学习
---
### [MNIST](http://yann.lecun.com/exdb/mnist/)

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

### Tensorflow `v1.10` API

```python
mnist = tf.keras.datasets.mnist.load_data(path='mnist.npz')
# type(mnist) is Tuple of Numpy arrays:
# `(x_train, y_train), (x_test, y_test)`.
(x_train, y_train), (x_test, y_test) = mnist
x_train.shape == (60000, 28, 28)
x_test.shape == (10000, 28, 28)
y_train.shape == (60000,)
y_test.shape == (10000,)
# x_train[0, :, :] (0, 255) uint8
# y_train[0] (0, 10) uint8

# show the pic
from PIL import Image
im = Image.fromarray(x_train[0,:,:])
im.show()
```

Other Low-level API
```python
from tensorflow.examples.tutorials.mnist import input_data
# Or use following import method
# from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = input_data.read_data_sets('./mnist/', one_hot=True)
mnist.train.images.shape == (55000, 784) # normalized to (0, 1) float32
mnist.test.images.shape == (10000, 784)
mnist.train.labels.shape == (55000, 10) # one_hot vector
# if one_hot is False(default) the axis 1 will be removed and label is (0, 10)
mnist.test.labels.shape == (10000, 10)
mnist.validation.images.shape == (5000, 784)
mnist.validation.labels.shape == (5000, 10)

x, y = mnist.train.next_batch(batch_size)
# x.shape == (batch_size, 784)
# y.shape == (batch_size, 10)
```

### scikit-learn

```python
#sklearn.datasets.fetch_mldata(dataname, target_name=’label’, data_name=’data’, transpose_data=True, data_home=None)
# Fetch an mldata.org data set

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home='./')
# you should put the mnist_original.mat in ./mldata

mnist.data.shape == (70000, 784) # scalar (0, 255) uint8
mnist.target.shape == (70000,) # scalar (0, 10) with increasing order
```

[mldata.org](ml.data.org) a machine learning data set repository
