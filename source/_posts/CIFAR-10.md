---
title: CIFAR-10
date: 2018-09-15 13:02:14
tags: 数据集
categories: 深度学习
---
The CIFAR-10 and CIFAR-100 are labeled subsets of the 80 million tiny images dataset. They were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

## [The CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html)

The CIFAR-10 dataset consists of **60000 32x32** colour images in **10 classes**, with 6000 images per class. There are **50000 training** images and **10000 test** images.

The dataset is divided into **five training batches** and **one test batch**, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

## Dataset layout (python/matlab version)

The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5, as well as test_batch. Each of these files is a Python "pickled" object

```python
def unpickle(file):  # python2
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def unpickle(file):  # python3
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
```

Loaded in this way, each of the batch files contains a dictionary with the following elements:

    data --
    a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
    The first 1024 entries contain the red channel values,
    the next 1024 the green, and the final 1024 the blue.
    The image is stored in row-major order, so that the first 32 entries of
    the array are the red channel values of the first row of the image.

    labels --
    a list of 10000 numbers in the range 0-9.
    The number at index i indicates the label of the ith image in the array data.

The dataset contains another file, called **batches.meta**. It too contains a Python dictionary object. It has the following entries:

    label_names --
    a 10-element list which gives meaningful names to the numeric labels in the labels array described above.
     For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.
