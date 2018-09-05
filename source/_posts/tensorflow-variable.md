---
title: tensorflow variable
date: 2018-09-05 10:50:55
tags: tensorflow_python_API
categories: Tensorflow
---
A TensorFlow variable is the best way to represent shared, persistent state manipulated by your program.

### Create a variable

```python
my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32,
  initializer=tf.zeros_initializer)
```

**tf.get_variable**

* 构造函数
    ```python
    tf.get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=None,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None,
    constraint=None,
    synchronization=tf.VariableSynchronization.AUTO,
    aggregation=tf.VariableAggregation.NONE
    )
    ```

* 说明

    **initializer:** Initializer for the variable if one is created. Can either be an initializer object or a Tensor. If it's a Tensor, its shape must be known unless validate_shape is False.

    **regularizer:** A (Tensor -> Tensor or None) function; the result of applying it on a newly created variable will be added to the collection *tf.GraphKeys.REGULARIZATION_LOSSES* and can be used for regularization.

    **trainable:** If True also add the variable to the graph collection *GraphKeys.TRAINABLE_VARIABLES* (see tf.Variable).
    collections: List of graph collections keys to add the Variable to. Defaults to *[GraphKeys.GLOBAL_VARIABLES]* (see tf.Variable).

### Variable collections

Because disconnected parts of a TensorFlow program might want to create variables, it is sometimes useful to have a single way to access all of them.For this reason TensorFlow provides collections, which are named lists of tensors or other objects, such as tf.Variable instances

By default every tf.Variable gets placed in the following two collections:

    tf.GraphKeys.GLOBAL_VARIABLES --- variables that can be shared across multiple devices,
    tf.GraphKeys.TRAINABLE_VARIABLES --- variables for which TensorFlow will calculate gradients.

If you don't want a variable to be trainable, add it to the **tf.GraphKeys.LOCAL_VARIABLES** collection instead.

```python
my_local = tf.get_variable("my_local", shape=(),
collections=[tf.GraphKeys.LOCAL_VARIABLES])

my_non_trainable = tf.get_variable("my_non_trainable",
                                   shape=(),
                                   trainable=False)

tf.add_to_collection("my_collection_name", my_local)
#  retrieve a list of all the variables
tf.get_collection("my_collection_name")                               
```

### Initializing variables

To initialize all trainable variables in one go, before training starts, call **tf.global_variables_initializer()**. This function returns a single operation responsible for initializing all variables in the **tf.GraphKeys.GLOBAL_VARIABLES** collection.

```python
session.run(tf.global_variables_initializer())
# Now all variables are initialized.
session.run(my_variable.initializer)
```

Note that by default tf.global_variables_initializer does not specify the order in which variables are initialized. Therefore, if the initial value of a variable depends on another variable's value, it's likely that you'll get an error. Any time you use the value of a variable in a context in which not all variables are initialized (say, if you use a variable's value while initializing another variable), it is best to use **variable.initialized_value()** instead of variable:

```python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = tf.get_variable("w", initializer=v.initialized_value() + 1)
```

### Using variable

To use the value of a tf.Variable in a TensorFlow graph, simply treat it like a normal tf.Tensor:

```python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = v + 1  # w is a tf.Tensor which is computed based on the value of v.
           # Any time a variable is used in an expression it gets automatically
           # converted to a tf.Tensor representing its value.
```

### Sharing variables

TensorFlow supports two ways of sharing variables:

    Explicitly passing tf.Variable objects around.
    Implicitly wrapping tf.Variable objects within tf.variable_scope objects.

Variable scopes allow you to control variable reuse when calling functions which implicitly create and use variables. They also allow you to name your variables in a hierarchical and understandable way.

```python
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME'):
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])
    # create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])
        conv = convolve(x, weights)
        bias = tf.reshape(tf.nn.bias_add(conv, biases))
        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)
        return relu
```
