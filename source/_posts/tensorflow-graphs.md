---
title: tensorflow graphs
date: 2018-09-05 17:03:01
tags: tensorflow_python_API
categories: Tensorflow
---
TensorFlow uses a dataflow graph to represent your computation in terms of the dependencies between individual operations. This leads to a low-level programming model in which you first define the dataflow graph, then create a TensorFlow session to run parts of the graph across a set of local and remote devices.

### Dataflow
![](https://tensorflow.google.cn/images/tensors_flowing.gif)

Dataflow is a common programming model for parallel computing. In a dataflow graph, the nodes represent units of computation, and the edges represent the data consumed or produced by a computation

Dataflow has several advantages that TensorFlow leverages when executing your programs:

 * **Parallelism**. By using explicit edges to represent dependencies between operations, it is easy for the system to identify operations that can execute in parallel.

 * **Distributed execution**. By using explicit edges to represent the values that flow between operations, it is possible for TensorFlow to partition your program across multiple devices (CPUs, GPUs, and TPUs) attached to different machines. TensorFlow inserts the necessary communication and coordination between devices.

 * **Compilation**. TensorFlow's XLA compiler can use the information in your dataflow graph to generate faster code, for example, by fusing together adjacent operations.

 * **Portability**. The dataflow graph is a language-independent representation of the code in your model. You can build a dataflow graph in Python, store it in a SavedModel, and restore it in a C++ program for low-latency inference.

### What is a tf.Graph?

 A tf.Graph contains two relevant kinds of information:

  * **Graph structure**. The nodes and edges of the graph, indicating how individual operations are composed together, but not prescribing how they should be used. The graph structure is like assembly code: inspecting it can convey some useful information, but it does not contain all of the useful context that source code conveys.

  * **Graph collections**. TensorFlow provides a general mechanism for storing collections of metadata in a tf.Graph. The **tf.add_to_collection** function enables you to associate a list of objects with a key (where **tf.GraphKeys** defines some of the standard keys), and **tf.get_collection** enables you to look up all objects associated with a key. Many parts of the TensorFlow library use this facility: for example, when you create a tf.Variable, it is added by default to collections representing "global variables" and "trainable variables". When you later come to create a tf.train.Saver or tf.train.Optimizer, the variables in these collections are used as the default arguments.

### Building a tf.Graph

  Most TensorFlow programs start with a dataflow graph construction phase. In this phase, you invoke TensorFlow API functions that construct new **tf.Operation (node)** and **tf.Tensor (edge)** objects and add them to a **tf.Graph instance**. TensorFlow provides a default graph that is an implicit argument to all API functions in the same context.

### Naming operations
 A tf.Graph object defines a namespace for the tf.Operation objects it contains. TensorFlow automatically chooses a unique name for each operation in your graph, but giving operations descriptive names can make your program easier to read and debug. The TensorFlow API provides two ways to override the name of an operation:

 * Each API function that creates a new tf.Operation or returns a new tf.Tensor accepts an optional name argument. For example, tf.constant(42.0, name="answer") creates a new tf.Operation named "answer" and returns a tf.Tensor named "answer:0". If the default graph already contains an operation named "answer", then TensorFlow would append "\_1", "\_2", and so on to the name, in order to make it unique.

 * The **tf.name_scope** function makes it possible to add a name scope prefix to all operations created in a particular context. The current name scope prefix is a "/"-delimited list of the names of all active tf.name_scope context managers. If a name scope has already been used in the current context, TensorFlow appends "\_1", "\_2", and so on. For example:

```python
 c_0 = tf.constant(0, name="c")  # => operation named "c"
 # Already-used names will be "uniquified".
 c_1 = tf.constant(2, name="c")  # => operation named "c_1"
 # Name scopes add a prefix to all operations created in the same context.
 with tf.name_scope("outer"):
     c_2 = tf.constant(2, name="c")  # => operation named "outer/c"
     # Name scopes nest like paths in a hierarchical file system.
     with tf.name_scope("inner"):
         c_3 = tf.constant(3, name="c")  # => operation named "outer/inner/c"
     # Already-used name scopes will be "uniquified".
     with tf.name_scope("inner"):
         c_5 = tf.constant(5, name="c")  # => operation named "outer/inner_1/c"
```

### Placing operations on different devices

If you want your TensorFlow program to use multiple different devices, the tf.device function provides a convenient way to request that all operations created in a particular context are placed on the same device (or type of device).

A device specification has the following form:

    /job:<JOB_NAME>/task:<TASK_INDEX>/device:<DEVICE_TYPE>:<DEVICE_INDEX>
    where:
        <JOB_NAME> is an alpha-numeric string that does not start with a number.
        <DEVICE_TYPE> is a registered device type (such as GPU or CPU).
        <TASK_INDEX> is a non-negative integer representing the index of the task in the job named <JOB_NAME>. See tf.train.ClusterSpec for an explanation of jobs and tasks.
        <DEVICE_INDEX> is a non-negative integer representing the index of the device, for example, to distinguish between different GPU devices used in the same process.

```python
# Operations created outside either context will run on the "best possible"
# device. For example, if you have a GPU and a CPU available, and the operation
# has a GPU implementation, TensorFlow will choose the GPU.
weights = tf.random_normal(...)

with tf.device("/device:CPU:0"):
  # Operations created in this context will be pinned to the CPU.
  img = tf.decode_jpeg(tf.read_file("img.jpg"))

with tf.device("/device:GPU:0"):
  # Operations created in this context will be pinned to the GPU.
  result = tf.matmul(weights, img)
```

### Visualizing your graph
TensorFlow includes tools that can help you to understand the code in a graph. The graph visualizer is a component of TensorBoard that renders the structure of your graph visually in a browser. The easiest way to create a visualization is to pass a tf.Graph when creating the **tf.summary.FileWriter**:

```python
# Build your graph.
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
# ...
loss = ...
train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
  writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)
  # Perform your computation...
  for i in range(1000):
    sess.run(train_op)
    # ...
  writer.close()
```

### Programming with multiple graphs

You can install a different tf.Graph as the default graph, using the tf.Graph.as_default context manager:
```python
g_1 = tf.Graph()
with g_1.as_default():
  # Operations created in this scope will be added to `g_1`.
  c = tf.constant("Node in g_1")
  # Sessions created in this scope will run operations from `g_1`.
  sess_1 = tf.Session()

g_2 = tf.Graph()
with g_2.as_default():
  # Operations created in this scope will be added to `g_2`.
  d = tf.constant("Node in g_2")

# `sess_2` will run operations from `g_2`.
sess_2 = tf.Session(graph=g_2)

assert c.graph is g_1
assert sess_1.graph is g_1

assert d.graph is g_2
assert sess_2.graph is g_2
```

To inspect the current default graph, call tf.get_default_graph, which returns a tf.Graph object:
```python
# Print all of the operations in the default graph.
g = tf.get_default_graph()
print(g.get_operations())
```
