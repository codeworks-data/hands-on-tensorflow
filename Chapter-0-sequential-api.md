## **1. Deep dive into TensorFlow/Keras Sequential API**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

### **1.1  Layers in TensorFlow**

Most models are made of layers. Layers are functions with a known mathematical structure that can be reused and have trainable variables. <br>
Layers and models are build built on the same foundational class : `tf.Module`  <br>

```python
class SimpleModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.a_variable = tf.Variable(5.0, name="train_me")
        self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")
    def __call__(self, x):
        return self.a_variable * x + self.non_trainable_variable

simple_module = SimpleModule(name="simple")

simple_module(tf.constant(5.0))
```
```
<tf.Tensor: shape=(), dtype=float32, numpy=30.0>
```
So, we can define a `Dense` layer as follows.

```python
class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        
        self.w = tf.Variable(tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')
        
    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)
```
```python
dense_1 = Dense(in_features=2, out_features=3)
dense_1.variables
```

A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor. <br>

