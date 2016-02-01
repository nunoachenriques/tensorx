TensorX
=======

A minimalist utility library to build neural network models in TensorFlow with minimum verbose (and without unnecessary 
levels of abstraction). _TensorX_ provides a minimum set of utility _parts_ to build computation graphs.

## Getting Started

```python

import tensorflow as tf
from tensorx.parts.core import Layer, NeuralNetwork

n_inputs = 784
x = tf.placeholder(tf.float32, [None, n_inputs], name="x")

# create neural network with tensorx
input_layer = Layer(n_units=n_inputs,activation=x)
network = NeuralNetwork(input_layer)
output_layer = Layer(10,tf.nn.softmax)
network.add_layer(output_layer,biased=True)

# train
target_output = tf.placeholder("float", shape=[None, 10])
# loss function
network_output = network.output()
cross_entropy = -tf.reduce_sum(target_output*tf.log(tf.clip_by_value(network_output,1e-50,1.0)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
...
# for a full example see "examples" folder
```

## Pip Installation
```
sudo pip3 install --upgrade git+https://github.com/davidelnunes/tensorx.git
```

## Licence
Copyright 2016 Davide Nunes

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

