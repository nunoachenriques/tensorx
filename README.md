TensorX
=======

A minimalist utility library to build neural network models in TensorFlow with minimum verbose (and without unnecessary 
levels of abstraction). _TensorX_ provides a minimum set of utility _parts_ to build computation graphs.

## Getting Started

```python
import tensorflow as tf
from tensorx.parts.core import NeuralNetwork,Layer
from tensorflow.examples.tutorials.mnist import input_data

# load MNIST data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
ss = tf.InteractiveSession()

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

# test
correct_prediction = tf.equal(tf.argmax(network_output,1),
                              tf.argmax(target_output,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# run train and test
tf.initialize_all_variables().run()

for i in range(1000):
    if i % 10 == 0:  # Record summary data, and the accuracy
        feed = {x: mnist.test.images, target_output: mnist.test.labels}
        acc = ss.run(accuracy, feed_dict=feed)

        print("Accuracy at step %s: %s" % (i, acc))
    else:
        batch_xs, batch_ys = mnist.train.next_batch(100)
        feed = {x: batch_xs, target_output: batch_ys}
        ss.run(train_step, feed_dict=feed)

print("Error:" + str((1 - accuracy.eval({x: mnist.test.images, target_output: mnist.test.labels}))*100)+"%")
ss.close()
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

