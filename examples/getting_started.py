import numpy as np
import tensorflow as tf
from tensorx.parts.core import NeuralNetwork,Layer


n_inputs = 2

x = tf.placeholder(tf.float32, [None, n_inputs], name = "x")

input_layer = Layer(n_units = n_inputs, activation = x)
network = NeuralNetwork(input_layer)
idd_layer = Layer(n_inputs)
network.add_layer(idd_layer,biased=False)

out = network.output()


with tf.Session() as ss:
    init_op = tf.global_variables_initializer()
    ss.run(init_op)

    matrix1 = np.random.uniform(-1,1,(2,2))
    print(matrix1)

    feed = {x: matrix1}

    result = ss.run(out,feed_dict=feed)
    print(result)