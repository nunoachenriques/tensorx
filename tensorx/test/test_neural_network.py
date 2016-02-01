# Copyright 2016 Davide Nunes

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================================================================================================

"""Unit Tests for NeuralNetwork."""
from unittest import TestCase,main
from tensorx.parts.core import Layer, NeuralNetwork,GraphKeys
from tensorx.parts.variable_init import normalised_weight_init
import tensorflow as tf
import numpy as np


class TestNeuralNetwork(TestCase):
    def test_size(self):
        n_inputs = 10

        nn = NeuralNetwork()
        self.assertEqual(nn.size(), 0)

        x = tf.ones([1,n_inputs])
        input_layer = Layer(n_units=n_inputs,activation=x)
        nn.add_layer(input_layer)
        self.assertEqual(nn.size(), 1)

    def test_weights(self):
        n_inputs = 10
        n_hidden = 20

        x = tf.ones([1,n_inputs])
        input_layer = Layer(n_units=n_inputs,activation=x)
        hidden_layer = Layer(n_units=n_hidden,activation=tf.sigmoid)

        nn = NeuralNetwork(input_layer)
        nn.add_layer(hidden_layer)

        weights_ih = nn.weights(0,1)
        self.assertIsNot(weights_ih,None)

        init = tf.initialize_all_variables()
        with tf.Session() as ss:
            ss.run(init)
            (c, r) = ss.run(tf.shape(weights_ih))
            self.assertEqual(c,n_inputs)
            self.assertEqual(r,n_hidden)

    def test_layer(self):
        n_inputs = 10
        n_hidden = 20

        x = tf.ones([1,n_inputs])
        input_layer = Layer(n_units=n_inputs,activation=x)

        nn = NeuralNetwork(input_layer)
        self.assertEqual(nn.size(), 1)

        # add a layer with biases
        hidden_layer = Layer(n_hidden,activation=tf.identity)
        nn.add_layer(hidden_layer, biased=True)

        layer_0 = nn.layer(0)
        layer_1 = nn.layer(1)

        self.assertEqual(layer_0,input_layer)
        self.assertEqual(layer_1,hidden_layer)

    def test_biases(self):
        n_inputs = 10
        n_hidden = 20

        x = tf.ones([1,n_inputs])
        input_layer = Layer(n_units=n_inputs,activation=x)
        nn = NeuralNetwork(input_layer)
        # add a layer with biases
        id_layer = Layer(n_hidden,activation=tf.identity)
        nn.add_layer(id_layer, biased=True)
        # get biases
        b = nn.biases(1)
        self.assertIsInstance(b,tf.Variable)

        init = tf.initialize_all_variables()
        with tf.Session() as ss:
            ss.run(init)
            (bs) = ss.run(tf.shape(b))
            self.assertEqual(bs,n_hidden)

    def test_append(self):
        n_inputs = 10
        n_hidden = 20

        x = tf.ones([1, n_inputs])
        input_layer = Layer(n_units=n_inputs,activation=x)

        nn = NeuralNetwork(input_layer)
        self.assertEqual(nn.size(), 1)

        # add an identity layer
        id_layer = Layer(n_hidden,activation=tf.identity)
        nn.add_layer(id_layer,biased=True)
        self.assertEqual(nn.size(), 2)

        # output node should have the last layer added
        last_layer = nn._output_node[GraphKeys.LAYER]
        self.assertIs(last_layer,id_layer)

    def test_output(self):
        n_inputs = 10
        n_hidden = 20

        x = tf.placeholder(dtype=tf.float32, shape=[None,n_inputs])

        input_layer = Layer(n_units=n_inputs,activation=x)
        nn = NeuralNetwork(input_layer)

        # add an identity layer
        id_layer = Layer(n_hidden, activation=tf.identity)
        nn.add_layer(id_layer,biased=True)

        wij = nn.weights(0,1)
        expected_out = id_layer.activation(tf.add(tf.matmul(x,wij), nn.biases(1)))

        init = tf.initialize_all_variables()
        with tf.Session() as ss:
            ss.run(init)

            feed = {x: np.ones((1,n_inputs),dtype=np.float32)}

            r1 = ss.run(nn.output(), feed_dict=feed)
            r2 = ss.run(expected_out,feed_dict=feed)

            self.assertTrue(np.array_equal(r1,r2))

    def test_connect(self):
        n_inputs = 10
        n_hidden = 20

        x = tf.ones([1, n_inputs])
        input_layer = Layer(n_units=n_inputs,activation=x)
        id_layer1 = Layer(n_hidden,activation=tf.identity)
        id_layer2 = Layer(n_hidden,activation=tf.identity)

        nn = NeuralNetwork()
        # layer 0
        nn.add_layer(input_layer)
        # layer 1
        nn.add_layer(id_layer1)
        # layer 2
        nn.add_layer(id_layer2)

        self.assertEqual(nn.size(),3)

        # connect input to output
        self.assertEqual(nn.graph.number_of_edges(),2)
        nn.connect_layers(0, 2)
        self.assertEqual(nn.graph.number_of_edges(),3)

        self.assertIsNot(nn.weights(0,1),nn.weights(0,2))
        self.assertIsNot(nn.weights(0,1),nn.weights(1,2))
        self.assertIsNot(nn.weights(1,2),nn.weights(0,2))

    def test_equivalent_networks(self):
        """
        Create two equivalent neural networks

            nn1 = x -> 1 -> div(x/in) -> w -> sigm -> o
            nn2 = x -> w -> sigm -> 0

            (with shared weights)
        """
        x = tf.ones([1, 4])
        input_layer = Layer(n_units=4,activation=x)
        weights = tf.ones([4,4])
        shared_w = tf.Variable(normalised_weight_init(4,4))

        nn1 = NeuralNetwork(input_layer)
        l11 = Layer(4, lambda x_in: tf.div(x_in, 4))
        l21 = Layer(4, tf.nn.sigmoid)
        nn1.add_layer(l11,biased=False,shared_weights=weights)
        nn1.add_layer(l21,biased=False,shared_weights=shared_w)

        self.assertEqual(nn1.graph.number_of_edges(), 2)
        self.assertEqual(nn1.size(),3)

        nn2 = NeuralNetwork(Layer(4,x))
        l12 = Layer(4, tf.nn.sigmoid)
        nn2.add_layer(l12,biased=False,shared_weights=shared_w)

        self.assertEqual(nn2.size(),2)
        self.assertEqual(nn2.graph.number_of_edges(),1)

        init_vars = tf.initialize_all_variables()

        with tf.Session() as ss:
            ss.run(init_vars)

            w1 = ss.run(nn1.weights(1,2))
            w2 = ss.run(nn2.weights(0,1))

            # shared weights should be the same in both networks
            self.assertTrue(np.array_equal(w1,w2))

            out_nn1 = ss.run(nn1.output())
            out_nn2 = ss.run(nn2.output())

            # outputs should be the same since the networks are equivalent
            self.assertTrue(np.array_equal(out_nn1,out_nn2))


if __name__ == '__main__':
    main()