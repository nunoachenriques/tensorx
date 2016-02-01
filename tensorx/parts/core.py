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

import tensorflow as tf
import networkx as nx
from .variable_init import normalised_weight_init


class Layer(object):
    """Layer placeholder for tensor operations.

    A layer is a node in the computational graph of a neural network.
    """
    def __init__(self, n_units, activation=tf.identity):
        """Creates an instance of a Layer to be added to a NeuralNetwork.

        Args:
            n_units:    number of units (neurons) in the layer;
            activation: tensor or op for this layer --this will be used to determine what a layer does with its
                           inputs (which are determined when you add a layer to a NeuralNetwork).
        """
        self.n_units = n_units
        self.activation = activation


class GraphKeys(object):
    WEIGHTS = 'W'
    BIASES = 'B'
    LAYER = 'L'
    ACTIVATION = 'ACT'
    PRE_ACTIVATION = 'PACT'


class NeuralNetwork(object):
    """Neural Network Container.

    Neural Network Parts help to build TensorFlow computation graphs by adding layers with various activation functions
    and automatically adding biases, creating weight matrices between each layer, and initialising those weight.

    Neural Network objects store the layers and computation graph nodes in a directed networkx graph; this can be used
    to access each element of the computation graph and display the network. You can access this graph object with (
    network.graph). The data is stored in the graph using core.GraphKeys. Usualy you don't need to manipulate this graph
    directly since some getter methods are supplied but if you want to (e.g. for debugging reasons) see bellow:

    Manipulating the Graph Directly:
        Weights: stored in the edges of the graph with the key GraphKeys.WEIGHTS;

        Biases: stored in the nodes of the graph (since they are associated with each layer) with the key
                GraphKeys.Biases;

        Layer: each layer object is stored in the nodes of the graph with the key GraphKeys.Layer;

        Node Activation: computation graph end-points are stored in the nodes (they correspond to the computation on a
                         particular layer. They are stored in the graph with the key GraphKeys.ACTIVATION;

        Pre Activation: for utility reasons we might want to access the computation on a layer before applying the
                        layers' activation function, this is stored in the graph nodes with the key
                        GraphKeys.PRE_ACTIVATION.

    Using the Network (Summary):
        To use the network you will want to use its methods instead of the graph:

        size: number of layers in the network;

        weights: gets the weights between two given layer indexes (starting index is 0);

        layer: gets a layer object stored in the graph in the given index;

        biases: gets the biases for a layer with the given index;

        output: gets the TensorFlow computation graph for a layer with the given index; the default is the last node.
    """

    def __init__(self,layer=None):
        self.graph = nx.DiGraph()

        if not layer:
            self._output_node = None
        else:
            self.add_layer(layer, biased=False)

    def size(self):
        """The size of the NeuralNetwork in number of layers.
        Returns:
            number of layers in the network.
        """
        return self.graph.number_of_nodes()

    def weights(self, li, lj):
        """Access weights between layers i and j (order sensitive).

        Args:
            li: predecessor layer of j;
            lj: successor layer of i.

        Returns:
            A TensorFlow Variable with the weights between the given layers;
            None of they don't exist.
        """
        edge_data = self.graph.get_edge_data(li, lj)
        if GraphKeys.WEIGHTS not in edge_data:
            return None
        return self.graph.edge[li][lj][GraphKeys.WEIGHTS]

    def layer(self, li):
        return self.graph.node[li][GraphKeys.LAYER]

    def biases(self, li):
        """Access biases of layer i.

        Args:
            li: index of layer from which we want the biases.
        Returns:
            A TensorFlow Variable with the biases for layer i;
            None if they don't exist.
        """
        return self.graph.node[li][GraphKeys.BIASES]

    def output(self, li=None):
        """Returns a handle to the neural network output.

        Returns a reference to the TensorFlow op at the output node of the neural network.
        Args:
            li: index for the layer we wich to treat as computation graph output.
        """
        out = self._output_node
        if li:
            out = self.graph.node[li]

        return out[GraphKeys.ACTIVATION]

    def _add_weights(self, li, lj, shared_weights=None, weight_init=normalised_weight_init):
        layer_i = self.layer(li)
        layer_j = self.layer(lj)

        # 1 create weights between layers i and j
        weights_ij = shared_weights
        if not shared_weights:
            weights_ij = tf.Variable(weight_init(layer_i.n_units, layer_j.n_units))

        self.graph.add_edge(li,lj)
        self.graph.edge[li][lj][GraphKeys.WEIGHTS] = weights_ij

        return weights_ij

    def _create_computation_node(self, li, lj):
        li_out = self.graph.node[li][GraphKeys.ACTIVATION]

        weights_ij = self.weights(li,lj)
        lj_pre_act = tf.matmul(li_out,weights_ij)
        # y = xW + b
        b = self.biases(lj)
        if b:
            lj_pre_act = tf.add(lj_pre_act,b)
        # update pre-activation op
        self.graph.node[lj][GraphKeys.PRE_ACTIVATION] = lj_pre_act
        # z = fn(xW + b)
        layer_j = self.layer(lj)
        lj_act = layer_j.activation(lj_pre_act)
        self.graph.node[lj][GraphKeys.ACTIVATION] = lj_act

    def _add_biases(self, li, biased=True):
        layer = self.layer(li)
        b = None
        if biased:
            b = tf.Variable(tf.zeros([layer.n_units]))
        self.graph.node[li][GraphKeys.BIASES] = b

    def _add_layer_node(self,layer):
        n_layers = self.size()
        self.graph.add_node(n_layers)
        self.graph.node[n_layers][GraphKeys.LAYER] = layer

    def add_layer(self, layer, biased=True, shared_weights=None, weight_init=normalised_weight_init):
        """Adds a new layer to the network.

        1. append to the last layer in the network;
        2. add input variables for the layer;
            2.1 creates TensorFlow variables for the weights between the new layer and the previous one;
            2.2. create bias units for this layer.
        3. update the TensorFlow computation graph with the last layer.


        Note: in the case of shared_weights instead of passing a tensor variable we might want to pass tf.transpose(w).

        Args:
            biased: if true network adds biases for the units in this layer;
            layer: the layer to be added;
            shared_weights: reference to the weight variable to be shared;
            weight_init: weight initialisation function.
        """
        if not isinstance(layer,Layer):
            raise ValueError("layer must be an instance of Layer")
        elif shared_weights and not (
                    isinstance(shared_weights, tf.Tensor) or
                    isinstance(shared_weights, tf.Variable)):
            raise ValueError("shared weights must be a reference to a TensorFlow.Variable or None")
        else:
            # 1. add new layer
            n_layers = self.size()
            lj = n_layers
            li = n_layers - 1
            self._add_layer_node(layer)
            # update output_node ref
            self._output_node = self.graph.node[lj]

            if n_layers == 0:
                self.graph.node[0][GraphKeys.ACTIVATION] = layer.activation
            # we are appending to existing layers
            elif n_layers > 0:
                # 2 create weights between layers i and j
                self._add_weights(li, lj, shared_weights, weight_init)

                # add biases
                self._add_biases(lj, biased)

                # 3 create computation graph y = xW + b
                self._create_computation_node(li, lj)

    def connect_layers(self, li, lj, shared_weights = None, weight_init=normalised_weight_init):
        """Connect layer li to layer lj with new weights.

        1. Create new weights;
        2. Change activation of layer lj to sum the new inputs from layer lj.

        Args:
            li: layer from which the inputs will come;
            lj: layer to which the inputs will be added;
            shared_weights: reference to the weight variable to be shared;
            weight_init: weight initialisation function.
        """

        # 1 create weights between layers i and j
        weights_ij = self._add_weights(li,lj, shared_weights, weight_init)

        # y = (xW1 + b) + xW2
        li_act = self.graph.node[li][GraphKeys.ACTIVATION]

        lj_pre_act_1 = self.graph.node[lj][GraphKeys.PRE_ACTIVATION]
        lj_pre_act_2 = tf.matmul(li_act, weights_ij)
        lj_pre_act = tf.add(lj_pre_act_1, lj_pre_act_2)

        # update pre-activation op
        self.graph.node[lj][GraphKeys.PRE_ACTIVATION] = lj_pre_act
        # update activation op
        layer_j = self.layer(lj)
        lj_act = layer_j.activation(lj_pre_act)
        self.graph.node[lj][GraphKeys.ACTIVATION] = lj_act
