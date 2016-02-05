import tensorflow as tf
from tensorx.parts.core import NeuralNetwork,Layer
from tensorflow.examples.tutorials.mnist import input_data

# load MNIST data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

ss = tf.InteractiveSession()

n_inputs = 784
x = tf.placeholder(tf.float32, [None, n_inputs], name = "x")

# create neural network with tensorx
input_layer = Layer(n_units = n_inputs, activation = x)
network = NeuralNetwork(input_layer)
output_layer = Layer(10, tf.nn.softmax)
network.add_layer(output_layer, biased=True)

# train
target_output = tf.placeholder("float", shape=[None, 10])
# loss function
network_output = network.output()
cross_entropy = -tf.reduce_sum(target_output*tf.log(tf.clip_by_value(network_output,1e-50,1.0)))
train_step_rate = 0.003 # default: 0.003 -> accuracy ~ 0.9162 (step 999)
train_step = tf.train.GradientDescentOptimizer(train_step_rate).minimize(cross_entropy)

# test
correct_prediction = tf.equal(tf.argmax(network_output,1), tf.argmax(target_output,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# run train and test
tf.initialize_all_variables().run()

n_steps = 1000
for i in range(n_steps):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    feed = {x: batch_xs, target_output: batch_ys}
    ss.run(train_step, feed_dict=feed)
    if i % 10 == 0:  # Record summary data, and the accuracy
        feed = {x: mnist.test.images, target_output: mnist.test.labels}
        acc = ss.run(accuracy, feed_dict=feed)
        if i == 0: # first step
            print("Accuracy at step {0:4d}: {1:.04f} | Error: {2:.2f} %".format(i, acc, (1 - acc) * 100))
        else:
            print("Accuracy at step {0:4d}: {1:.04f}".format(i, acc))

# last step
feed = {x: mnist.test.images, target_output: mnist.test.labels}
acc = ss.run(accuracy, feed_dict=feed)
print("Accuracy at step {0:4d}: {1:.04f} | Error: {2:.2f} %".format(i, acc, (1 - acc) * 100))

ss.close()
