from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

class MNIST_NN:
    def __init__(self, input_dim=48, output_dim=10, batch_size=64):

        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dims = [input_dim, input_dim, output_dim]
        self.num_dims = [input_dim * input_dim, input_dim, input_dim*output_dim, output_dim]
        mnist = tf.keras.datasets.mnist

        (x_train_unflattened, y_train), (x_test, y_test) = mnist.load_data()
        x_train_unflattened = x_train_unflattened / 255.0
        x_train = tf.reshape(x_train_unflattened, [60000, -1, 784]) # shape 60000(N) x 1 x 784
        rand_projection = tf.random.normal(shape=[784, self.input_dim], dtype=tf.dtypes.float64)
        self.projected_x = tf.matmul(x_train, rand_projection)
        y_train_ = tf.reshape(tf.dtypes.cast(y_train, dtype=tf.dtypes.int32), shape=[-1])
        self.labels = y_train_
        self.build_model()

    def init_weights(shape, name=None):
            return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

    def init_bias(shape, name=None):
            return tf.Variable(tf.zeros(shape, dtype='float'), name=name)

    def build_model(self):

        self.indices = tf.placeholder(shape=[self.batch_size], dtype=tf.int32)
        self.data = tf.gather(indices=self.indices, params=self.projected_x, axis=0)
        self.labels = tf.one_hot(indices=tf.gather(indices=self.indices, params=self.labels), depth=10)
        self.weights_1 = tf.placeholder(shape=[self.input_dim, self.input_dim], dtype=tf.dtypes.float64)
        self.biases_1 = tf.placeholder(shape=[1, self.input_dim], dtype=tf.dtypes.float64)
        self.weights_2 = tf.placeholder(shape=[self.input_dim, self.output_dim], dtype=tf.dtypes.float64)
        self.biases_2 = tf.placeholder(shape=[1, self.output_dim], dtype=tf.dtypes.float64)
        self.fc1 = tf.nn.relu(tf.matmul(self.data, self.weights_1) + self.biases_1)
        self.fc2 = tf.nn.relu(tf.matmul(self.fc1, self.weights_2) + self.biases_2)
        self.output = tf.reshape(tf.nn.softmax(self.fc2), shape=[1, self.batch_size, 10])
        self.loss_out = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.output))

    def unpack_x(self, x):
        unpacked_x = []
        prev_dim = 0
        for dim in self.num_dims:
            unpacked_x.append(x[prev_dim:prev_dim+dim])
            prev_dim += dim
        return unpacked_x

    def evaluate(self, x):
        sess=tf.Session()
        indices = np.random.choice(np.arange(60000), size=64)

        unpacked_x = self.unpack_x(np.array(x))
        weights = []
        biases = []
        for i in range(len(self.dims) - 1):
            weights.append(unpacked_x[2*i].reshape((self.dims[i], self.dims[i+1])))
            biases.append(unpacked_x[2*i+1].reshape((1, self.dims[i+1])))
        feed_dict={self.indices: indices, self.weights_1: weights[0], self.biases_1: biases[0], self.weights_2: weights[1], self.biases_2: biases[1]}
        loss = sess.run([self.output, self.loss_out], feed_dict=feed_dict)

        return loss[1]

