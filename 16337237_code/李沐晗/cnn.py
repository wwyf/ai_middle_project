import os
import re
import time
import numpy as np
import tensorflow as tf
from typing import Any, Tuple, List

import utils


def batch_norm(x,
              epsilon=1e-5, momentum=0.9, name="batch_norm", training=None):
    """
    helper class to create a batch normalization layer
    """
    return tf.layers.batch_normalization(x,
                                         momentum=momentum,
                                         epsilon=epsilon,
                                         name=name,
                                         training=training)


def conv2d(x,
           kernel_h, kernel_w,
           stride_h, stride_w,
           mean=0.0, std_dev=0.01,
           name="conv2d"):
    """
        helper function to create a convolution layer (note: single channel)
        @param kernel_h: convolution kernel height
        @param kernel_w: convolution kernel width
        @param stride_h: stride height
        @param stride_w: stride width
        @param mean: mean value used to initialize kernel
        @param std_dev: standard deviation used to initialize kernel
        @return: output
    """

    # calculate output dim from input dim
    output_w = tf.shape(x)[2] - kernel_w + stride_w
    output_h = tf.shape(x)[1] - kernel_h + stride_h
    with tf.variable_scope(name):
        kernel = tf.get_variable('kernel', [kernel_h, kernel_w, 1, 1],
                                 initializer=tf.truncated_normal_initializer(mean=mean, stddev=std_dev))
        conv = tf.nn.conv2d(x, kernel, strides=[1, stride_h, stride_w, 1], padding="VALID")

        # add zero bias
        # biases = tf.fill(dims=[tf.shape(x)[0], output_h, output_w, 1], value=0.0)
        # conv = tf.add(conv, biases)
        return conv


def max_pool_1_1d(x, size, name="max_pool_1_1d"):
    return tf.nn.max_pool(x,
                          ksize=[1, size, 1, 1],
                          strides=[1, 1, 1, 1], padding="VALID", name=name)


def sigmoid(x, name="sigmoid"):
    return tf.nn.sigmoid(x, name=name)


def tanh(x, name="tanh"):
    return tf.nn.tanh(x, name=name)


def relu(x, name="relu"):
    return tf.nn.relu(x, name)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x, name=name)

def selu(x, name="selu"):
    return tf.nn.selu(x, name=name)

def neuron_layer(x, dim_in, dim_out,
                 activation="relu",
                 weight=None,
                 bias=None,
                 mean=0.0,
                 std_dev=0.3,
                 seed=1,
                 dropout=False,
                 dropout_rate=0.5,
                 name="neuron"):
    """
    helper function to construct hidden layers of MLP
    """
    with tf.variable_scope(name):
        # initialize weights with truncated random normal distribution
        if weight is None:
            w = tf.Variable(tf.truncated_normal([dim_in, dim_out],
                                                mean=mean,
                                                stddev=std_dev,
                                                seed=seed))
        else:
            w = tf.Variable(weight)

        # initialize bias to zero
        if bias is None:
            b = tf.Variable(tf.zeros([dim_out]))
        else:
            b = tf.Variable(bias)
    if dropout:
        w = tf.nn.dropout(w, keep_prob=tf.Variable(1.0 - dropout_rate))
    if activation == "relu":
        return relu(tf.matmul(x, w) + b, name=name + "_relu")
    elif activation == "lrelu":
        return lrelu(tf.matmul(x, w) + b, name=name + "_lrelu")
    elif activation == "tanh":
        return tanh(tf.matmul(x, w) + b, name=name + "_tanh")
    elif activation == "sigmoid":
        return sigmoid(tf.matmul(x, w) + b, name=name + "_sigmoid")
    elif activation == "none":
        return tf.matmul(x, w) + b


def MLP(x,
        dim_in: int,
        dim_out: int,
        layer_sizes: Any,
        dropout_rates: Any,
        activations: Any,
        mean=0.0,
        std_dev=0.1,
        seed=1,
        reuse=False):
    """
    construct a multilayer perceptron
    @param x: input tensor
    @param layer_sizes: array-like structures, containing size of each layer
    @param dropout_rates: array-like structures, containing dropout rates of each layer
    @param activations: array-like structures, containing activation function name of each layer
    @param reuse: reuse tensorflow variables or not
    """
    with tf.variable_scope("MLP") as scope:
        if reuse:
            scope.reuse_variables()

        num = len(layer_sizes)
        # create hidden layers
        for i in range(0, num):
            if i == 0:
                last_layer = neuron_layer(x,
                                          dim_in=dim_in,
                                          dim_out=layer_sizes[0],
                                          activation=activations[0],
                                          mean=mean,
                                          std_dev=std_dev,
                                          seed=seed,
                                          dropout=(dropout_rates[0] == 0.0),
                                          dropout_rate=dropout_rates[0]
                                          )
                tf.summary.histogram('neuron_%d' % i, last_layer)
            else:
                last_layer = neuron_layer(last_layer,
                                          dim_in=layer_sizes[i - 1],
                                          dim_out=layer_sizes[i],
                                          activation=activations[i],
                                          mean=mean,
                                          std_dev=std_dev,
                                          seed=seed,
                                          dropout=(dropout_rates[i] == 0.0),
                                          dropout_rate=dropout_rates[i]
                                          )
                tf.summary.histogram('neuron_%d' % i, last_layer)
        # create output layer
        last_layer = neuron_layer(last_layer, dim_in=layer_sizes[-1], dim_out=dim_out, activation="none")
        return tf.nn.softmax(last_layer, axis=0, name="soft_max")

class CNN:
    def __init__(self,
                 sess,
                 vector_dim: int,
                 output_dim: int,
                 max_rows: int,
                 conv_dim=[3, 4, 5],
                 feature_maps=[2, 2, 2],
                 hidden_units=[128, 128],
                 hidden_activation=["relu", "relu"],
                 dropout_rates=[0.5, 0.5],
                 enable_batch=True,
                 batch_size=64,
                 conv_activation="relu",
                 checkpoint_dir=None,
                 summary_dir=None
                 ):
        self.sess = sess
        self.vector_dim = vector_dim
        self.output_dim = output_dim
        self.conv_dim = conv_dim
        self.feature_maps = feature_maps
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        self.dropout_rates = dropout_rates
        self.batch_size = batch_size
        self.enable_batch = enable_batch
        self.conv_activation = conv_activation
        self.checkpoint_dir = checkpoint_dir
        self.summary_dir = summary_dir
        self.max_rows = max_rows

        self.is_training = None
        self.input = None
        self.output = None
        self.labels = None
        self.padded_input = None
        self.vars = None
        self.saver = None
        self.optimizer = None

    def build_model(self):
        self.is_training = tf.placeholder(dtype=bool, name="is_training")
        self.labels = tf.placeholder(dtype=np.int32, shape=(None), name="labels")
        if self.enable_batch:
            self.input = tf.placeholder(dtype=tf.float32,
                                        shape=(self.batch_size, None, self.vector_dim),
                                        name="input")
            self.output = tf.placeholder(dtype=tf.float32,
                                         shape=[self.batch_size],
                                         name="output")
        else:
            self.input = tf.placeholder(dtype=tf.float32,
                                        shape=(1, None, self.vector_dim),
                                        name="input")
            self.output = tf.placeholder(dtype=tf.float32,
                                         shape=[1],
                                         name="output")

        # pad all input sentence vector number to self.sentence_max_num
        self.padded_input = tf.pad(self.input,
                                   paddings=[(0, 0), (0, self.max_rows - tf.shape(self.input)[1]), (0, 0)],
                                   constant_values=0,
                                   name="padded_input")

        # create the convolution layer
        self.layer_conv = []
        conv_dim_map = {}
        layer_count = 0
        count = 0
        for cv, ft in zip(self.conv_dim, self.feature_maps):
            for i in range(0, ft):
                self.layer_conv.append(conv2d(tf.expand_dims(self.padded_input, -1),
                                              kernel_h=cv,
                                              kernel_w=self.vector_dim,
                                              stride_h=1,
                                              stride_w=1,
                                              name="conv_%d_%d" % (layer_count, i)))
                tf.summary.histogram('conv_%d_%d' % (cv, i), self.layer_conv[-1])
                conv_dim_map[count] = cv
                count += 1
            layer_count += 1

        self.layer_max_pool = []
        if self.enable_batch:
            # create the max-pooling layer + relu layer + batch norm layer

            for i in range(0, len(self.layer_conv)):
                self.layer_max_pool.append(
                    batch_norm(
                        tanh(
                            max_pool_1_1d(self.layer_conv[i], self.max_rows - conv_dim_map[i] + 1)
                        ),
                    name="batch_norm_%d" % i,
                    training=self.is_training))

        else:
            # create the max-pooling layer + relu layer
            for i in range(0, len(self.layer_conv)):
                self.layer_max_pool.append(
                        lrelu(
                            max_pool_1_1d(self.layer_conv[i], self.max_rows - conv_dim_map[i] + 1)
                        ))

        max_pool_out = tf.concat(self.layer_max_pool, 2)
        tf.summary.histogram('max_pool', max_pool_out)

        # create MLP layer
        self.layer_mlp = MLP(tf.reshape(max_pool_out, [max_pool_out.get_shape()[0], len(self.layer_conv)]),
                             dim_in=len(self.layer_conv),
                             dim_out=self.output_dim,
                             layer_sizes=self.hidden_units,
                             dropout_rates=self.dropout_rates,
                             activations=self.hidden_activation)

        # define loss function
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.layer_mlp,
                                                           labels=self.labels)
        )

        # define predict result function
        tf.summary.histogram('input', self.input)
        tf.summary.histogram('labels', self.labels)
        tf.summary.histogram('softmax', self.layer_mlp )
        self.output = tf.argmax(self.layer_mlp, axis=1)
        # initialize other utilities
        tf.summary.scalar("loss", self.loss)
        self.vars = tf.trainable_variables()
        self.writer = tf.summary.FileWriter(logdir=self.summary_dir)
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()

    def train(self,
              data: Any,
              labels: Any,
              learning_rate=0.0001,
              beta1=0.5,
              epochs=25,
              save_every_round=500):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                beta1=beta1) \
            .minimize(self.loss, var_list=self.vars)

        tf.global_variables_initializer().run()

        # start training
        round_counter = 0
        is_loaded, checkpoint_counter = self.load(self.checkpoint_dir)
        if is_loaded:
            round_counter = checkpoint_counter
        if self.enable_batch:
            current_epoch = round_counter // (len(data) // self.batch_size)
        else:
            current_epoch = round_counter // len(data)

        current_time = time.time()
        for epoch in range(current_epoch, epochs):
            if self.enable_batch:
                round_size = len(data) // self.batch_size
                for idx in range(0, round_size):
                    batch_data = data[idx * self.batch_size : (idx + 1) * self.batch_size]
                    batch_labels = labels[idx * self.batch_size : (idx + 1) * self.batch_size]

                    # pad batches with different size samples
                    max_h = np.max([ sample.shape[0] for sample in batch_data])
                    padded = np.zeros([self.batch_size, max_h, self.vector_dim])
                    for i in range(0, self.batch_size):
                        padded[i][0:batch_data[i].shape[0]] = batch_data[i]

                    _, summary_result, current_loss = self.sess.run([self.optimizer, self.summary, self.loss],
                                                                    feed_dict={
                                                                        self.is_training: True,
                                                                        self.input: padded,
                                                                        self.labels: batch_labels
                                                                    })
                    self.writer.add_summary(summary_result, round_counter)
                    if round_counter % save_every_round == 1:
                        self.save(self.checkpoint_dir, round_counter)
                    round_counter += 1

                    new_time = time.time()
                    if round_counter >= epochs * round_size:
                        return
                    if new_time - current_time >= 0.1:
                        utils.logger.info("Epoch: [%2d/%2d] [%4d/%4d], time: %3f loss: %.8f" \
                                          % (epoch + 1, epochs, round_counter % round_size, round_size,
                                             new_time - current_time, current_loss))
                        current_time = new_time

            else:
                round_size = len(data)
                for idx in range(0, round_size):
                    _, summary_result, current_loss = self.sess.run([self.optimizer, self.summary, self.loss],
                                                                    feed_dict={
                                                                        self.is_training: True,
                                                                        self.input: np.expand_dims(data[idx], axis=0),
                                                                        self.labels: np.array([labels[idx]])
                                                                    })
                    self.writer.add_summary(summary_result, round_counter)
                    if round_counter % save_every_round == 1:
                        self.save(self.checkpoint_dir, round_counter)
                    round_counter += 1

                    new_time = time.time()
                    if round_counter >= epochs * round_size:
                        return
                    if new_time - current_time >= 0.1:
                        utils.logger.info("Epoch: [%2d/%2d] [%4d/%4d], time: %3f loss: %.8f" \
                                          % (epoch + 1, epochs, round_counter % round_size, round_size,
                                             new_time - current_time, current_loss))
                        current_time = new_time


    def predict(self,
                data: Any) -> np.ndarray:
        result = np.full(len(data), -1, np.int32)
        if self.enable_batch:
            for idx in range(0, len(data) // self.batch_size):
                batch_data = data[idx * self.batch_size: (idx + 1) * self.batch_size]
                # pad batches with different size samples
                max_h = np.max([sample.shape[0] for sample in batch_data])
                padded = np.zeros([self.batch_size, max_h, self.vector_dim])
                for i in range(0, self.batch_size):
                    padded[i][0:batch_data[i].shape[0]] = batch_data[i]

                result[idx*self.batch_size: (idx+1)*self.batch_size] = self.sess.run([self.output],
                                            feed_dict={
                                                self.is_training: False,
                                                self.input: padded,
                                            })[0]
        else:
            for idx in range(0, len(data)):
                result[idx] = self.sess.run([self.output],
                                             feed_dict={
                                                 self.is_training: False,
                                                 self.input: np.expand_dims(data[idx], axis=0),
                                             })[0]
        return result

    def save(self,
             checkpoint_dir: str,
             step: int) -> None:
        model_name = "CNN.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load_file(self,
                  meta_file: str) -> Tuple[bool, int]:
        utils.logger.info("Reading checkpoints...")
        try:
            self.saver.restore(self.sess, meta_file)
        except:
            utils.logger.info("Failed to read {}".format(meta_file))
            return False, -1
        utils.logger.info("Success to read {}".format(meta_file))
        counter = int(re.findall("(\d+)(?!.*\d)", os.path.basename(meta_file))[0])
        return True, counter

    def load(self,
             checkpoint_dir: str) -> Tuple[bool, int]:
        utils.logger.info("Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            utils.logger.info("Success to read {}".format(ckpt_name))
            return True, counter
        else:
            utils.logger.warn("Failed to find a checkpoint")
            return False, 0


class CNN2:
    def __init__(self,
                 sess,
                 vector_dim: int,
                 output_dim: int,
                 max_rows: int,
                 conv_dim=[[3], [3]],
                 feature_maps=[100, 20],
                 hidden_units=[256],
                 hidden_activation=["tanh"],
                 dropout_rates=[0],
                 enable_batch=True,
                 batch_size=64,
                 conv_activation="relu",
                 checkpoint_dir=None,
                 summary_dir=None
                 ):
        self.sess = sess
        self.vector_dim = vector_dim
        self.output_dim = output_dim
        self.conv_dim = conv_dim
        self.feature_maps = feature_maps
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        self.dropout_rates = dropout_rates
        self.batch_size = batch_size
        self.enable_batch = enable_batch
        self.conv_activation = conv_activation
        self.checkpoint_dir = checkpoint_dir
        self.summary_dir = summary_dir
        self.max_rows = max_rows

        self.is_training = None
        self.input = None
        self.output = None
        self.labels = None
        self.padded_input = None
        self.vars = None
        self.saver = None
        self.optimizer = None

    def build_model(self):
        self.is_training = tf.placeholder(dtype=bool, name="is_training")
        self.labels = tf.placeholder(dtype=np.int32, shape=(None), name="labels")
        if self.enable_batch:
            self.input = tf.placeholder(dtype=tf.float32,
                                        shape=(self.batch_size, None, self.vector_dim),
                                        name="input")
            self.output = tf.placeholder(dtype=tf.float32,
                                         shape=[self.batch_size],
                                         name="output")
        else:
            self.input = tf.placeholder(dtype=tf.float32,
                                        shape=(1, None, self.vector_dim),
                                        name="input")
            self.output = tf.placeholder(dtype=tf.float32,
                                         shape=[1],
                                         name="output")

        # pad all input sentence vector number to self.sentence_max_num
        self.padded_input = tf.pad(self.input,
                                   paddings=[(0, 0), (0, self.max_rows - tf.shape(self.input)[1]), (0, 0)],
                                   constant_values=0,
                                   name="padded_input")

        # create the first convolution layer
        self.layer_conv_0 = []
        self.layer_max_pool_0 = []
        max_pool_out_0 = []
        conv_dim_map = {}
        count = 0

        for cv in self.conv_dim[0]:
            for i in range(0, self.feature_maps[0]):
                self.layer_conv_0.append(conv2d(tf.expand_dims(self.padded_input, -1),
                                                  kernel_h=cv,
                                                  kernel_w=self.vector_dim,
                                                  stride_h=1,
                                                  stride_w=1,
                                                  name="conv_0_%d_%d" % (cv, i)))
                conv_dim_map[count] = cv
                count += 1

            # create the max-pooling layer + relu layer + batch norm layer
            for i in range(0, self.feature_maps[0]):
                self.layer_max_pool_0.append(
                    batch_norm(
                        tanh(
                            max_pool_1_1d(self.layer_conv_0[i], self.max_rows - conv_dim_map[i] + 1)
                        ),
                    name="batch_norm_0_%d_%d" % (cv, i),
                    training=self.is_training))

            max_pool_out_0.append(tf.concat(self.layer_max_pool_0[-self.feature_maps[0]:], 2))
        max_pool_out_0 = tf.concat(max_pool_out_0, 1)
        tf.summary.histogram('max_pool_0', max_pool_out_0)

        # create the second convolution layer
        self.layer_conv_1 = []
        self.layer_max_pool_1 = []
        max_pool_out_1 = []
        conv_dim_map = {}
        count = 0

        for cv in self.conv_dim[1]:
            for i in range(0, self.feature_maps[1]):
                self.layer_conv_1.append(conv2d(max_pool_out_0,
                                                kernel_h=len(self.conv_dim[0]),
                                                kernel_w=cv,
                                                stride_h=1,
                                                stride_w=1,
                                                name="conv_1_%d_%d" % (cv, i)))
                self.layer_max_pool_1.append(
                    batch_norm(
                        tanh(
                            tf.nn.max_pool(self.layer_conv_1[-1],
                                           ksize=[1, 1, self.layer_conv_1[-1].get_shape()[2], 1],
                                           strides=[1, 1, 1, 1], padding="VALID", name="max_pool_1")
                        ),
                        name="batch_norm_1_%d_%d" % (cv, i),
                        training=self.is_training))
                count += 1


        max_pool_out_1 = tf.concat(self.layer_max_pool_1, 2)

        tf.summary.histogram('max_pool_1', max_pool_out_1)

        # create MLP layer
        self.layer_mlp = MLP(tf.reshape(max_pool_out_1, [max_pool_out_1.get_shape()[0], len(self.layer_conv_1)]),
                             dim_in=len(self.layer_conv_1),
                             dim_out=self.output_dim,
                             layer_sizes=self.hidden_units,
                             dropout_rates=self.dropout_rates,
                             activations=self.hidden_activation)

        # define loss function
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.layer_mlp,
                                                           labels=self.labels)
        )

        # define predict result function
        tf.summary.histogram('input', self.input)
        tf.summary.histogram('labels', self.labels)
        tf.summary.histogram('softmax', self.layer_mlp )
        self.output = tf.argmax(self.layer_mlp, axis=1)
        # initialize other utilities
        tf.summary.scalar("loss", self.loss)
        self.vars = tf.trainable_variables()
        self.writer = tf.summary.FileWriter(logdir=self.summary_dir)
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()

    def train(self,
              data: Any,
              labels: Any,
              learning_rate=0.0001,
              beta1=0.5,
              epochs=25,
              save_every_round=500):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                beta1=beta1) \
            .minimize(self.loss, var_list=self.vars)

        tf.global_variables_initializer().run()



        # start training
        round_counter = 0
        is_loaded, checkpoint_counter = self.load(self.checkpoint_dir)
        if is_loaded:
            round_counter = checkpoint_counter
        if self.enable_batch:
            current_epoch = round_counter // (len(data) // self.batch_size)
        else:
            current_epoch = round_counter // len(data)

        current_time = time.time()
        for epoch in range(current_epoch, epochs):
            if self.enable_batch:
                round_size = len(data) // self.batch_size
                for idx in range(0, round_size):
                    batch_data = data[idx * self.batch_size : (idx + 1) * self.batch_size]
                    batch_labels = labels[idx * self.batch_size : (idx + 1) * self.batch_size]

                    # pad batches with different size samples
                    max_h = np.max([ sample.shape[0] for sample in batch_data])
                    padded = np.zeros([self.batch_size, max_h, self.vector_dim])
                    for i in range(0, self.batch_size):
                        padded[i][0:batch_data[i].shape[0]] = batch_data[i]
                    _, summary_result, current_loss = self.sess.run([self.optimizer, self.summary, self.loss],
                                                                    feed_dict={
                                                                        self.is_training: True,
                                                                        self.input: padded,
                                                                        self.labels: batch_labels
                                                                    })
                    self.writer.add_summary(summary_result, round_counter)
                    if round_counter % save_every_round == 1:
                        self.save(self.checkpoint_dir, round_counter)
                    round_counter += 1

                    new_time = time.time()
                    if round_counter >= epochs * round_size:
                        return
                    if new_time - current_time >= 0.1:
                        utils.logger.info("Epoch: [%2d/%2d] [%4d/%4d], time: %3f loss: %.8f" \
                                          % (epoch + 1, epochs, round_counter % round_size, round_size,
                                             new_time - current_time, current_loss))
                        current_time = new_time

            else:
                round_size = len(data)
                for idx in range(0, round_size):
                    _, summary_result, current_loss = self.sess.run([self.optimizer, self.summary, self.loss],
                                                                    feed_dict={
                                                                        self.is_training: True,
                                                                        self.input: np.expand_dims(data[idx], axis=0),
                                                                        self.labels: np.array([labels[idx]])
                                                                    })
                    self.writer.add_summary(summary_result, round_counter)
                    if round_counter % save_every_round == 1:
                        self.save(self.checkpoint_dir, round_counter)
                    round_counter += 1

                    new_time = time.time()
                    if round_counter >= epochs * round_size:
                        return
                    if new_time - current_time >= 0.1:
                        utils.logger.info("Epoch: [%2d/%2d] [%4d/%4d], time: %3f loss: %.8f" \
                                          % (epoch + 1, epochs, round_counter % round_size, round_size,
                                             new_time - current_time, current_loss))
                        current_time = new_time


    def predict(self,
                data: Any) -> np.ndarray:
        result = np.full(len(data), -1, np.int32)
        if self.enable_batch:
            for idx in range(0, len(data) // self.batch_size):
                batch_data = data[idx * self.batch_size: (idx + 1) * self.batch_size]
                # pad batches with different size samples
                max_h = np.max([sample.shape[0] for sample in batch_data])
                padded = np.zeros([self.batch_size, max_h, self.vector_dim])
                for i in range(0, self.batch_size):
                    padded[i][0:batch_data[i].shape[0]] = batch_data[i]

                result[idx*self.batch_size: (idx+1)*self.batch_size] = self.sess.run([self.output],
                                            feed_dict={
                                                self.is_training: False,
                                                self.input: padded,
                                            })[0]
        else:
            for idx in range(0, len(data)):
                result[idx] = self.sess.run([self.output],
                                             feed_dict={
                                                 self.is_training: False,
                                                 self.input: np.expand_dims(data[idx], axis=0),
                                             })[0]
        return result

    def save(self,
             checkpoint_dir: str,
             step: int) -> None:
        model_name = "CNN.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load_file(self,
                  meta_file: str) -> Tuple[bool, int]:
        utils.logger.info("Reading checkpoints...")
        try:
            self.saver.restore(self.sess, meta_file)
        except:
            utils.logger.info("Failed to read {}".format(meta_file))
            return False, -1
        utils.logger.info("Success to read {}".format(meta_file))
        counter = int(re.findall("(\d+)(?!.*\d)", os.path.basename(meta_file))[0])
        return True, counter

    def load(self,
             checkpoint_dir: str) -> Tuple[bool, int]:
        utils.logger.info("Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            utils.logger.info("Success to read {}".format(ckpt_name))
            return True, counter
        else:
            utils.logger.warn("Failed to find a checkpoint")
            return False, 0
