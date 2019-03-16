# coding: utf-8

import os
import math
import cmath
import random
import datetime
import numpy as np
import scipy.io as sio
import tensorflow as tf


def normalized(vec):
    return vec / np.linalg.norm(vec)


# -----------------配置-----------------
file_name = ('depth_range_Data.mat', 'td', '/home/passerby/Desktop/model.ckpt', 'record.txt')
sample_size = (20, 250, 100)
snapshots = 10
snr = 5
class_nu = (5, 10, 50, 10, 500, 500)
ratio_train = 0.8
ratio_test = 0.1
n_epoch = 300
# -----------------结束-----------------
load_file = file_name[0]
load_data = sio.loadmat(load_file)
load_matrix = load_data[file_name[1]]
model_path = file_name[2]
file_name = open(file_name[3], "a")
input_images = np.zeros([sample_size[1] * sample_size[2], sample_size[0] * 2, sample_size[0] * 2, 1], np.float32)
labels = []

for i in range(sample_size[2]):
    for j in range(sample_size[1]):
        port_pre = np.zeros(sample_size[0], np.complex64)
        for k in range(sample_size[0]):
            port_pre[k] = load_matrix[k][j][i]
        port_pre = normalized(port_pre)
        port_pre_cun = np.zeros((sample_size[0], sample_size[0]), np.complex64)
        for k in range(snapshots):
            port = [normalized(port_pre + np.random.randn(sample_size[0]) * np.sqrt(
                (sum([np.real(m) * np.real(m) + np.imag(m) * np.imag(m) for m in port_pre]) / sample_size[0] * 1.) / (
                        10 ** (snr / 10.))))]
            port_pre_cun += (np.dot(np.transpose(port), port))
        port_1 = (port_pre_cun / snapshots * 1.)
        port_1_real = np.real(port_1)
        port_1_imag = np.imag(port_1)
        sample_nu = i * sample_size[1] + j
        for m in range(sample_size[0]):
            for n in range(sample_size[0]):
                input_images[sample_nu][m][n][0] = port_1_real[m][n]
                input_images[sample_nu][m][sample_size[0] + n][0] = port_1_imag[m][n]
                input_images[sample_nu][m + sample_size[0]][n][0] = -1 * port_1_imag[m][n]
                input_images[sample_nu][m + sample_size[0]][sample_size[0] + n][0] = port_1_real[m][n]
        labels.append(i / class_nu[1] * class_nu[2] + j / class_nu[0])
input_labels = np.asarray(labels, np.int32)

arr = np.arange(sample_size[1] * sample_size[2])
np.random.shuffle(arr)
input_images = input_images[arr]
input_labels = input_labels[arr]

s = np.int(sample_size[1] * sample_size[2] * ratio_train)
s_1 = np.int(sample_size[1] * sample_size[2] * ratio_test)
x_train = input_images[:s]
y_train = input_labels[:s]
x_val = input_images[s:s + s_1]
y_val = input_labels[s:s + s_1]
x_prv = input_images[s + s_1:]
y_prv = input_labels[s + s_1:]
file_name.write("start:" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
# -----------------构建网络-----------------
x_image = tf.placeholder(tf.float32, shape=[None, sample_size[0] * 2, sample_size[0] * 2, 1], name='x_image')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')


def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        # 定义第一个卷积层的variables和ops
        conv1_weights = tf.get_variable("weight", [7, 7, 1, 32], initializer=tf.variance_scaling_initializer())
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        # 定义第二个卷积层的variables和ops
        conv2_weights = tf.get_variable("weight", [5, 5, 32, 64], initializer=tf.variance_scaling_initializer())
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable("weight", [3, 3, 64, 128], initializer=tf.variance_scaling_initializer())
        conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        nodes = 5 * 5 * 128  # pre
        reshaped = tf.reshape(pool3, [-1, nodes])

    with tf.variable_scope('layer7-fc1'):
        # 全连接层1
        fc1_weights = tf.get_variable("weight", [nodes, 1024], initializer=tf.variance_scaling_initializer())
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        # dropout
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer8-fc2'):
        # 全连接层2
        fc2_weights = tf.get_variable("weight", [1024, class_nu[4]], initializer=tf.variance_scaling_initializer())
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [class_nu[4]], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit


# -----------------网络结束-----------------
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logits = inference(x_image, False, regularizer)

# (小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1, dtype=tf.float32)
logits_eval = tf.multiply(logits, b, name='logits_eval')

# 定义优化器和训练op
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
mad = tf.cast(tf.abs(tf.subtract(tf.cast(tf.argmax(logits, 1), tf.int32), y_)), tf.float32)


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, (len(inputs) - batch_size + 1 > 0) and (len(inputs) - batch_size + 1) or len(inputs),
                           batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# 训练和测试数据，可将n_epoch设置更大一些
batch_size = class_nu[5]
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(n_epoch):
    # training
    print(epoch)
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x_image: x_train_a, y_: y_train_a})
        train_loss += err;
        train_acc += ac;
        n_batch += 1
    file_name.write("train loss: %f" % (np.sum(train_loss) / n_batch))
    file_name.write("train acc: %f" % (np.sum(train_acc) / n_batch))
    print("train loss: %f" % (np.sum(train_loss) / n_batch))
    print("train acc: %f" % (np.sum(train_acc) / n_batch))

    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x_image: x_val_a, y_: y_val_a})
        val_loss += err;
        val_acc += ac;
        n_batch += 1
    # print val_loss, val_acc, n_batch
    file_name.write("   validation loss: %f" % (np.sum(val_loss) / n_batch))
    file_name.write("   validation acc: %f\n" % (np.sum(val_acc) / n_batch))
    print("   validation loss: %f" % (np.sum(val_loss) / n_batch))
    print("   validation acc: %f\n" % (np.sum(val_acc) / n_batch))
file_name.write("stop:" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
prv_loss, prv_acc, n_batch = 0, 0, 0
for x_prv_a, y_prv_a in minibatches(x_prv, y_prv, batch_size, shuffle=False):
    err, ac = sess.run([mad, acc], feed_dict={x_image: x_prv_a, y_: y_prv_a})
    prv_loss += err;
    prv_acc += ac;
    n_batch += 1
file_name.write("   prove mad: %f" % (np.sum(prv_loss) / (sample_size[1] * sample_size[2] - s - s_1)))
file_name.write("   prove acc: %f" % (np.sum(prv_acc) / n_batch))
print("   prove mad: %f" % (np.sum(prv_loss) / (sample_size[1] * sample_size[2] - s - s_1)))
print("   prove acc: %f" % (np.sum(prv_acc) / n_batch))
# save_fn = $file_name
# sio.savemat(save_fn, {'x_prv': x_prv, 'y_prv': y_prv})
saver.save(sess, model_path)
sess.close()
file_name.close()
