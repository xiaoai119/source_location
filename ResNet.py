# coding: utf-8
from __future__ import division, print_function, absolute_import

import os
import math
import cmath
import random
import tflearn
import datetime
import numpy as np
import scipy.io as sio
import tensorflow as tf

# Residual blocks 32 layers: n=5, 56 layers: n=9, 110 layers: n=18

n = 9
load_file = 'sample.mat'
input_size = (40, 40)
class_nu = (25, 0)
n_epoch = 100

load_data = sio.loadmat(load_file)
x_train = load_data['x_train']
y_train = load_data['y_train']
x_val = load_data['x_val']
y_val = load_data['y_val']
x_prv = load_data['x_prv']
y_prv = load_data['y_prv']

# Building Residual Network
net = tflearn.input_data(shape=[None, input_size[0], input_size[1], 1])
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.residual_block(net, n, 16)
net = tflearn.residual_block(net, 1, 32, downsample=True)
net = tflearn.residual_block(net, n-1, 32)
net = tflearn.residual_block(net, 1, 64, downsample=True)
net = tflearn.residual_block(net, n-1, 64)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, class_nu[0], activation='softmax')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=mom, loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, checkpoint_path='model_multi_s', max_checkpoints=1, tensorboard_verbose=0, clip_gradients=0.)
# model.load("model_multi_s-95000")
model.fit(x_train, y_train, n_epoch=n_epoch, validation_set=(x_val, y_val), snapshot_epoch=False, snapshot_step=500, show_metric=True, batch_size=128, shuffle=True, run_id='resnet_multi_s') 

