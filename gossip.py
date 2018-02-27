# -*- coding: utf-8 -*-
# Copyright 2017 The Xiaoyu Fang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import sys
import numpy
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras.backend as K

from windpuller import WindPuller
from dataset import DataSet
from feature import extract_from_file
import math
from config import exchange_fee, selector, points_for_test, input_shape
from rawdata import RawData, read_sample_data
from chart import extract_feature
from datetime import datetime


def read_ultimate_feature_from_file(x_path, y_path, input_shape):
    features = numpy.loadtxt(x_path + str(input_shape[0]))
    features = numpy.reshape(features, [-1, input_shape[0], input_shape[1]])
    labels = numpy.loadtxt(y_path + str(input_shape[0]))
    # test_labels = numpy.reshape(test_labels, [-1, 1])
    dataset = DataSet(features, labels)
    return dataset


def read_ultimate(path, input_shape):
    ultimate_features = numpy.loadtxt(path + "ultimate_feature." + str(input_shape[0]))
    ultimate_features = numpy.reshape(ultimate_features, [-1, input_shape[0], input_shape[1]])
    ultimate_labels = numpy.loadtxt(path + "ultimate_label." + str(input_shape[0]))
    # ultimate_labels = numpy.reshape(ultimate_labels, [-1, 1])
    train_set = DataSet(ultimate_features, ultimate_labels)
    test_features = numpy.loadtxt(path + "ultimate_feature.test." + str(input_shape[0]))
    test_features = numpy.reshape(test_features, [-1, input_shape[0], input_shape[1]])
    test_labels = numpy.loadtxt(path + "ultimate_label.test." + str(input_shape[0]))
    # test_labels = numpy.reshape(test_labels, [-1, 1])
    test_set = DataSet(test_features, test_labels)
    return train_set, test_set


def read_feature(path, input_shape, prefix):
    ultimate_features = numpy.loadtxt("%s/%s_feature.%s" % (path, prefix, str(input_shape[0])))
    ultimate_features = numpy.reshape(ultimate_features, [-1, input_shape[0], input_shape[1]])
    ultimate_labels = numpy.loadtxt("%s/%s_label.%s" % (path, prefix, str(input_shape[0])))
    # ultimate_labels = numpy.reshape(ultimate_labels, [-1, 1])
    train_set = DataSet(ultimate_features, ultimate_labels)
    test_features = numpy.loadtxt("%s/%s_feature.test.%s" % (path, prefix, str(input_shape[0])))
    test_features = numpy.reshape(test_features, [-1, input_shape[0], input_shape[1]])
    test_labels = numpy.loadtxt("%s/%s_label.test.%s" % (path, prefix, str(input_shape[0])))
    # test_labels = numpy.reshape(test_labels, [-1, 1])
    test_set = DataSet(test_features, test_labels)
    return train_set, test_set


def calculate_market_return(labels):
    mr = []
    if len(labels) <= 0:
        return mr
    mr.append(1. + labels[0])
    for l in range(1, len(labels)):
        mr.append(mr[l - 1] * (1 + labels[l]))
    return mr


def predict(model_path, input_path="dataset_back/new_data.csv", output_path='predict_output.' + str(input_shape[0])):
    # extract feature from file
    file_path = input_path
    raw_data = read_sample_data(file_path)
    del raw_data[-1]  # remove the last element, because the kline of it just started
    window = input_shape[0]
    moving_features = extract_feature(raw_data=raw_data, selector=selector, window=window,
                                      with_label=False, flatten=True)

    # preprocess the file
    # read feature from file and reshape
    # new_data_features = numpy.loadtxt(file_path)
    new_data_features = numpy.reshape(moving_features, [-1, input_shape[0], input_shape[1]])
    saved_wp = WindPuller(input_shape).load_model(model_path)
    # predict and get result
    pred = saved_wp.predict(new_data_features, 1024)
    # print('result is 1!')
    print(pred[len(pred) - 1])
    naive_dt = datetime.now()
    with open(output_path, 'a') as fp:
        fp.write(str(naive_dt) + "\t"
                 + model_path + "\t"
                 + str(pred[len(pred) - 1][0]))
        fp.write('\n')


def make_model(input_shape, nb_epochs=100, batch_size=128, lr=0.01, n_layers=1, n_hidden=16, rate_dropout=0.3):
    model_path = '/output/model.{}.{}c.{}l.{}'.format(input_shape[0], n_hidden, n_layers, nb_epochs)

    wp = WindPuller(input_shape=input_shape, lr=lr, n_layers=n_layers, n_hidden=n_hidden, rate_dropout=rate_dropout)

    train_set, test_set = read_ultimate("./", input_shape)
    wp.fit(train_set.images, train_set.labels, batch_size=batch_size,
           nb_epoch=nb_epochs, shuffle=True, verbose=1,
           validation_split=0.02,
           # validation_data=(test_set.images, test_set.labels),
           callbacks=[TensorBoard(histogram_freq=1),
                      ModelCheckpoint(filepath=model_path + '.best', save_best_only=True, mode='min')])

    scores = wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    wp.model.save(model_path)
    saved_wp = wp.load_model(model_path)
    scores = saved_wp.evaluate(test_set.images, test_set.labels, verbose=0)
    print('Test loss:', scores[0])
    print('test accuracy:', scores[1])
    pred = saved_wp.predict(test_set.images, 1024)
    # print(pred)
    # print(test_set.labels)
    pred = numpy.reshape(pred, [-1])
    result = numpy.array([pred, test_set.labels]).transpose()
    with open('output.' + str(input_shape[0]), 'w') as fp:
        for i in range(result.shape[0]):
            for val in result[i]:
                fp.write(str(val) + "\t")
            fp.write('\n')


if __name__ == '__main__':
    K.clear_session()

    operation = "train"
    if len(sys.argv) > 1:
        operation = sys.argv[1]
    if operation == "train":
        make_model([30, 61], 100, 512, n_layers=2, lr=0.001)
    elif operation == "predict":
        model_path = "model.30.best"
        if len(sys.argv) > 2:
            # use_high_as_close = True if sys.argv[2] == 'use_high_as_close' else False
            model_path = "model.2LSTM.short.30.best.3.5" if sys.argv[2] == 'short_model' else "model.2LSTM.long.30.best.3.5"

        input_path = "dataset_back/new_data.csv"
        output_path = 'predict_output.' + str(input_shape[0])
        if len(sys.argv) > 3:
            input_path = sys.argv[3]
            output_path = sys.argv[4]

        predict(model_path, input_path, output_path)
    else:
        print("Usage: gossip.py [train | evaluate | predict | clear_session]")
