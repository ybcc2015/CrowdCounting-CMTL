# -*- coding: utf-8 -*-
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from model import CMTL
from utils.metrics import MAE, MSE
import os
import argparse


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset = args.dataset  # 'A' or 'B'

    # 定义模型
    input_shape = (None, None, 1)
    model = CMTL(input_shape)
    # 编译
    adam = Adam(lr=0.00001)
    loss = {'output_density': 'mse', 'output_class': 'categorical_crossentropy'}
    loss_weights = {'output_density': 1.0, 'output_class': 0.0001}
    model.compile(optimizer=adam, loss=loss, loss_weights=loss_weights,
                  metrics={'output_density': [MAE, MSE]})
