#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : cyl
# @Time : 2020/5/13 8:44 
import pandas as pd
import sklearn

def pd_to_np(pd_data):
    """
    transform pandans to numpy
    :param pd_data:
    :return:
    """
    return pd.DataFrame.to_numpy(pd_data)


def encode_onehot():
    """
    encode nonnumeric features by one-hot
    :return:OneHotEncoder object
    """
    return sklearn.preprocessing.OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')

def split_sample(data, labels):
    return sklearn.model_selection.train_test_split(data, labels, train_size=0.80)