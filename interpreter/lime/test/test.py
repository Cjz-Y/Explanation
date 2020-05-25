#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : cyl
# @Time : 2020/5/25 17:06 
from interpreter.lime.lib.predict import Predict

if __name__ == '__main__':
    predict = Predict(num_features=5)
    print(predict.feature_value())
    print(predict.predict_proba())
    print(predict.feature_list())