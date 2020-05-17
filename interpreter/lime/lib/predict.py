#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : cyl
# @Time : 2020/5/11 16:45
from interpreter.lime.utils.insurance_dataset import InsuranceDataset
from interpreter.lime.utils.base import encode_onehot, split_sample
from interpreter.lime.common.constant import *
from interpreter.lime.lib.model import Model

class Singleton(type):
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls._instance = None

    def __call__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__call__(*args, **kw)
        return cls._instance


class Predict():
    __metaclass__ = Singleton

    def __init__(self):
        self.insur_dataset = InsuranceDataset()
        self.encode_onehot = encode_onehot()
        self.model_predict = Model().xgb_predict()

    def predict_proba(self):
        """
        prediction probabilities,key is class name,value is probability.
        exmaple:{0: 0.8450577855110168, 1: 0.15494222939014435}
        :return:
        """
        predict_proba = self.model_predict.predict_proba
        class_names =  self.model_predict.class_names
        return dict(zip(class_names,predict_proba))

    def feature_list(self):
        """
        text explanation.
            example:[('0.00 < HHInsurance <= 1.00', -0.34299729931535844), ('CarLoan <= 0.00', 0.20479311844631276)]
        :return:
        """
        list = self.model_predict.as_list()

        result = {}
        for item in list:
            result[item[0]] = item[1]

        return result

    def feature_max(self):
        """
        get the maximum positive and negative correlation
       example:[(0, '0.00 < HHInsurance <= 1.00', -0.3485989808506065), (1, 'CarLoan <= 0.00', 0.22576244589728836)]
       the first parameter is class name,the second parameter is feature name, the third parameter is value.
        :return:
        """
        exp_list = self.feature_list()

        # turn list into dict
        exp_dict = exp_list
        # for exp in exp_list.items():
        #     exp_dict[exp[0]] = exp[1]

        feature_max_pos = max(exp_dict.keys(), key=(lambda x: exp_dict[x]))
        feature_max_neg = min(exp_dict.keys(), key=(lambda x: exp_dict[x]))

        result = {}

        result['0'] = {feature_max_neg: exp_dict[feature_max_neg]}
        result['1'] = {feature_max_pos: exp_dict[feature_max_pos]}

        # class_names = self.model_predict.class_names
        # feature_max_keys = [feature_max_neg,feature_max_pos]
        # feature_max_values = [exp_dict[feature_max_neg],exp_dict[feature_max_pos]]
        return result

    def feature_value(self):
        """
        get the feature and value.
        example;{'Age': '53.00', 'Job=retired': 'True', 'Marital=married': 'True', 'Education=secondary': 'True',
         'Balance': '665.00', 'HHInsurance': '1.00', 'CarLoan': '0.00', 'NoOfContacts': '2.00', 'DaysPassed': '-1.00',
          'PrevAttempts': '0.00'}
        :return:
        """
        domain_mapper = self.model_predict.domain_mapper
        feature_names = domain_mapper.feature_names
        feature_values = domain_mapper.feature_values
        return dict(zip(feature_names, feature_values))





