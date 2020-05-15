#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : cyl
# @Time : 2020/5/11 16:45 
import sklearn
import sklearn.ensemble
import numpy as np
import lime.lime_tabular
from lime.lime_tabular import TableDomainMapper
import pandas as pd
import xgboost
from interpreter.lime.utils.insurance_dataset import InsuranceDataset
from interpreter.lime.utils.base import encode_onehot, split_sample
from interpreter.lime.common.constant import *

class Predict():
    def __init__(self):
        self.insur_dataset = InsuranceDataset()
        self.encode_onehot = encode_onehot()

    def pd_to_np(self, pd_data):
        return pd.DataFrame.to_numpy(pd_data)

    def rf_predict(self):

        data, labels, class_names, categorical_names= self.insur_dataset.load()
        train, test, labels_train, labels_test = split_sample(data, labels)

        self.encode_onehot.fit(data)
        encoded_train = self.encode_onehot.transform(train)

        # use RandomForestClassifier
        rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)

        # fit
        rf.fit(encoded_train, labels_train)

        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=train,
                                                           feature_names=FEATURE_NAMES,
                                                           class_names=class_names,
                                                           categorical_features=CATEGORICAL_FEATURES,
                                                           categorical_names=categorical_names, kernel_width=3)
        predict_fn = lambda x: rf.predict_proba(self.encode_onehot.transform(x)).astype(float)

        # pick one at random
        i = np.random.randint(0, test.shape[0])
        exp = explainer.explain_instance(test[i], predict_fn, num_features=10, top_labels=2)

        return exp.as_list()

    def xgb_predict(self):
        data, labels, class_names, categorical_names= self.insur_dataset.load()
        train, test, labels_train, labels_test = split_sample(data, labels)

        self.encode_onehot.fit(data)
        encoded_train = self.encode_onehot.transform(train)

        # use gradient boosted trees as the model
        gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
        gbtree.fit(encoded_train, labels_train)

        # accuracy score
        acc_score = sklearn.metrics.accuracy_score(labels_test, gbtree.predict(self.encode_onehot.transform(test)))

        predict_fn = lambda x: gbtree.predict_proba(self.encode_onehot.transform(x)).astype(float)

        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=train,
                                                           feature_names=FEATURE_NAMES,
                                                           class_names=class_names,
                                                           categorical_features=CATEGORICAL_FEATURES,
                                                           categorical_names=categorical_names, kernel_width=3)
        np.random.seed(1)

        i = 37
        # pick a random instance
        # i= np.random.randint(0, test.shape[0])

        exp = explainer.explain_instance(test[i], predict_fn, num_features=10)

        return exp

    def predict_proba(self):
        """
        prediction probabilities,key is class name,value is probability.
        exmaple:{0: 0.8450577855110168, 1: 0.15494222939014435}
        :return:
        """
        predict_proba = self.xgb_predict().predict_proba
        class_names =  self.xgb_predict().class_names
        return dict(zip(class_names,predict_proba))

    def feature_list(self):
        """
        text explanation.
            example:[('0.00 < HHInsurance <= 1.00', -0.34299729931535844), ('CarLoan <= 0.00', 0.20479311844631276)]
        :return:
        """
        return self.xgb_predict().as_list()

    def feature_max(self):
        """
        get the maximum positive and negative correlation
       example:[(0, '0.00 < HHInsurance <= 1.00', -0.3485989808506065), (1, 'CarLoan <= 0.00', 0.22576244589728836)]
       the first parameter is class name,the second parameter is feature name, the third parameter is value.
        :return:
        """
        exp_list = self.feature_list()

        # turn list into dict
        exp_dict = {}
        for exp in exp_list:
            exp_dict[exp[0]] = exp[1]

        feature_max_pos = max(exp_dict.keys(), key=(lambda x: exp_dict[x]))
        feature_max_neg = min(exp_dict.keys(), key=(lambda x: exp_dict[x]))
        class_names = self.xgb_predict().class_names
        feature_max_keys = [feature_max_neg,feature_max_pos]
        feature_max_values = [exp_dict[feature_max_neg],exp_dict[feature_max_pos]]
        return list(zip(class_names, feature_max_keys, feature_max_values))

    def feature_value(self):
        """
        get the feature and value.
        example;{'Age': '53.00', 'Job=retired': 'True', 'Marital=married': 'True', 'Education=secondary': 'True',
         'Balance': '665.00', 'HHInsurance': '1.00', 'CarLoan': '0.00', 'NoOfContacts': '2.00', 'DaysPassed': '-1.00',
          'PrevAttempts': '0.00'}
        :return:
        """
        domain_mapper = self.xgb_predict().domain_mapper
        feature_names = domain_mapper.feature_names
        feature_values = domain_mapper.feature_values
        return dict(zip(feature_names, feature_values))





