#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : cyl
# @Time : 2020/5/15 15:29
import sklearn
import sklearn.ensemble
import numpy as np
import lime.lime_tabular
import pandas as pd
import xgboost
import joblib
from interpreter.lime.utils.insurance_dataset import InsuranceDataset
from interpreter.lime.utils.base import encode_onehot, split_sample
from interpreter.lime.common.constant import *


class Model():

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
        # save model to file
        if not os.path.exists(MODEL_XGBOOST):
            # use gradient boosted trees as the model
            gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
            gbtree.fit(encoded_train, labels_train)
            joblib.dump(gbtree, MODEL_XGBOOST)

        # load model from file
        gbtree = joblib.load(MODEL_XGBOOST)
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