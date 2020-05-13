import os
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing

from interpreter.lime.common.constant import *
from interpreter.lime.utils.base import pd_to_np

class InsuranceDataset():
    """
    handle data && load data
    """

    def __init__(self, dirpath=None):
        self._dirpath = dirpath
        if not self._dirpath:
            self._dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../data', 'insurance_data')


    def load(self, data_train=CSV_INSURANCE):
        """
        load and handle data.
        :param data_train: train dataset name
        :return:
        """
        self._train_filepath = os.path.join(self._dirpath, data_train)

        data = pd.read_csv(self._train_filepath)
        data = data[FEATURE_ALL]

        # use U to fill NAN
        for category in CATEGORICAL:
            data[category].fillna("U",inplace=True)

        data = pd_to_np(data)

        # encode lable car insurance
        labels = data[:, -1]
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        class_names = le.classes_

        # get rid of the last column
        data = data[:, :-1]

        # lableEncode nonnumeric features
        categorical_names = {}
        for feature in CATEGORICAL_FEATURES:
            le = sklearn.preprocessing.LabelEncoder()
            le.fit(data[:, feature])
            data[:, feature] = le.transform(data[:, feature])
            categorical_names[feature] = le.classes_

        data = data.astype(float)

        # Ensure that the generated random Numbers are predictable
        np.random.seed(1)
        return data, labels, class_names, categorical_names





