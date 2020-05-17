#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : cyl
# @Time : 2020/5/13 8:30
import os
# all features
FEATURE_ALL= ["Age", "Job", "Marital", "Education", "Balance", "HHInsurance", "CarLoan",
               "NoOfContacts", "DaysPassed", "PrevAttempts",  "CarInsurance"]
# nonnumeric features
CATEGORICAL= ["Job", "Marital", "Education"]

# useful features
FEATURE_NAMES = ["Age", "Job", "Marital", "Education", "Balance", "HHInsurance", "CarLoan",
                 "NoOfContacts", "DaysPassed", "PrevAttempts"]

#  nonnumeric features ("Job", "Marital", "Education","Outcome") index
CATEGORICAL_FEATURES = [1, 2, 3]

CSV_INSURANCE = "carInsurance_train.csv"

PATH_CUR = os.path.abspath(os.path.dirname(__file__))
PATH_MODEL = os.path.join(PATH_CUR, "../data", "model")
MODEL_XGBOOST = os.path.join(PATH_MODEL, "xgboost.model")