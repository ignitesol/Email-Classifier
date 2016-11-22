# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import re
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import sqlite3
import sqlalchemy


### function to extract list of features #########################################################
def get_unique_words(email_txt):
    '''
    list all the unique words in a text
    '''
    # list all email ids
    regex_for_email_ids = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    email_ids = re.findall(regex_for_email_ids, email_txt)
    # splitter with non-alphabetic chars, with the exception of @
    regex_for_splitter = r'[\W]+'
    splitter = re.compile(regex_for_splitter)
    # split text with splitter as separator
    words = [s.lower() for s in re.split(splitter, email_txt) if len(s)>2]
    return list(set(words)) + email_ids


### classifier class #############################################################################
class classifier:
    # init with feature_extraction_menthod and storage_db_filename
    def __init__(self, get_features, filename=None):
        self.feature_category_count = pd.DataFrame() # number of features by category
        self.get_features = get_features # function to extract features

    # increment the (feature,category) count
    def increment_feature_category_count(self, feature, category):
        feature_category_ones = pd.DataFrame(1,index=feature,columns=category)
        self.feature_category_count = self.feature_category_count\
                                            .add(feature_category_ones,fill_value=0)\
                                            .fillna(0)

    # number of times a feature occured in a category - (feature,category) value
    def get_feature_category_count(self, feature, category):
        try:
            return self.feature_category_count.ix[feature][category]
        except:
            return 0

    # total number of items in a category
    def get_category_count(self, category):
        try:
            return self.feature_category_count[category].sum(skipna=True)
        except:
            return 0

    # total number of items
    def get_items_count(self):
        return self.feature_category_count.sum(skipna=True).sum()

    # list all categories
    def get_categories(self):
        return list(self.feature_category_count.columns)

    # train classifier given an item and category
    def train(self, item, category):
        features = self.get_features(item)
        self.increment_feature_category_count(features,category)
