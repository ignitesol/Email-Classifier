# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import re
import pandas as pd
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from scipy import stats
import sqlite3
import sqlalchemy

# placeholder for custom exceptions ###############################################################
class CustomException(Exception):
    pass

# function to extract list of features ############################################################
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


# classifier class ################################################################################
class FeatureClassifier:
    # init with feature_extraction_method and storage_db_filename
    def __init__(self, get_features, filename=None):
        self.feature_category_count = pd.DataFrame() # number of features by category
        self.category_count = pd.Series().rename('n_items') # number of items in each category
        self.get_features = get_features # function to extract features

    # increment the (feature,category) count
    def increment_feature_category_count(self, features, categories):
        feature_category_ones = pd.DataFrame(1,index=features,columns=categories)
        self.feature_category_count = self.feature_category_count\
                                            .add(feature_category_ones,fill_value=0)\
                                            .fillna(0)
        self.feature_category_count.index.name = 'features'
        self.feature_category_count.columns.name = 'categories'

    # increment the item count in a category
    def increment_category_count(self, categories):
        try:
            self.category_count[categories] += 1
        except:
            for cat in categories:
                try:
                    self.category_count[cat] += 1
                except:
                    self.category_count[cat] = 1
                    self.category_count.index.name = 'categories'

    # train classifier given an item and category
    def train(self, item, categories):
        features = self.get_features(item)
        self.increment_feature_category_count(features,categories)
        self.increment_category_count(categories)

    # number of times a feature occurred in a category - (feature,category) value
    def get_feature_category_count(self, feature, category):
        try:
            return self.feature_category_count.ix[feature][category]
        except:
            return 0

    # total number of items in a category
    def get_category_count(self, category):
        try:
            return self.category_count[category]
        except:
            return 0

    # total number of items
    def get_items_count(self):
        return self.category_count.sum()

    # list all categories
    def get_categories(self):
        return list(self.category_count.index)

    # probability that a givem feature will appear in an item belonging to given category
    def get_feature_category_prob(self, feature, category):
        try:
            return self.get_feature_category_count(feature, category)\
                    /self.get_category_count(category)
        except:
            return 0

    # weighted probablility - p(feature/category)
    def get_feature_category_weightedprob(self, feature, category, probfunc,
                                          init_weight=1, init_prob=0.5):
        try:
            feature_prob = probfunc(feature, category)
            feature_count = self.feature_category_count.sum(axis=1)[feature]
            prob = (init_weight*init_prob + feature_count*feature_prob)/(init_weight+feature_count)
            return prob
        except:
            return init_prob


# training function ###############################################################################
def train_classifier(cl):
    cl.train('Nobody owns the water.',['good'])
    cl.train('the quick rabbit jumps fences',['good'])
    cl.train('buy pharmaceuticals now',['ugly','bad'])
    cl.train('make quick money at the online casino',['bad'])
    cl.train('the quick brown fox jumps',['good'])