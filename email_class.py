# -*- coding: utf-8 -*-
"""
@author: srikant
"""
import re
import pandas as pd
import numpy as np
from scipy import stats
import sqlite3
import sqlalchemy


# function to extract list of features ############################################################
def get_words(item):
    '''
    list all the unique words in a text
    '''
    # list all email ids
    regex_for_email_ids = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    email_ids = re.findall(regex_for_email_ids, item)
    # splitter with non-alphabetic chars, with the exception of @
    regex_for_splitter = r'[\W]+'
    splitter = re.compile(regex_for_splitter)
    # split text with splitter as separator
    words = [s.lower() for s in re.split(splitter, item) if len(s)>2] + email_ids
    # list unique words and assign count of 1 for each - as a series of word counts
    words_count = pd.Series(1, index = list(set(words)))
    words_count.index.name = 'Features'
    return words_count


# training function ###############################################################################
def train_example(cl):
    cl.train('Nobody owns the water.',['good'])
    cl.train('the quick rabbit jumps fences',['good'])
    cl.train('buy pharmaceuticals now',['bad'])
    cl.train('make quick money at the online casino',['bad'])
    cl.train('the quick brown fox jumps',['good'])


# basic classifier ################################################################################
class BasicClassifier:
    # init with feature_extraction_method and database db_name_table
    def __init__(self, get_features, db_name_table=None):
        self.df_feature_category_count = pd.DataFrame() # number of features by category
        self.ds_category_count = pd.Series().rename('N_Items') # number of items in each category
        self.get_features = get_features # function to extract features

    # increment the (feature,category) count
    def increment_feature_category_count(self, features_categories):
        self.df_feature_category_count = self.df_feature_category_count\
                                            .add(features_categories, fill_value=0)\
                                            .fillna(0)
        self.df_feature_category_count.index.name = 'Features'
        self.df_feature_category_count.columns.name = 'Categories'

    # increment the item count in a category
    def increment_category_count(self, categories):
        try:
            self.ds_category_count[categories] += 1
        except:
            for cat in categories:
                try:
                    self.ds_category_count[cat] += 1
                except:
                    self.ds_category_count[cat] = 1
                    self.ds_category_count.index.name = 'Categories'

    # train classifier given an item and category
    def train(self, item, categories):
        features_count = self.get_features(item)
        features_categories = pd.concat([features_count]*len(categories), axis=1)
        features_categories.columns = categories
        self.increment_feature_category_count(features_categories)
        self.increment_category_count(categories)

    # number of times a feature occurred in a category - (feature,category) value
    def feature_category_count(self, feature, category):
        try:
            return self.df_feature_category_count.ix[feature][category]
        except:
            return 0

    # total number of items in a category
    def category_count(self, category):
        try:
            return self.ds_category_count[category]
        except:
            return 0

    # total number of items
    def items_count(self):
        return self.category_count.sum()

    # list all categories
    def categories_list(self):
        return list(self.ds_category_count.index)

    # probability that a given feature will appear in an item belonging to given category
    # p(feature/category)
    def feature_category_prob(self, feature, category):
        try:
            return self.feature_category_count(feature, category)/self.category_count(category)
        except:
            return 0

    # weighted probability
    # weighted_p(feature/category)
    def feature_category_wghtprob(self, feature, category, probfunc, init_weight=1, init_prob=0.5):
        init_weight = pd.Series(init_weight,index=feature)
        init_prob = pd.Series(init_prob, index=feature)
        try:
            feature_prob = probfunc(feature, category)
            feature_count = self.df_feature_category_count.sum(axis=1)[feature]
            prob = feature_prob.apply(lambda x: (x*feature_count + init_weight*init_prob)\
                                                /(init_weight+feature_count) ,axis=0)
            return prob
        except:
            return init_prob


# Bernoulli Naive Bayesian Classifier #############################################################
class BernoulliNBclassifier(BasicClassifier):
    # prior
    # p(item/category)
    # probability that an item belongs to given category
    # p(item/category) = product[ p(feature/category) ] for each feature in item
    def get_item_category_prob(self, item, category):
        features_count = self.get_features(item)
        features = list(features_count.index)
        p_item_category = self.feature_category_wghtprob(features, category,
                                                         self.feature_category_prob).product()
        return p_item_category

