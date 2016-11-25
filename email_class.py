# -*- coding: utf-8 -*-
"""
@author: srikant
"""
import re
import pandas as pd
import numpy as np
import sqlalchemy
import os
from scipy import stats


# custom exception ################################################################################
class CustomException(Exception):
    pass


# function to extract list of features ############################################################
def get_words(item):
    '''
    list all the words in a text
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


# training functions ###############################################################################
def train_example(cl):
    '''
    train on example texts
    '''
    cl.train('Nobody owns the water.',['good'])
    cl.train('the quick rabbit jumps fences',['good'])
    cl.train('buy pharmaceuticals now',['bad'])
    cl.train('make quick money at the online casino',['bad'])
    cl.train('the quick brown fox jumps',['good'])

def train_on_dir(dir_name):
    pass


# basic classifier ################################################################################
class BasicClassifier:
    '''
    basic classifier
    '''
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
        except (KeyError,ValueError):
            for cat in categories:
                try:
                    self.ds_category_count[cat] += 1
                except KeyError:
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
    def feature_category_count(self, features, categories):
        try:
            return self.df_feature_category_count.ix[features][categories]
        except KeyError:
            categories_inc = list(set(categories).intersection(set(self.ds_category_count.index)))
            categories_exc = list(set(categories).difference(set(self.ds_category_count.index)))
            df_count = pd.concat([self.df_feature_category_count.ix[features][categories_inc],
                                  pd.DataFrame(0,index=features,columns=categories_exc)],axis=1)\
                                .fillna(0)
            return df_count


    # total number of items in a category
    def category_count(self, categories):
        return self.ds_category_count[categories].fillna(0)

    # total number of items
    def items_count(self):
        return self.ds_category_count.sum()

    # list all categories
    def categories_list(self):
        return list(self.ds_category_count.index)

    # probability that a given feature will appear in an item belonging to given category
    # p(feature/category)
    def feature_category_prob(self, features, categories):
        prob = self.feature_category_count(features, categories)/self.category_count(categories)
        return prob.fillna(0)

    # weighted probability
    # weighted_p(feature/category)
    def feature_category_wghtprob(self, features, categories, pfunc, init_weight=1, init_prob=0.5):
        init_weight = pd.Series(init_weight,index=features)
        init_prob = pd.Series(init_prob, index=features)
        features_prob = pfunc(features, categories)
        features_count = self.df_feature_category_count.ix[features].sum(axis=1)
        prob = features_prob.apply(lambda x: (x*features_count + init_weight*init_prob)\
                                            /(init_weight+features_count))
        return prob.fillna(0)


# Bernoulli Naive Bayesian Classifier #############################################################
class BernoulliNBclassifier(BasicClassifier):
    '''
    bernoulli naive bayesian classifier
    '''
    def __init__(self,get_features):
        BasicClassifier.__init__(self,get_features)
        self.ds_category_thresholds = pd.Series().rename('Thresholds')

    # set thresholds
    def set_thresholds(self, categories, thresholds):
        self.ds_category_thresholds[categories] = thresholds

    # get thresholds
    def get_threshold(self, category):
        try:
            return self.ds_category_thresholds[category]
        except KeyError:
            self.ds_category_thresholds[category] = 1
            return self.ds_category_thresholds[category]

    # prior probability of p(item/category)
    # probability that an item belongs to given category
    # p(item/category) = product[ p(feature/category) ] for each feature in item
    def p_item_given_category(self, item, categories):
        features_count = self.get_features(item)
        features = list(features_count.index)
        p_item_categories = self.feature_category_wghtprob(features, categories,
                                                           self.feature_category_prob)\
                                                        .product()
        return p_item_categories

    # p(category/item) - pseudo probability ignoring p(item)
    # p(category/item) ~ p(item/category)*p(category)
    def p_category_given_item(self, item, categories):
        p_categories = self.category_count(categories)/self.items_count()
        p_item_categories = self.p_item_given_category(item, categories)
        p_categories_item = p_item_categories * p_categories
        return p_categories_item.fillna(0)

    # classify item
    def classify(self, item, threshold=1):
        categories = self.categories_list()
        if not categories:
            raise CustomException('No training data.')
        p_categories_item = self.p_category_given_item(item, categories)
        categories_max10_p = p_categories_item.nlargest(5).rename('p_Category')
        c_max = categories_max10_p.idxmax()
        p_max = categories_max10_p.iloc[0]
        self.set_thresholds(c_max, threshold)
        p_threshold = p_max/self.get_threshold(c_max)
        best_categories = list(categories_max10_p[categories_max10_p >= p_threshold].index)
        return categories_max10_p, best_categories


###################################################################################################