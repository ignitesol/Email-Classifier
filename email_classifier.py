# -*- coding: utf-8 -*-
"""
@author: srikant
"""
import re
import pandas as pd
import numpy as np
import sqlalchemy
from scipy import stats
import nltk


pd.set_option('display.max_columns',5)
pd.set_option('display.max_rows',24)
pd.set_option('expand_frame_repr',False)

# Globals #########################################################################################
STOP_WORDS = set(nltk.corpus.stopwords.words('english'))


# chi2sf function #################################################################################
# currently using scipi.stats.chi2.sf
def chi2sf(chi,df):
    m = chi/2
    sum = term = np.exp(-m)
    for i in range(1,df//2):
        term *= m/i
        sum += term
    return min(sum,1)


# custom exception ################################################################################
class CustomException(Exception):
    pass


# function to extract list of features ############################################################
def get_unique_tokens(item):
    # list all unique alphanumeric words
    tokens = set(nltk.word_tokenize(item))
    tokens_alnum = [s.lower() for s in tokens if s.isalpha()]
    words = [s for s in tokens_alnum if s not in STOP_WORDS]
    # list all email ids
    # regex_for_email_ids = r'[a-zA-Z0-9_.-]+@[a-zA-Z0-9_.-]+\.[a-zA-Z]{2,3}'
    # email_ids = re.findall(regex_for_email_ids, item)
    # words += email_ids
    # list unique words and assign count of 1 for each - as a series of word counts
    words_count = pd.Series(1, index = list(set(words)))
    words_count.index.name = 'Features'
    return words_count


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
        self.ds_category_thresholds = pd.Series().rename('Thresholds')

    # increment the (feature,category) count
    def increment_feature_category_count(self, features_categories):
        self.df_feature_category_count = self.df_feature_category_count\
                                            .add(features_categories, fill_value=0).fillna(0)
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
        df_count = self.df_feature_category_count.ix[features][categories]
        df_count.fillna(0, inplace = True)
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

    # load data
    def load_data(self, file_name):
        store = pd.HDFStore(file_name)
        self.df_feature_category_count = store.df_feature_category_count
        self.ds_category_count = store.ds_category_count
        store.close()

    # save data
    def save_data(self, file_name):
        store = pd.HDFStore(file_name)
        store['df_feature_category_count'] = self.df_feature_category_count
        store['ds_category_count'] = self.ds_category_count
        store.close()

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

    # set thresholds
    def set_thresholds(self, categories, thresholds):
        self.ds_category_thresholds[categories] = thresholds

    # get thresholds
    def get_threshold(self, category):
        try:
            return self.ds_category_thresholds[category]
        except KeyError:
            self.ds_category_thresholds[category] = 2
            return self.ds_category_thresholds[category]

    # prior probability of p(item/category)
    # probability that an item belongs to given category
    # p(item/category) = product[ p(feature/category) ] for each feature in item
    def p_item_given_category(self, item, categories):
        features_count = self.get_features(item)
        features = list(features_count.index)
        p_item_categories = self.feature_category_wghtprob(features, categories,
                                                           self.feature_category_prob).product()
        return p_item_categories

    # p(category/item) - pseudo probability ignoring p(item)
    # p(category/item) ~ p(item/category)*p(category)
    def p_category_given_item(self, item, categories):
        p_categories = self.category_count(categories)/self.items_count()
        p_item_categories = self.p_item_given_category(item, categories)
        p_categories_item = p_item_categories * p_categories
        return p_categories_item.fillna(0)

    # classify item
    def classify(self, item, n_multi):
        categories = self.categories_list()
        if not categories:
            raise CustomException('No training data.')
        p_categories_item = self.p_category_given_item(item, categories)
        categories_max10_p = p_categories_item.nlargest(n_multi).rename('p_Category')
        c_max = categories_max10_p.idxmax()
        p_max = categories_max10_p.iloc[0]
        p_threshold = p_max/self.get_threshold(c_max)
        best_categories = list(categories_max10_p[categories_max10_p >= p_threshold].index)
        return best_categories


# Log-Likelihood Classifier #######################################################################
class LogLikelihoodClassifier(BasicClassifier):
    '''
    Log-Likelihood Classifier
    '''
    def __init__(self,get_features):
        BasicClassifier.__init__(self,get_features)

    # set thresholds
    def set_thresholds(self, categories, thresholds):
        self.ds_category_thresholds[categories] = thresholds

    # get thresholds
    def get_threshold(self, category):
        try:
            return self.ds_category_thresholds[category]
        except KeyError:
            self.ds_category_thresholds[category] = 0
            return self.ds_category_thresholds[category]

    # probability an item with a particular feature belongs to given category
    # p(category/feature)
    def p_category_given_features(self, features, category):
        all_categories = self.categories_list()
        p_features_all_categories = self.feature_category_prob(features, all_categories)
        p_features_given_category = p_features_all_categories[category]
        sump_features_all_categories = p_features_all_categories.sum(axis=1)
        p_category = p_features_given_category.apply(lambda x: x/sump_features_all_categories)
        return p_category.fillna(0)

    # probability that an item belongs to given category
    # p(category/item) = product[ p(category/feature) ] for each feature in item
    def log_likelihood_score(self, item, categories):
        features_count = self.get_features(item)
        features = list(features_count.index)
        degf = 2*len(features)
        p_categories = self.feature_category_wghtprob(features, categories,
                                                      self.p_category_given_features).product()
        log_likelihood = -2*np.log(p_categories)
        chi2sf_score = log_likelihood.map(lambda x: stats.chi2.sf(x,degf))
        return chi2sf_score

    # classify item
    def classify(self, item, n_multi):
        all_categories = self.categories_list()
        if not all_categories:
            raise CustomException('No training data.')
        p_categories_item = self.log_likelihood_score(item, all_categories)
        categories_top_p = p_categories_item.nlargest(n_multi).rename('p_Category')
        thresholds = [self.get_threshold(t) for t in categories_top_p.index]
        threshold_top_p = pd.Series(thresholds, index = categories_top_p.index)
        best_categories = list(categories_top_p[categories_top_p >= threshold_top_p].index)
        return best_categories


###################################################################################################
