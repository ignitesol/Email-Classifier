# -*- coding: utf-8 -*-
"""
@author: srikant bondugula

"""
import pandas as pd
import numpy as np
import json
import nltk
import scipy
#import threading
#import pymongo
import sqlite3
import warnings
import pickle

STOP_WORDS = set(nltk.corpus.stopwords.words('english'))

warnings.simplefilter('error', RuntimeWarning)

# basic classifier ################################################################################
class BasicClassifier:
    '''
    basic classifier
    '''
    # init with feature_extraction_method and database db_name_table
    def __init__(self, user_id):
        #
        self.df_feature_category_count = pd.DataFrame(dtype='int32') # number of features by category
        self.ds_category_count = pd.Series(dtype='int32').rename('N_Items') # number of items in each category
        self.ds_category_ll_thresholds = pd.Series().rename('LL_Thresholds')
        self.ds_category_nb_thresholds = pd.Series().rename('NB_Thresholds')
        self.user_id = user_id
        #
        self.sqlite_db = './sqlite_db/'
        self.mongo_db = 'email_classifier_db'
        #
        self.df_feature_category_count.index.name = 'Features'
        self.ds_category_count.index.name = 'Categories'
        self.ds_category_ll_thresholds.index.name = 'Categories'
        self.ds_category_nb_thresholds.index.name = 'Categories'
        # self.lock = threading.RLock()

    # function to extract list of features
    def get_features(self, raw_txt):
        item = '\n'.join(raw_txt.split('\n')[1:])
        tokens = set(nltk.word_tokenize(item))
        # list all unique alphanumeric words
        tokens_alnum = [s.lower() for s in tokens if s.isalpha()]
        words_nostop = [s for s in tokens_alnum if s not in STOP_WORDS]
        # list all email ids
        # regex_for_email_ids = r'[a-zA-Z0-9_.-]+@[a-zA-Z0-9_.-]+\.[a-zA-Z]{2,3}'
        # email_ids = re.findall(regex_for_email_ids, item)
        # words += email_ids
        # list unique words and assign count of 1 for each - as a series of word counts
        words_count = pd.Series(1, index=set(words_nostop), dtype='int32')
        words_count.index.name = 'Features'
        return words_count

    # increment the (feature,category) count
    def increment_feature_category_count(self, features_categories):
        self.df_feature_category_count = self.df_feature_category_count\
                                            .add(features_categories, fill_value=0).fillna(0)

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

    # train classifier given an item and category
    def train(self, item, categories):
        # with self.lock:
        features_count = self.get_features(item)
        features_categories = pd.concat([features_count]*len(categories), axis=1)
        features_categories.columns = categories
        self.increment_feature_category_count(features_categories)
        self.increment_category_count(categories)

    # number of times a feature occurred in a category - (feature,category) value
    def feature_category_count(self, features, categories):
        df_count = self.df_feature_category_count.loc[features,categories]
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

    # save df_features_categories_count to db
    def update_db_features_categories_count(self, collection, new_df):
        #drop old data and insert new
        query = {'user_id': self.user_id}
        collection.delete_many(query)
        collection.insert_many(json.loads(new_df.to_json(orient='records')))
        ## bulk replace each row one by one, append if it doesnt exist
        # bulk_replace = collection.initialize_unordered_bulk_op()
        # for index, row in new_df.iterrows():
        #     query = {'user_id': self.user_id,
        #              'Features': row['Features']}
        #     bulk_replace.find(query).upsert().replace_one(json.loads(row.to_json()))
        # bulk_replace.execute()

    # save data to mongodb
    def save_data_to_mongodb(self, mongo_db_name):
        query = {'user_id': self.user_id}
        # save features_categories_count
        fcc_collection = mongo_db_name.feature_categories_count
        df_fcc = pickle.dumps(self.df_feature_category_count.to_sparse(fill_value=0))
        fcc_collection.replace_one(query,
                                  {'user_id': self.user_id,'df_feature_category_count': df_fcc},
                                  upsert = True)
        # save categories_count
        ds_cc = json.loads(self.ds_category_count.to_json())
        cc_collection = mongo_db_name.categories_count
        cc_collection.replace_one(query,
                                  {'user_id': self.user_id, 'ds_category_count': ds_cc},
                                  upsert = True)
        # save nb_category_thresholds
        ds_cnbt = json.loads(self.ds_category_nb_thresholds.to_json())
        cnbt_collection = mongo_db_name.category_nb_thresholds
        cnbt_collection.replace_one(query,
                                    {'user_id': self.user_id, 'ds_category_nb_thresholds': ds_cnbt},
                                    upsert = True)
        # save ll_category_thresholds
        ds_cllt = json.loads(self.ds_category_ll_thresholds.to_json())
        cllt_collection = mongo_db_name.category_ll_thresholds
        cllt_collection.replace_one(query,
                                    {'user_id': self.user_id, 'ds_category_ll_thresholds': ds_cllt},
                                    upsert = True)

    # load data from mongodb
    def load_data_from_mongodb(self, mongo_db_name):
        query = {'user_id': self.user_id}
        # load features_categories_count
        fcc_collection = mongo_db_name.feature_categories_count
        try:
            df_fcc = pickle.loads(fcc_collection.find_one(query)['df_feature_category_count'])
        except TypeError:
            raise Exception('There is no training data for this user')
        df_fcc.index.name = 'Features'
        self.df_feature_category_count = df_fcc.to_dense()
        # load category_count
        cc_collection = mongo_db_name.categories_count
        ds_cc = pd.Series(cc_collection.find_one(query)['ds_category_count'])
        ds_cc.index.name = 'Categories'
        self.ds_category_count = ds_cc.rename('N_Items')
        # load nb_category_thresholds
        cnbt_collection = mongo_db_name.category_nb_thresholds
        ds_cnbt = pd.Series(cnbt_collection.find_one(query)['ds_category_nb_thresholds'])
        ds_cnbt.index.name = 'Categories'
        self.ds_category_nb_thresholds = ds_cnbt.rename('NB_Thresholds')
        # load ll_category_thresholds
        cllt_collection = mongo_db_name.category_ll_thresholds
        ds_cllt = pd.Series(cllt_collection.find_one(query)['ds_category_ll_thresholds'])
        ds_cllt.index.name = 'Categories'
        self.ds_category_ll_thresholds = ds_cllt.rename('LL_Thresholds')
        # check if df_feature_category_count is empty
        if self.df_feature_category_count.empty:
            raise Exception('There is no training data for this user')


    # save data to sqlite
    def save_data_to_sqlite(self):
        db_filename = self.sqlite_db + str(self.user_id) + '.sqlite'
        db_conn = sqlite3.connect(db_filename)
        self.ds_category_count.to_sql('ds_category_count', db_conn, if_exists='replace')
        self.df_feature_category_count.to_sql('df_feature_category_count', db_conn,
                                              if_exists='replace')
        self.ds_category_ll_thresholds.to_sql('ds_category_ll_thresholds', db_conn,
                                              if_exists='replace')
        self.ds_category_nb_thresholds.to_sql('ds_category_nb_thresholds', db_conn,
                                              if_exists='replace')
        db_conn.close()
        return

    # load data from sqlite
    def load_data_from_sqlite(self):
        db_filename = self.sqlite_db + str(self.user_id) + '.sqlite'
        db_conn = sqlite3.connect(db_filename)
        self.df_feature_category_count = pd.read_sql("SELECT * FROM df_feature_category_count",
                                                     db_conn, index_col='Features')
        self.ds_category_count = pd.read_sql("SELECT * FROM ds_category_count", db_conn,
                                             index_col='Categories').iloc[:,0]
        self.ds_category_ll_thresholds = pd.read_sql("SELECT * FROM ds_category_ll_thresholds",
                                                     db_conn, index_col='Categories').iloc[:,0]
        self.ds_category_nb_thresholds = pd.read_sql("SELECT * FROM ds_category_nb_thresholds",
                                                     db_conn, index_col='Categories').iloc[:,0]
        self.ds_category_nb_thresholds.name = 'NB_Thresholds'
        self.ds_category_ll_thresholds.name = 'LL_Thresholds'
        self.ds_category_count.name = 'N_Items'
        db_conn.close()
        return


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
        features_count = self.df_feature_category_count.loc[features,:].sum(axis=1)
        prob = features_prob.apply(lambda x:(x*features_count+init_weight*init_prob)\
                                            /(init_weight+features_count))
        return prob.fillna(0)


# Bernoulli Naive Bayesian Classifier #############################################################
class BernoulliNBclassifier(BasicClassifier):
    '''
    bernoulli naive bayesian classifier
    '''
    def __init__(self, user_id):
        BasicClassifier.__init__(self, user_id)

    # set thresholds
    def set_thresholds(self, categories, thresholds):
        # with self.lock:
        self.ds_category_nb_thresholds[categories] = thresholds

    # get thresholds
    def get_threshold(self, category):
        try:
            return self.ds_category_nb_thresholds[category]
        except KeyError:
            # with self.lock:
            self.set_thresholds(category,2)
            return self.ds_category_nb_thresholds[category]

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
    def classify(self, item, n_multi, default='Default_Category'):
        categories = self.categories_list()
        p_categories_item = self.p_category_given_item(item, categories)
        categories_top_p = p_categories_item.nlargest(n_multi).rename('p_Category')
        c_max = categories_top_p.idxmax()
        p_max = categories_top_p.iloc[0]
        thresholds = {c:self.get_threshold(c) for c in categories_top_p.index}
        p_threshold = p_max/thresholds[c_max]
        best_categories = list(categories_top_p[categories_top_p >= p_threshold].index)
        best_categories = best_categories + [default]
        return best_categories[:n_multi]


# Log-Likelihood Classifier #######################################################################
class LogLikelihoodClassifier(BasicClassifier):
    '''
    Log-Likelihood Classifier
    '''
    def __init__(self, user_id):
        BasicClassifier.__init__(self, user_id)

    # set thresholds
    def set_thresholds(self, categories, thresholds):
        # with self.lock:
        self.ds_category_ll_thresholds[categories] = thresholds

    # get thresholds
    def get_threshold(self, category):
        try:
            return self.ds_category_ll_thresholds[category]
        except KeyError:
            # with self.lock:
            self.set_thresholds(category, 0)
            return self.ds_category_ll_thresholds[category]

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
        try:
            log_likelihood = -2*np.log(p_categories)
        except RuntimeWarning:
            log_likelihood = -2*np.log(np.nextafter(p_categories, 1))
        chi2sf_score = log_likelihood.map(lambda x : scipy.stats.chi2.sf(x,degf))
        return chi2sf_score

    # classify item
    def classify(self, item, n_multi, default='Default_Category'):
        all_categories = self.categories_list()
        p_categories_item = self.log_likelihood_score(item, all_categories)
        categories_top_p = p_categories_item.nlargest(n_multi).rename('p_Category')
        thresholds = [self.get_threshold(t) for t in categories_top_p.index]
        threshold_top_p = pd.Series(thresholds, index = categories_top_p.index)
        best_categories = list(categories_top_p[categories_top_p >= threshold_top_p].index)
        best_categories = best_categories + [default]
        return best_categories[:n_multi]


###################################################################################################
