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
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import time

pd.set_option('display.max_columns',5)
pd.set_option('display.max_rows',24)
pd.set_option('expand_frame_repr',False)


# train it ##########################################################
def train_classifier(cl, X_train, y_train):
    for i, (rowidx, file_path) in enumerate(X_train.iteritems()):
        try:
            with open(file_path, 'r') as txt_file:
                txt = txt_file.read()
        except UnicodeDecodeError:
            continue
        category = y_train[rowidx]
        cl.train(txt, [category])
        if not bool((i + 1) % 100):
            print('Trained on', i + 1, 'files', flush=True)
    return


# predict_categories ##############################################################################
def predict_categories(cl, X_test):
    df_test = pd.DataFrame(index=X_test.index)
    df_test['Pred_One_Category'] = '-'
    df_test['Pred_Multi_Category'] = '-'
    df_test['Accuracy_One_Category'] = 0
    df_test['Accuracy_Multi_Category'] = 0
    for i, (rowidx, file_path) in enumerate(X_test.iteritems()):
        try:
            with open(file_path, 'r') as txt_file:
                txt = txt_file.read()
        except UnicodeDecodeError:
            continue
        p_categories, top_categories = cl.classify(txt, threshold=2)
        df_test.set_value(rowidx, 'Pred_Multi_Category', top_categories)
        df_test.set_value(rowidx, 'Pred_One_Category', top_categories[0])
        # if not bool((i + 1) % 100):
        #     print('Tested on', i + 1, 'files', flush=True)
    return df_test


# accuracy of prediction ##########################################################################
def prediction_accuracy(df_pred, y_test):
    df_pred['True_Category'] = y_test
    def check_single_accuracy(row):
        return int( row['True_Category'] == row['Pred_One_Category'] )
    def check_multi_accuracy(row):
        return int( row['True_Category'] in row['Pred_Multi_Category'] )
    df_pred['Accuracy_One_Category'] = df_pred.apply(check_single_accuracy, axis=1)
    df_pred['Accuracy_Multi_Category'] = df_pred.apply(check_multi_accuracy, axis=1)
    column_order = ['True_Category','Pred_One_Category','Accuracy_One_Category',
                    'Pred_Multi_Category','Accuracy_Multi_Category']
    print('\n')
    print(df_pred[['Accuracy_One_Category','Accuracy_Multi_Category']].sum()/len(df_pred),'\n')
    return df_pred[column_order]


# walk through and list files in DataSet folder ###################################################
def list_files_paths(dir_name):
    data_dir = './DataSets/' + dir_name
    df_list = []
    for dirpath,dirnames,filenames in os.walk(data_dir):
        df = pd.DataFrame(filenames,columns=['filename'])
        df['filepath'] = df['filename'].map(lambda fname: dirpath + '/' + fname)
        df['category'] = dirpath.split(sep='/')[-1]
        df_list.append(df)
    df_items_categories = pd.concat(df_list, axis=0, ignore_index=True)
    return df_items_categories


# train and test on datadir #######################################################################
def train_test_on_datadir(cl, dir_name='20_newsgroup', samplefrac=0.2, randstate=42, testsize=0.2):

    print('\nTotal number of items in persistent training data:', cl.ds_category_count.sum())
    print('Number of items in persistent training data, by category:')
    print(cl.ds_category_count)

    df_items_categories = list_files_paths(dir_name).sample(frac=samplefrac, replace=False,
                                                            random_state = randstate*42)
    X = df_items_categories['filepath']
    y = df_items_categories['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testsize,
                                                        random_state = randstate)
    n_train = X_train.size
    n_test = X_test.size
    print('\nTotal Sample Size:', n_train+n_test)
    print('Training Sample Size:', n_train)
    print('Testing Sample Size:', n_test)

    # train the classifier on training dataset
    print('\nTraining on new data ...')
    t1 = time.time()
    train_classifier(cl, X_train, y_train)
    t2 = time.time()
    print('\nFinished Training on {:0.0f} items in in {:0.0f} sec - {:0.0f} items per sec.'\
                .format(n_train, t2-t1, n_train/(t2-t1)) )
    print('\nTotal number of items used:', cl.ds_category_count.sum())
    print('Number of items trained, by category:')
    print(cl.ds_category_count)

    # test the classifier on testing dataset
    print('\nTesting ...')
    t1 = time.time()
    n_jobs = 4 # number of parallel jobs
    parallelizer = Parallel(n_jobs)
    parts_X_test = np.array_split(X_test,n_jobs)
    tasks_iterator = ( delayed(predict_categories)(cl, part_X) for part_X in  parts_X_test)
    list_df_test = parallelizer(tasks_iterator)
    df_prediction = pd.concat(list_df_test,axis=0)
    # df_test = predict_categories(cl,X_test)
    t2 = time.time()
    print('\nFinished Classification of {:0.0f} items in {:0.0f} sec - {:0.0f} items per sec.'\
                .format(n_test,t2-t1, n_test/(t2-t1)) )

    # find accuracy of prediction
    df_test = prediction_accuracy(df_prediction, y_test)
    # return accuracy and predictions
    return df_test


# train on sentences ##############################################################################
def train_on_sentences(cl):
    '''
    train on example texts
    '''
    cl.train('Nobody owns the water.',['good'])
    cl.train('the quick rabbit jumps fences',['good'])
    cl.train('buy pharmaceuticals now',['bad'])
    cl.train('make quick money at the online casino',['bad'])
    cl.train('the quick brown fox jumps',['good'])


# custom exception ################################################################################
class CustomException(Exception):
    pass


# function to extract list of features ############################################################
def get_words_emails(item):
    '''
    list all the words in a text
    '''
    # list all email ids
    regex_for_email_ids = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    email_ids = re.findall(regex_for_email_ids, item)
    # splitter with non-alphabetic chars, with the exception of @
    sub_item = re.sub('\W+\d+',' ', item)
    regex_for_splitter = r'[\W]+'
    splitter = re.compile(regex_for_splitter)
    words = [s.lower() for s in re.split(splitter, sub_item) if len(s)>2] + email_ids
    # list unique words and assign count of 1 for each - as a series of word counts
    words_count = pd.Series(1, index = list(set(words)))
    words_count.index.name = 'Features'
    return words_count


def get_words(item):
    '''
    list all the words in a text
    '''
    # splitter with non-alphabetic chars, with the exception of @
    sub_item = re.sub('\W+\d+',' ', item)
    regex_for_splitter = r'[\W]+'
    splitter = re.compile(regex_for_splitter)
    words = [s.lower() for s in re.split(splitter, sub_item) if len(s)>2]
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
            # categories_in = list(set(categories).intersection(set(self.ds_category_count.index)))
            # categories_ex = list(set(categories).difference(set(self.ds_category_count.index)))
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

    def load_data(self, file_name):
        store = pd.HDFStore(file_name)
        self.df_feature_category_count = store.df_feature_category_count
        self.ds_category_count = store.ds_category_count
        self.ds_category_thresholds = store.ds_category_thresholds
        store.close()

    def save_data(self, file_name):
        store = pd.HDFStore(file_name)
        store['df_feature_category_count'] = self.df_feature_category_count
        store['ds_category_count'] = self.ds_category_count
        store['ds_category_thresholds'] = self.ds_category_thresholds
        store.close()

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